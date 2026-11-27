"""
SDEdit (Meng et al., ICLR 2022) を ILVR のループに足したもの。SDEdit・ILVR・マスク合成をそれぞれ独立に使える。

ループは ilvr_eye.ilvr_sample と同じ計算・同じ乱数の消費順で、SDEdit とマスク合成を無効にすると
ilvr_eye.ilvr_sample と同じ結果になる (diagnostics/verify_sdedit.py の回帰確認)。

ステップ番号 i はすべて間引き後 (SpacedDiffusion) の番号 0..T-1 (T = diffusion.num_timesteps)。
diffusion.q_sample(x, i) は間引き後の alphas_cumprod[i] (= 元の過程の alphas_cumprod[timestep_map[i]]) を使い、
モデルには p_mean_variance の中で timestep_map[i] が渡される。番号 i の p_sample は x_i から x_{i-1} を作る。

1 ステップの処理順: p_sample -> ILVR (有効なら) -> マスク合成 (有効なら)。
  ILVR        i > range_t のステップで x <- x - up(down(x)) + up(down(y))。
              y は t_ref="t" (既定、公式と同じ) なら q_sample(ref, i)、"t-1" なら q_sample(ref, i-1) (i=0 では ref)
  マスク合成  x <- m * x + (1 - m) * q_sample(x0, i-1)。最後のステップ (i=0) の後は x <- m * x + (1 - m) * x0。
              m は 1 (白) が編集する領域
SDEdit: t0 ステップ (1..T) だけ逆過程を回す。x = q_sample(x_init, t0-1) から始めて、i = t0-1, ..., 0 を回す。
  t0 = T なら、純粋なノイズの代わりに q_sample(x_init, T-1) から始める通常の長さのサンプリングになる。
  t0 は割合 (sdedit_t0) か、開始時の √ᾱ (sdedit_t0_from_sqrt_alpha) で決める。
"""

import numpy as np
import torch as th

T_REF_CHOICES = ("t", "t-1")


def sdedit_t0(t0_ratio, num_timesteps):
    """t0 = max(1, int(t0_ratio * T)) を 1..T に収めて返す (回すステップ数。開始の番号は t0 - 1)。"""
    return min(max(1, int(t0_ratio * num_timesteps)), num_timesteps)


def sdedit_t0_from_sqrt_alpha(diffusion, sqrt_alpha):
    """√ᾱ_i が sqrt_alpha に最も近い番号 i を開始の番号にする t0 (= i + 1) を返す (同じ近さなら小さい i)。"""
    d = np.abs(np.sqrt(diffusion.alphas_cumprod) - sqrt_alpha)
    return int(np.argmin(d)) + 1


def start_info(diffusion, start):
    """開始の番号 start (間引き後) の、元の過程の t、√ᾱ、SNR = ᾱ / (1 - ᾱ)。"""
    a = float(diffusion.alphas_cumprod[start])
    t_orig = int(diffusion.timestep_map[start]) if hasattr(diffusion, "timestep_map") else start
    return {"start_index": start, "t_original": t_orig, "sqrt_alpha_bar": a ** 0.5, "snr": a / (1 - a)}


def stop_ratio_to_range_t(ilvr_stop_ratio, num_timesteps):
    """「i >= int(ratio * T) の間だけ ILVR」を、既存の判定 i > range_t の range_t に直す。"""
    return int(ilvr_stop_ratio * num_timesteps) - 1


def sdedit_ilvr_sample(diffusion, model, x0, resizers=None, range_t=0, t_ref="t", rng=None, t0=None,
                       x_init=None, mask=None, x_blend=None, clip_denoised=True, intermediate=None,
                       intermediate_every=10, progress=False):
    """
    x0 ([N x 3 x H x W], [-1, 1]): 入力画像。ILVR の参照画像と、マスク合成の編集しない領域に使う。
    x_blend: マスク合成の編集しない領域に使う画像 (省略すると x0)。加工した入力を x0 に渡し、マスクの外は
        元の入力のままにしたいときに使う。
    t0: None なら純粋なノイズ x_T から (番号 T-1 から) 始める。整数 (1..T) なら SDEdit として
        q_sample(x_init, t0-1) から t0 ステップ回す。x_init は省略すると x0 (繰り返しの 2 回目以降に前回の出力を渡す)。
    resizers: (down, up) か None (ILVR を使わない)。
    mask ([N x 1 x H x W], [0, 1]): None ならマスク合成をしない。
    rng: shape -> 標準正規乱数。省略するとグローバルな乱数。
    intermediate: list を渡すと (名前, 画像 [N x 3 x H x W]) を追加する: 開始時のノイズ画像と、
        intermediate_every ステップおき (と最後) の pred_xstart。
    """
    assert t_ref in T_REF_CHOICES, t_ref
    shape = tuple(x0.shape)
    device = x0.device
    num_timesteps = diffusion.num_timesteps
    if rng is None:
        def rng(shape):
            return th.randn(*shape, device=device)

    if t0 is None:
        start = num_timesteps - 1
        img = rng(shape)  # ilvr_eye.ilvr_sample と同じく、最初に初期ノイズを引く
    else:
        assert 1 <= t0 <= num_timesteps, (t0, num_timesteps)
        start = t0 - 1
        x_init = x0 if x_init is None else x_init
        img = diffusion.q_sample(x_init, th.full((shape[0],), start, device=device, dtype=th.long), rng(shape))
    if intermediate is not None:
        intermediate.append((f"start_i{start:03d}", img.clone()))

    indices = list(range(start + 1))[::-1]
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices, leave=False)
    for i in indices:
        t = th.full((shape[0],), i, device=device, dtype=th.long)
        with th.no_grad():
            # diffusion.p_sample と同じ計算 (乱数だけ rng から引く。t == 0 でも乱数は引き、0 を掛ける)
            out = diffusion.p_mean_variance(model, img, t, clip_denoised=clip_denoised, model_kwargs={})
            nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)
            x = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * rng(shape)
            if resizers is not None and i > range_t:
                down, up = resizers
                eps = rng(shape)
                if t_ref == "t":
                    y = diffusion.q_sample(x0, t, eps)
                elif i == 0:
                    y = x0
                else:
                    y = diffusion.q_sample(x0, t - 1, eps)
                x = x - up(down(x)) + up(down(y))
            if mask is not None:
                xb = x0 if x_blend is None else x_blend
                y_m = xb if i == 0 else diffusion.q_sample(xb, t - 1, rng(shape))
                x = mask * x + (1 - mask) * y_m
        if intermediate is not None and (i % intermediate_every == 0 or i == start):
            intermediate.append((f"pred_x0_i{i:03d}", out["pred_xstart"].clone()))
        img = x
    return img


def sdedit_repeat(diffusion, model, x0, repeats=1, t0=None, **kwargs):
    """
    SDEdit を repeats 回繰り返す。2 回目以降は前回の出力を q_sample(., t0-1) で再びノイズ化して始める。
    ILVR の参照画像とマスク合成の x0 は、どの回も元の入力のまま。乱数は同じ rng から続けて引く。
    intermediate を渡すと、名前の先頭に k<回>_ を付ける (repeats > 1 のときだけ)。
    """
    assert repeats >= 1
    assert repeats == 1 or t0 is not None, "繰り返しは SDEdit (t0 の指定) のときだけ"
    inter = kwargs.pop("intermediate", None)
    x = None
    for k in range(repeats):
        buf = [] if inter is not None else None
        x = sdedit_ilvr_sample(diffusion, model, x0, t0=t0, x_init=x, intermediate=buf, **kwargs)
        if inter is not None:
            inter.extend(((f"k{k}_{n}" if repeats > 1 else n), v) for n, v in buf)
    return x


def gaussian_feather(mask, radius):
    """mask ([N x 1 x H x W]) の境界を、半径 radius px のガウシアン (sigma = radius / 2、±2 sigma で打ち切り) でぼかす。"""
    if radius <= 0:
        return mask
    from scipy.ndimage import gaussian_filter

    m = mask.cpu().numpy()
    out = np.stack([gaussian_filter(mi[0], sigma=radius / 2, truncate=2.0, mode="nearest")[None] for mi in m])
    return th.from_numpy(out).to(mask.device, mask.dtype)


def color_fix(out, inp, sigma=8.0, mask=None):
    """
    CIELAB の a*, b* の低周波だけを入力のものに置き換える: ab <- out_ab - G(out_ab) + G(in_ab)。L* はそのまま。
    G は sigma px のガウシアン (端は反射)。out, inp: [N x 3 x H x W] in [-1, 1]。
    mask ([N x 1 x H x W]) を渡すと、補正の後に out = m * out + (1 - m) * inp をかけ直し、マスクの外を入力に戻す。
    """
    from scipy.ndimage import gaussian_filter
    from skimage.color import lab2rgb, rgb2lab

    def lab(x):
        return rgb2lab(((x.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).cpu().double().numpy())

    lo, li = lab(out), lab(inp)
    res = []
    for o, n in zip(lo, li):
        o = o.copy()
        for c in (1, 2):
            o[..., c] = o[..., c] - gaussian_filter(o[..., c], sigma, mode="reflect") \
                + gaussian_filter(n[..., c], sigma, mode="reflect")
        res.append(np.clip(lab2rgb(o), 0, 1))
    res = th.from_numpy(np.stack(res)).permute(0, 3, 1, 2).to(out.device, out.dtype) * 2 - 1
    if mask is not None:
        res = mask * res + (1 - mask) * inp
    return res
