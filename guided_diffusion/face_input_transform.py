"""
SDEdit / ILVR に渡す前の顔全体の入力の加工 (scripts/sdedit_sample.py のオプション)。どれも既定では無効。

抽象化 (実写の細かい模様を消す。Winnemöller et al. 2006 "Real-Time Video Abstraction" の方式):
  bilateral   双方向フィルタを iters 回かける (CIELAB で。空間の sigma_s px、色の sigma_r は ΔE の単位)。
              肌の質感・しわ・ひげなどの細かい模様を消し、輪郭や色の境界は残す
  quantize    L* を levels 段に滑らかに段階化する (tanh で境界をぼかす。sharpness が大きいほど段がはっきりする)
  xdog        XDoG (Winnemöller et al. 2012) で輝度の輪郭線を求め、strength の濃さで線を描き足す
ストローク化 (SDEdit の論文の「ストローク画 -> 画像」のように、均一な色の塗りの集まりにする):
  generic     SLIC の超画素に分け、平均色の差が merge (ΔE) 未満の隣どうしを近い順にまとめ、領域ごとの平均色で塗る
  parts       顔パーシングのラベル (scripts/parse_faces.py) の部位ごとに、その部位の平均色で塗る。
              鼻・耳・眼鏡は肌に、眉は髪の色に、目は髪の色 (暗い色) で塗る。shade_levels >= 2 なら、肌・首・髪・服を
              ぼかした L* の分位点で shade_levels 段に分け、段ごとの平均色で塗る (アニメのセル塗りの影)。
              eye_scale_w / eye_scale_h で目の領域を (目ごとの重心を中心に) 横・縦に広げる。
              eye_style="iris" なら、目を白目・瞳 (重心を中心にした縦長の楕円を iris_rgb の一定の色で)・
              上まぶたの線 (領域の上 eyelid 割を髪の色で) に塗り分ける (ビューが違っても瞳の色を同じにするため)
色の写像: ilvr_ref_transform.ColorTransfer (CIELAB の統計を実写の顔からアニメ顔のデータへ)。前景だけに適用する。
適用の順: 抽象化 (bilateral -> ストローク化 -> quantize -> xdog) -> 色の写像。背景 (前景マスクの外) はどれでも元のまま。
"""

import math

import numpy as np
import torch as th
import torch.nn.functional as F
from skimage.color import lab2rgb, rgb2lab

FG_THRESH = 0.94  # [0, 1] でこれより暗いチャネルがある画素を前景とする (背景除去済みの白背景)


def fg_mask(x, dilate=2):
    """x: [N x 3 x H x W] in [-1, 1] -> [N x 1 x H x W] (0/1 の float)"""
    m = (((x + 1) / 2).min(1, keepdim=True).values < FG_THRESH).float()
    if dilate > 0:
        m = F.max_pool2d(m, 2 * dilate + 1, stride=1, padding=dilate)
    return m


def to_lab(x):
    """[N x 3 x H x W] in [-1, 1] -> torch [N x 3 x H x W] (CIELAB, float64)"""
    lab = rgb2lab(((x.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).double().cpu().numpy())
    return th.from_numpy(lab).permute(0, 3, 1, 2)


def from_lab(lab):
    rgb = np.stack([lab2rgb(o) for o in lab.permute(0, 2, 3, 1).numpy()])
    return th.from_numpy(rgb).float().permute(0, 3, 1, 2) * 2 - 1


def bilateral_lab(lab, sigma_s=3.0, sigma_r=8.0, iters=3, device="cpu"):
    """CIELAB の画像 [N x 3 x H x W] に双方向フィルタを iters 回かける (窓の半径は 2 sigma_s)。"""
    r = max(1, int(math.ceil(2 * sigma_s)))
    k = 2 * r + 1
    yy, xx = th.meshgrid(th.arange(-r, r + 1), th.arange(-r, r + 1), indexing="ij")
    w_s = th.exp(-(xx ** 2 + yy ** 2) / (2 * sigma_s ** 2)).reshape(1, 1, k * k, 1).to(device)
    out = []
    for xi in lab:  # メモリを抑えるため 1 枚ずつ
        x = xi[None].float().to(device)
        _, c, h, w = x.shape
        for _ in range(iters):
            pad = F.pad(x, (r, r, r, r), mode="reflect")
            patches = F.unfold(pad, k).view(1, c, k * k, h * w)
            center = x.view(1, c, 1, h * w)
            d2 = ((patches - center) ** 2).sum(1, keepdim=True)
            wgt = w_s * th.exp(-d2 / (2 * sigma_r ** 2))
            x = ((patches * wgt).sum(2) / wgt.sum(2)).view(1, c, h, w)
        out.append(x.double().cpu())
    return th.cat(out)


def quantize_l(lab, levels=8, sharpness=3.0):
    """L* (0〜100) を levels 段に滑らかに段階化する (Winnemöller 2006 の式)。"""
    out = lab.clone()
    L = lab[:, 0]
    dq = 100.0 / levels
    nearest = th.floor(L / dq) * dq + dq / 2
    out[:, 0] = nearest + dq / 2 * th.tanh(sharpness * (L - nearest) / dq)
    return out


def xdog_lines(lab, sigma=1.0, k=1.6, p=20.0, eps=0.0, phi=10.0):
    """L* から XDoG の線 (0 = 線、1 = 線なし) [N x 1 x H x W] を求める。"""
    L = (lab[:, :1] / 100.0).float()

    def blur(t, s):
        r = max(1, int(math.ceil(3 * s)))
        g = th.exp(-th.arange(-r, r + 1, dtype=th.float32) ** 2 / (2 * s ** 2))
        g = g / g.sum()
        t = F.conv2d(F.pad(t, (r, r, 0, 0), mode="reflect"), g.view(1, 1, 1, -1))
        return F.conv2d(F.pad(t, (0, 0, r, r), mode="reflect"), g.view(1, 1, -1, 1))

    d = (1 + p) * blur(L, sigma) - p * blur(L, sigma * k)
    e = th.where(d >= eps, th.ones_like(d), 1 + th.tanh(phi * (d - eps)))
    return e.clamp(0, 1).double()


STROKE_CHOICES = ("none", "generic", "parts")
# parts の部位 (CelebAMask-HQ の 19 クラス。scripts/parse_faces.py の docstring)。塗る順 (後のものが上)
PART_GROUPS = (
    ("other", (0,)),          # 前景だがパーシングで背景になった画素
    ("cloth", (15, 16)),
    ("neck", (14,)),
    ("skin", (1, 6, 7, 8, 9, 10)),
    ("mouth", (11, 12, 13)),
    ("eye", (4, 5)),
    ("brow", (2, 3)),
    ("hair", (17, 18)),
)
SHADED_PARTS = ("cloth", "neck", "skin", "hair")
EYE_STYLES = ("dark", "iris")
SCLERA_LAB = np.array([95.0, 0.0, 2.0])


def _merge_mean_color(g, src, dst):
    g.nodes[dst]["total color"] += g.nodes[src]["total color"]
    g.nodes[dst]["pixel count"] += g.nodes[src]["pixel count"]
    g.nodes[dst]["mean color"] = g.nodes[dst]["total color"] / g.nodes[dst]["pixel count"]


def _weight_mean_color(g, src, dst, n):
    return {"weight": np.linalg.norm(g.nodes[dst]["mean color"] - g.nodes[n]["mean color"])}


def stroke_generic(lab, fg, n_segments=200, compactness=10.0, merge=8.0, edge=2):
    """
    lab: [N x 3 x H x W] (CIELAB)、fg: [N x 1 x H x W] -> 前景を領域ごとの平均色で塗った lab
    SLIC の超画素を、平均色の差 (ΔE) が merge 未満の隣どうしから順に、平均色を更新しながらまとめる
    (skimage の merge_hierarchical。閾値で一度に切ると、近い色が連鎖して顔が髪に吸収されるため)。
    前景の縁 edge px (白背景と混ざった画素) は分割に使わず、内側の最も近い領域に含める。
    """
    from scipy.ndimage import binary_erosion, distance_transform_edt
    from skimage import graph
    from skimage.segmentation import slic

    out = lab.clone()
    for k in range(lab.shape[0]):
        L = lab[k].permute(1, 2, 0).numpy()
        m = fg[k, 0].numpy() > 0.5
        inner = binary_erosion(m, iterations=edge) if edge > 0 else m
        if not inner.any():
            continue
        seg = slic(L, n_segments=n_segments, compactness=compactness, convert2lab=False, start_label=1,
                   mask=inner, channel_axis=-1)
        rag = graph.rag_mean_color(L, seg, mode="distance")
        seg = graph.merge_hierarchical(seg, rag, thresh=merge, rag_copy=False, in_place_merge=True,
                                       merge_func=_merge_mean_color, weight_func=_weight_mean_color)
        seg = np.where(inner, seg + 1, 0)  # 0 を内側の外にする (merge_hierarchical は番号を振り直す)
        _, (iy, ix) = distance_transform_edt(seg == 0, return_indices=True)
        seg = np.where(m, seg[iy, ix], 0)
        o = L.copy()
        for lbl in np.unique(seg[inner]):
            o[seg == lbl] = L[(seg == lbl) & inner].mean(0)
        out[k] = th.from_numpy(o).permute(2, 0, 1)
    return out


def _scale_region(mask, sw, sh):
    """bool の領域 [H x W] を重心を中心に横 sw 倍・縦 sh 倍に広げる (逆写像の最近傍)"""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0 or (sw == 1 and sh == 1):
        return mask
    cy, cx = ys.mean(), xs.mean()
    h, w = mask.shape
    yy, xx = np.mgrid[0:h, 0:w]
    sy = np.clip(np.rint(cy + (yy - cy) / sh).astype(int), 0, h - 1)
    sx = np.clip(np.rint(cx + (xx - cx) / sw).astype(int), 0, w - 1)
    return mask[sy, sx]


def _paint_iris(o, eye, iris_lab, lid_lab, eyelid=0.25):
    """目の領域 eye (bool) を、白目・瞳 (縦長の楕円)・上まぶたの線に塗り分けて o (H x W x 3, Lab) に書く"""
    ys, xs = np.nonzero(eye)
    if len(ys) == 0:
        return
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1
    cy, cx = ys.mean(), xs.mean()
    ry, rx = 0.6 * h, min(0.35 * w, 0.55 * h)
    yy, xx = np.mgrid[0:eye.shape[0], 0:eye.shape[1]]
    o[eye] = SCLERA_LAB
    o[eye & (((yy - cy) / max(ry, 0.5)) ** 2 + ((xx - cx) / max(rx, 0.5)) ** 2 <= 1)] = iris_lab
    o[eye & (yy < ys.min() + eyelid * h)] = lid_lab


def stroke_parts(lab, fg, labels, shade_levels=1, eye_scale_w=1.0, eye_scale_h=1.0, shade_sigma=3.0,
                 eye_style="dark", iris_rgb=(90, 60, 50)):
    """
    lab: [N x 3 x H x W] (CIELAB)、fg: [N x 1 x H x W]、labels: [N x H x W] (0〜18 の整数)
    -> 前景を部位ごとの平均色で塗った lab (背景はそのまま)
    """
    from scipy.ndimage import gaussian_filter

    assert eye_style in EYE_STYLES, eye_style
    iris_lab = rgb2lab(np.asarray(iris_rgb, dtype=np.float64).reshape(1, 1, 3) / 255.0)[0, 0]
    out = lab.clone()
    for k in range(lab.shape[0]):
        L = lab[k].permute(1, 2, 0).numpy()
        m = fg[k, 0].numpy() > 0.5
        lb = labels[k].numpy()
        regions = {name: np.isin(lb, ids) & m for name, ids in PART_GROUPS}
        eyes = [_scale_region((lb == c) & m, eye_scale_w, eye_scale_h) & m for c in (4, 5)]
        regions["eye"] = eyes[0] | eyes[1]
        means = {n: L[r].mean(0) for n, r in regions.items() if r.any()}
        dark = means.get("hair", np.array([20.0, 0.0, 0.0]))
        means["brow"] = dark
        means["eye"] = dark
        # 段の分け方を決める L* は、部位の中だけで正規化してぼかす (隣の部位の明るさが混ざらないように)
        o = L.copy()
        for name, _ in PART_GROUPS:
            r = regions[name]
            if not r.any():
                continue
            if shade_levels >= 2 and name in SHADED_PARTS and r.sum() >= 4 * shade_levels:
                rf = r.astype(np.float64)
                num = gaussian_filter(L[..., 0] * rf, shade_sigma)
                den = gaussian_filter(rf, shade_sigma)
                ls = num / np.maximum(den, 1e-6)
                qs = np.quantile(ls[r], np.linspace(0, 1, shade_levels + 1)[1:-1])
                band = np.digitize(ls, qs)
                for b in range(shade_levels):
                    rb = r & (band == b)
                    if rb.any():
                        o[rb] = L[rb].mean(0)
            elif name == "eye" and eye_style == "iris":
                for e in eyes:
                    _paint_iris(o, e, iris_lab, dark)
            else:
                o[r] = means[name]
        out[k] = th.from_numpy(o).permute(2, 0, 1)
    return out


def transform_input(x, abstract_iters=0, abstract_sigma_s=3.0, abstract_sigma_r=8.0, quant_levels=0,
                    quant_sharpness=3.0, xdog_strength=0.0, xdog_sigma=1.0, color_transfer=None, device="cpu",
                    stroke="none", stroke_segments=200, stroke_compactness=10.0, stroke_merge=8.0, parse=None,
                    stroke_shade_levels=1, stroke_eye_scale_w=1.0, stroke_eye_scale_h=1.0, stroke_eye_style="dark",
                    stroke_iris_rgb=(90, 60, 50)):
    """
    x: [N x 3 x H x W] in [-1, 1] -> 加工した入力 (CPU)。color_transfer は ilvr_ref_transform.ColorTransfer か None。
    stroke: STROKE_CHOICES。"parts" には parse (ラベル [N x H x W]) が要る。
    何も有効でなければ x をそのまま (同じ値で) 返す。
    """
    assert stroke in STROKE_CHOICES, stroke
    assert stroke != "parts" or parse is not None, "stroke='parts' にはパーシングのラベルが要る"
    use_abs = abstract_iters > 0 or quant_levels > 0 or xdog_strength > 0 or stroke != "none"
    if not use_abs and color_transfer is None:
        return x
    m = fg_mask(x).cpu()
    y = x.cpu()
    if use_abs:
        lab = to_lab(y)
        if abstract_iters > 0:
            lab = bilateral_lab(lab, abstract_sigma_s, abstract_sigma_r, abstract_iters, device)
        if stroke == "generic":
            lab = stroke_generic(lab, m, stroke_segments, stroke_compactness, stroke_merge)
        elif stroke == "parts":
            lab = stroke_parts(lab, m, parse.cpu(), stroke_shade_levels, stroke_eye_scale_w, stroke_eye_scale_h,
                               eye_style=stroke_eye_style, iris_rgb=stroke_iris_rgb)
        if quant_levels > 0:
            lab = quantize_l(lab, quant_levels, quant_sharpness)
        if xdog_strength > 0:
            lines = xdog_lines(lab, sigma=xdog_sigma)
            lab[:, :1] = lab[:, :1] * (1 - xdog_strength * (1 - lines))
        y = m * from_lab(lab) + (1 - m) * y
    if color_transfer is not None:
        y = color_transfer(y, m)  # 前景だけに適用 (m * 写像 + (1 - m) * y)
    return y
