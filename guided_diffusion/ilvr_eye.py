"""
目DDPM (EDM2 の MPUNetModel, unet_mp.py) を ILVR で条件付けして、実写の目からアニメ調の目を生成する。

拡散過程は公式の create_gaussian_diffusion (gaussian_diffusion.py / respace.py) をそのまま使い、
ILVR のループだけを書き直している。公式の p_sample_loop_progressive の ILVR 補正

    x_{t-1} <- x_{t-1} - up(down(x_{t-1})) + up(down(q_sample(y, t)))   (i > range_t のステップだけ)

と同じ処理・同じ乱数の消費順で、次の 2 点を選べるようにした:
  - t_ref: q_sample に渡すステップ。"t" (公式と同じ、既定) か "t-1" (論文 Algorithm 1 と同じ)。
    SpacedDiffusion では t は間引き後のステップ番号で、p_sample の後の x はひとつ前の番号 (t-1) の
    ノイズレベルにある。"t-1" はその番号に合わせて参照画像にノイズを乗せる。
  - リサイズ: 公式の Resizer (resizer.py, 既定) か F.interpolate(bicubic, antialias=True)。
    Resizer は縮小後が 1 画素でも動くサブクラス (resizer_1px.Resizer1px) を使う。2 画素以上では公式と同一。
加えて、乱数をサンプルごとの Generator から引けるようにした (SampleRNG。バッチの組み方によらず再現できる)。
"""

import os
import random
import re
import zlib

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .image_datasets import center_crop_arr
from .script_util import NUM_CLASSES, create_gaussian_diffusion
from .unet_mp import MPUNetModel

T_REF_CHOICES = ("t", "t-1")
MASK_MODE_CHOICES = ("composite", "multiply")
RESIZER_CHOICES = ("resizer", "interpolate")


def read_cfg(path):
    """学習フォークの scripts/load_cfg.sh (save_cfg) が書く KEY="VALUE" 形式の train.cfg を読む。"""
    cfg = {}
    with open(path) as f:
        for line in f:
            m = re.match(r"^\s*([A-Z_][A-Z0-9_]*)=(.*)$", line.split("#")[0].rstrip())
            if m:
                cfg[m.group(1)] = m.group(2).strip().strip('"')
    return cfg


def _bool(s):
    return str(s).lower() in ("true", "1", "yes")


def create_model_from_cfg(cfg):
    """
    学習フォーク (tamochaman-nu/guided-diffusion@5bb6a5c) の script_util.create_model の arch="mp" 分岐と
    同じ引数で MPUNetModel を作る。推論なので fp32 (use_bf16 / use_fp16 は使わない)。
    """
    assert cfg["ARCH"] == "mp", f"ARCH={cfg['ARCH']}: MPUNetModel 用の cfg ではない"
    assert not _bool(cfg.get("ATTR_COND", "False")), "属性条件付きのモデルには対応していない"
    image_size = int(cfg["IMAGE_SIZE"])
    channel_mult = cfg.get("CHANNEL_MULT", "")
    assert channel_mult, "CHANNEL_MULT が空 (image_size ごとの既定値は使わず、cfg に明示されている前提)"
    clip_act = float(cfg.get("MP_CLIP_ACT", "256"))
    return MPUNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=int(cfg["NUM_CHANNELS"]),
        out_channels=6 if _bool(cfg["LEARN_SIGMA"]) else 3,
        num_res_blocks=int(cfg["NUM_RES_BLOCKS"]),
        attention_resolutions=tuple(image_size // int(r) for r in cfg["ATTENTION_RESOLUTIONS"].split(",")),
        dropout=float(cfg["DROPOUT"]),
        channel_mult=tuple(int(m) for m in channel_mult.split(",")),
        num_classes=NUM_CLASSES if _bool(cfg["CLASS_COND"]) else None,
        use_checkpoint=False,
        use_fp16=False,
        use_bf16=False,
        num_heads=int(cfg["NUM_HEADS"]),
        num_head_channels=int(cfg["NUM_HEAD_CHANNELS"]),
        diffusion_steps=int(cfg["DIFFUSION_STEPS"]),
        noise_schedule=cfg["NOISE_SCHEDULE"],
        clip_act=clip_act if clip_act > 0 else None,
    )


def create_diffusion_from_cfg(cfg, timestep_respacing=""):
    # 学習フォークの run_train.sh は diffusion_steps / noise_schedule 以外を渡さないので、残りは script_util の既定値
    return create_gaussian_diffusion(
        steps=int(cfg["DIFFUSION_STEPS"]),
        learn_sigma=_bool(cfg["LEARN_SIGMA"]),
        sigma_small=False,
        noise_schedule=cfg["NOISE_SCHEDULE"],
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        timestep_respacing=timestep_respacing,
    )


def load_model(cfg, model_path, device):
    model = create_model_from_cfg(cfg)
    model.load_state_dict(th.load(model_path, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval()


class DropRefImg(nn.Module):
    """
    公式の p_sample_loop は model_kwargs["ref_img"] をモデルにも渡す (公式の UNetModel は ref_img を受け取って
    捨てる)。MPUNetModel は受け取れないので、公式のループに通すときはこれで包む。
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, timesteps, ref_img=None, **kwargs):
        return self.model(x, timesteps, **kwargs)


def make_resizers(height, width, down_N, method="resizer", device="cpu", down_size=0):
    """
    (down, up) を返す。down_N は H と W の両方を割り切る必要がある。down_N=0 なら None (無条件生成)。
    down_size > 0 なら down_N の代わりに縮小後の一辺を直接指定する (6x6 など、32 を割り切らない大きさ用)。
    """
    assert method in RESIZER_CHOICES, method
    if down_size > 0:
        h = w = down_size
    elif down_N == 0:
        return None
    else:
        assert down_N >= 1 and height % down_N == 0 and width % down_N == 0, (
            f"down_N={down_N} が画像サイズ {height}x{width} を割り切らない"
        )
        h, w = height // down_N, width // down_N
    if method == "resizer":
        # 公式の Resizer (resizer.py) は縮小後が 1 画素だと IndexError になるので、そこだけ直したサブクラスを使う
        from .resizer_1px import Resizer1px

        # Resizer は後ろ 2 次元だけを変換し、重みはバッチサイズに依存しないのでバッチ 1 の形で作る。
        # 出力の大きさは output_shape で与える (割り切れない倍率で ceil の丸めがずれないように)
        down = Resizer1px((1, 3, height, width), output_shape=(1, 3, h, w)).to(device)
        up = Resizer1px((1, 3, h, w), output_shape=(1, 3, height, width)).to(device)
        return down, up

    def down(x):
        return F.interpolate(x, size=(h, w), mode="bicubic", antialias=True, align_corners=False)

    def up(x):
        return F.interpolate(x, size=(height, width), mode="bicubic", align_corners=False)

    return down, up


def sample_seed(seed, name):
    """シードと参照画像のファイル名から、そのサンプル用のシードを決める (参照画像の組や順序によらない)。"""
    return int(np.random.SeedSequence([seed, zlib.crc32(name.encode())]).generate_state(1, np.uint64)[0] >> 1)


class SampleRNG:
    """サンプルごとの Generator から標準正規乱数を引く。バッチの組み方を変えても各サンプルの乱数は同じ。"""

    def __init__(self, seeds, device):
        self.device = th.device(device)
        self.generators = [th.Generator(device=self.device).manual_seed(int(s)) for s in seeds]

    def __call__(self, shape):
        assert shape[0] == len(self.generators), (shape, len(self.generators))
        return th.stack([th.randn(shape[1:], generator=g, device=self.device) for g in self.generators])


def ilvr_sample(diffusion, model, ref, resizers, range_t=0, t_ref="t", noise=None, clip_denoised=True,
                progress=False, rng=None, mask=None, mask_mode="composite"):
    """
    ref ([N x 3 x H x W], [-1, 1]) を参照画像として ILVR でサンプリングする。resizers=None なら無条件生成。
    diffusion が SpacedDiffusion なら、ループのステップ番号 i は間引き後の番号で、p_mean_variance はそれを
    元のタイムステップに直してモデルに渡し、q_sample は間引き後の alphas_cumprod を使う。
    乱数は公式の p_sample_loop_progressive と同じ順序で引く (初期ノイズ -> 各ステップの p_sample ->
    補正するステップでは q_sample 用のノイズ)。rng (shape -> 乱数) を省くとグローバルな乱数を使い、
    同じシードなら公式のループと同じ結果になる。
    mask ([N x 1 x H x W], [0, 1]) を渡すとマスク付き ILVR にする (1 の部分だけ参照画像の低周波に合わせ、
    0 の部分はモデルに任せる)。None なら公式と同じ補正。mask_mode:
      composite  参照画像のマスクの外を現在の x で置き換えてから補正する:
                 x <- x - up(down(x)) + up(down(m*y + (1-m)*x)) = x + up(down(m * (y - x)))
                 補正量はマスク内の差の低周波なので、ステップを重ねても発散しない (既定)
      multiply   補正量にマスクを掛ける: x <- x + m * (up(down(y)) - up(down(x)))。down はマスクの外も
                 含めて平均するので、外のずれをマスク内の画素だけで埋め合わせようとして値が飽和する (比較用)
    """
    assert mask_mode in MASK_MODE_CHOICES, mask_mode
    assert t_ref in T_REF_CHOICES, t_ref
    shape = tuple(ref.shape)
    device = ref.device
    if rng is None:
        def rng(shape):
            return th.randn(*shape, device=device)
    img = noise if noise is not None else rng(shape)
    indices = list(range(diffusion.num_timesteps))[::-1]
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices, leave=False)
    for i in indices:
        t = th.full((shape[0],), i, device=device, dtype=th.long)
        with th.no_grad():
            # diffusion.p_sample と同じ計算 (乱数だけ rng から引く)
            out = diffusion.p_mean_variance(model, img, t, clip_denoised=clip_denoised, model_kwargs={})
            nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)  # t == 0 ではノイズを足さない
            x = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * rng(shape)
            if resizers is not None and i > range_t:
                down, up = resizers
                eps = rng(shape)
                if t_ref == "t":
                    y = diffusion.q_sample(ref, t, eps)
                elif i == 0:
                    y = ref  # t-1 = -1: ノイズの無い参照画像そのもの
                else:
                    y = diffusion.q_sample(ref, t - 1, eps)
                if mask is None:
                    x = x - up(down(x)) + up(down(y))
                elif mask_mode == "composite":
                    x = x + up(down(mask * (y - x)))
                else:
                    x = x + mask * (up(down(y)) - up(down(x)))
        img = x
    return img


def list_images(data_dir):
    return sorted(f for f in os.listdir(data_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))


def select_refs(data_dir, num_refs, seed=0, balance_lr=True, candidates=None):
    """
    参照画像を固定シードで選ぶ (ファイル名順で返す)。balance_lr なら、パーツ切り出しの命名規則
    (<元画像>_l.png / <元画像>_r.png) に従って左右の目を半数ずつにする。candidates (ファイル名の集合) を
    渡すとその中から選ぶ。
    """
    files = list_images(data_dir) if candidates is None else sorted(candidates)
    rng = random.Random(seed)
    if not balance_lr:
        return sorted(rng.sample(files, min(num_refs, len(files))))
    left = [f for f in files if os.path.splitext(f)[0].endswith("_l")]
    right = [f for f in files if os.path.splitext(f)[0].endswith("_r")]
    assert left and right, f"{data_dir} に *_l / *_r の画像が無い (--balance_lr False で使う)"
    n_left = (num_refs + 1) // 2
    return sorted(rng.sample(left, min(n_left, len(left))) + rng.sample(right, min(num_refs - n_left, len(right))))


def load_image(path, image_size):
    """学習時の ImageDataset と同じ前処理 (center_crop_arr, [-1, 1]、反転なし)。[3 x H x W] を返す。"""
    with open(path, "rb") as f:
        pil_image = Image.open(f)
        pil_image.load()
    arr = center_crop_arr(pil_image.convert("RGB"), image_size).astype(np.float32) / 127.5 - 1
    return th.from_numpy(np.transpose(arr, [2, 0, 1]))


def to_uint8_hwc(x):
    """[3 x H x W] in [-1, 1] -> uint8 [H x W x 3]"""
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(th.uint8).permute(1, 2, 0).cpu().numpy()


def save_png(x, path):
    Image.fromarray(to_uint8_hwc(x)).save(path)
