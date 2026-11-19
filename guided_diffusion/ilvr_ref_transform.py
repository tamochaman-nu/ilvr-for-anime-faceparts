"""
ILVR に渡す前の参照画像の加工 (Step 3 の改善案)。

色の写像 (ColorTransfer): 実写の目の色の分布をアニメ目の学習データの色の分布に写す。
  Phase 3 では ILVR の出力が参照画像の平均色 (暗さ・肌の色) をそのまま受け継ぎ、それが KID の差の大部分だった。
  写像はデータセット単位で固定 (画像ごとに統計を合わせない) なので、参照画像どうしの色の違い (虹彩の色・
  肌の明るさ) は相対的に残る。CIELAB のチャネルごとに次のどちらかで写す:
    lab_meanstd   平均と標準偏差を合わせる一次変換 (Reinhard 型)
    lab_quantile  分位点を合わせる単調変換 (ヒストグラムの一致)
  strength (0〜1) で元の色との間を CIELAB で線形に補間する。
  統計は scripts/fit_color_stats.py が評価用の参照画像を除いた画像から求めて JSON に保存する。

マスク (load_masks): パーツ切り出しが出力した目のマスク (eye_mask/<同じファイル名>, 楕円をぼかしたもの) を読む。
  dilate 画素だけ膨らませて (最大値プーリング)、境界を 3x3 の平均でぼかす。
"""

import json
import os

import numpy as np
import torch as th
import torch.nn.functional as F
from PIL import Image
from skimage.color import lab2rgb, rgb2lab

COLOR_TRANSFER_CHOICES = ("none", "lab_meanstd", "lab_quantile")
NUM_QUANTILES = 1001


def to_lab_pixels(x):
    """[N x 3 x H x W] in [-1, 1] -> numpy [N x H x W x 3] (CIELAB)"""
    return rgb2lab(((x.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).double().cpu().numpy())


def lab_stats(x):
    """CIELAB のチャネルごとの平均・標準偏差・分位点 (画素単位でまとめる)。"""
    lab = to_lab_pixels(x).reshape(-1, 3)
    q = np.linspace(0, 1, NUM_QUANTILES)
    return {"mean": lab.mean(0).tolist(), "std": lab.std(0).tolist(),
            "quantiles": [np.quantile(lab[:, c], q).tolist() for c in range(3)]}


class ColorTransfer:
    def __init__(self, stats_path, method, strength=1.0, src="real", dst="anime"):
        assert method in COLOR_TRANSFER_CHOICES and method != "none", method
        with open(stats_path) as f:
            stats = json.load(f)
        self.src, self.dst = stats[src], stats[dst]
        self.method, self.strength = method, strength

    def __call__(self, x):
        """[N x 3 x H x W] in [-1, 1] -> 同じ形 (CPU)。色域外は lab2rgb が [0, 1] に切り詰める。"""
        lab = to_lab_pixels(x)
        out = np.empty_like(lab)
        for c in range(3):
            if self.method == "lab_meanstd":
                out[..., c] = ((lab[..., c] - self.src["mean"][c]) / self.src["std"][c] * self.dst["std"][c]
                               + self.dst["mean"][c])
            else:
                out[..., c] = np.interp(lab[..., c], self.src["quantiles"][c], self.dst["quantiles"][c])
        out = (1 - self.strength) * lab + self.strength * out
        rgb = np.stack([lab2rgb(o) for o in out])
        return th.from_numpy(rgb).float().permute(0, 3, 1, 2) * 2 - 1


def load_masks(mask_dir, names, image_size, dilate=0):
    """[N x 1 x H x W] in [0, 1]。マスクの画像は参照画像と同じ大きさ (切り出し時に同じ枠で作られている) を前提にする。"""
    masks = []
    for n in names:
        m = np.asarray(Image.open(os.path.join(mask_dir, n)).convert("L"), dtype=np.float32) / 255
        assert m.shape == (image_size, image_size), (n, m.shape)
        masks.append(m)
    m = th.from_numpy(np.stack(masks))[:, None]
    if dilate > 0:
        m = F.max_pool2d(m, 2 * dilate + 1, stride=1, padding=dilate)
        m = F.avg_pool2d(F.pad(m, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
    return m.clamp(0, 1)
