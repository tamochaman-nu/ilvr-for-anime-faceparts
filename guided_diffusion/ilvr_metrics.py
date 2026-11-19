"""
ILVR の出力の評価指標。画像はすべて [N x 3 x H x W]、値は [-1, 1]。

参照画像との対応 (サンプルごと。参照画像の構造・色がどれだけ残るか):
  lowfreq_rmse(x, ref, size)  size x size に縮小した画像どうしの RMSE。縮小は条件付けと同じ Resizer で、
                              条件 (down_N) によらず同じ size で測る (条件付けと同じ N で測ると自明に小さくなるため)
  lpips                       LPIPS (AlexNet)。32px のままでは AlexNet の特徴マップが潰れるので 64px に拡大して測る
  delta_e                     CIELAB の平均色の差 (CIE76)。目の付近 (中央の楕円) と画像全体の 2 通り
アニメらしさ (出力だけで決まる):
  saturation, laplacian_var, luminance  彩度・輝度のラプラシアンの分散 (線のくっきりさ)・平均輝度
  KID                         アニメ目の学習データとの Kernel Inception Distance (torch-fidelity の Inception-v3)
"""

import numpy as np
import torch as th
import torch.nn.functional as F
from skimage.color import rgb2lab

from .resizer_1px import Resizer1px

LPIPS_SIZE = 64
# 目の付近とみなす中央の楕円 (32px での半径。切り出しは目の幅が枠の約 0.5〜0.55 になるよう較正されている)
EYE_ELLIPSE_RX, EYE_ELLIPSE_RY = 9.0, 6.0


def lowfreq_rmse(x, ref, size):
    h = x.shape[-1]
    down = Resizer1px((1, 3, h, h), size / h).to(x.device)
    return ((down(x) - down(ref)) ** 2).flatten(1).mean(1).sqrt()


class LPIPS:
    def __init__(self, device):
        import lpips

        self.model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    def __call__(self, x, ref):
        up = lambda a: F.interpolate(a, size=(LPIPS_SIZE, LPIPS_SIZE), mode="bicubic", align_corners=False)  # noqa
        with th.no_grad():
            return self.model(up(x).clamp(-1, 1), up(ref).clamp(-1, 1)).flatten()


def eye_mask(h, w):
    yy, xx = np.mgrid[:h, :w]
    cy, cx = (h - 1) / 2, (w - 1) / 2
    return ((xx - cx) / EYE_ELLIPSE_RX * (w / 32)) ** 2 + ((yy - cy) / EYE_ELLIPSE_RY * (h / 32)) ** 2 <= 1


def to_lab(x):
    """[N x 3 x H x W] in [-1, 1] -> numpy [N x H x W x 3] (CIELAB, D65)"""
    rgb = ((x.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).cpu().double().numpy()
    return rgb2lab(rgb)


def delta_e(x, ref):
    """(目の付近, 画像全体) の平均色の差 (CIE76) をサンプルごとに返す。"""
    lx, lr = to_lab(x), to_lab(ref)
    m = eye_mask(*x.shape[-2:])
    d_eye = np.linalg.norm(lx[:, m].mean(1) - lr[:, m].mean(1), axis=1)
    d_img = np.linalg.norm(lx.mean((1, 2)) - lr.mean((1, 2)), axis=1)
    return d_eye, d_img


def style_stats(x):
    """彩度 (HSV の S)、輝度のラプラシアンの分散、平均輝度をサンプルごとに返す。"""
    x01 = (x.clamp(-1, 1) + 1) / 2
    lum = 0.299 * x01[:, 0] + 0.587 * x01[:, 1] + 0.114 * x01[:, 2]
    mx, mn = x01.max(1).values, x01.min(1).values
    sat = ((mx - mn) / mx.clamp(min=1e-6)).mean((1, 2))
    k = th.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=x.dtype, device=x.device)[None, None]
    lap = F.conv2d(lum[:, None], k).flatten(1).var(1)
    return {"saturation": sat, "laplacian_var": lap, "luminance": lum.mean((1, 2))}


class InceptionFeatures:
    """torch-fidelity の Inception-v3 (FID/KID の標準の重み) の 2048 次元のプール特徴。入力は内部で 299px に拡大される。"""

    def __init__(self, device):
        from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

        self.device = device
        self.model = FeatureExtractorInceptionV3("inception-v3-compat", ["2048"]).to(device).eval()

    def __call__(self, x, batch_size=500):
        feats = []
        with th.no_grad():
            for i in range(0, len(x), batch_size):
                u8 = ((x[i:i + batch_size].clamp(-1, 1) + 1) * 127.5).round().to(th.uint8).to(self.device)
                feats.append(self.model(u8)[0].double().cpu())
        return th.cat(feats)


def kid(feats_1, feats_2, subset_size=1000, subsets=100, seed=0):
    """KID (平均と、サブセット間の標準偏差)。サブセットの大きさは両方の件数以下に切り詰める。"""
    from torch_fidelity.metric_kid import KEY_METRIC_KID_MEAN, KEY_METRIC_KID_STD, kid_features_to_metric

    size = min(subset_size, len(feats_1), len(feats_2))
    out = kid_features_to_metric(feats_1, feats_2, kid_subsets=subsets, kid_subset_size=size, kid_degree=3,
                                 kid_gamma=None, kid_coef0=1, rng_seed=seed, verbose=False)
    return out[KEY_METRIC_KID_MEAN], out[KEY_METRIC_KID_STD], size
