"""
アニメの目の形をした代理画像 (Step 3 の改善案 2)。

Phase 3 / Step 3 では、ILVR が実写の参照画像の陰影 (目の周りの影・眉・まぶたの溝) をそのまま低周波として渡すため、
それが眼鏡のような縁や前髪として生成されることがあった。代理画像は実写の陰影を渡さず、目の位置・大きさ・傾き・
虹彩の位置と色、肌の色だけを、アニメの目に近い単純な図形で渡す。

  目の楕円   目のマスク (パーツ切り出しの eye_mask。楕円をぼかしたもの) の重み付きモーメントから、中心・半径・傾き
             (一様な楕円なら、主軸方向の分散 = 半径^2 / 4)
  虹彩       楕円の中で暗い画素の重心 (横方向だけ) に置く。半径は楕円の短半径の iris_scale 倍
             (長半径の 0.45 倍まで)。色は虹彩の付近の参照画像 (色を写したもの) の平均。上側を暗くし、瞳孔とハイライトを描く
  白目       楕円の内側
  まぶたの線 楕円の上側の縁に沿った太い線 (lash_px 画素) と、下側の細い線
  肌         マスクの外 (膨らませたマスクの外) の参照画像の平均色で塗りつぶす
32px の SS 倍の大きさで描いてから平均で縮小し、縁を滑らかにする。

型 (白目・虹彩の塗り方・まぶたの線の形) は 1 種類で、画像ごとに変わるのは寸法・位置・色だけ。実写の目の形を
どこまで残すかは blend (代理画像と実写の参照画像の混合比。0 なら代理画像を使わない従来の方式) で調整する。
"""

import numpy as np
import torch as th
import torch.nn.functional as F

PROXY_CHOICES = ("none", "ellipse")
SS = 4  # 描画の拡大率
SCLERA_RGB = (0.96, 0.95, 0.95)
LASH_RGB = (0.16, 0.10, 0.10)


def ellipse_from_mask(m):
    """m: [H x W] in [0, 1] -> (cx, cy, a, b, theta)。a >= b、theta は長軸の向き (ラジアン、画像座標)。"""
    h, w = m.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float64)
    s = m.sum()
    cx, cy = (m * xx).sum() / s, (m * yy).sum() / s
    cov = np.array([[(m * (xx - cx) ** 2).sum(), (m * (xx - cx) * (yy - cy)).sum()],
                    [(m * (xx - cx) * (yy - cy)).sum(), (m * (yy - cy) ** 2).sum()]]) / s
    evals, evecs = np.linalg.eigh(cov)  # 昇順
    a, b = 2 * np.sqrt(evals[1]), 2 * np.sqrt(evals[0])
    theta = np.arctan2(evecs[1, 1], evecs[0, 1])
    if theta > np.pi / 2:
        theta -= np.pi
    elif theta < -np.pi / 2:
        theta += np.pi
    return cx, cy, a, b, theta


def draw_proxy(ref, m, iris_scale=1.1, lash_px=1.6, skin_dilate=3):
    """
    ref: [3 x H x W] in [-1, 1] (色を写した参照画像)、m: [H x W] in [0, 1] (目のマスク)。
    代理画像 [3 x H x W] in [-1, 1] と、描いた図形のパラメータ (dict) を返す。
    """
    ref01 = ((ref.clamp(-1, 1) + 1) / 2).double().cpu().numpy()  # [3 x H x W]
    m = m.double().cpu().numpy()
    h, w = m.shape
    cx, cy, a, b, theta = ellipse_from_mask(m)
    yy, xx = np.mgrid[:h, :w].astype(np.float64)

    # 肌の色: 膨らませたマスクの外
    md = F.max_pool2d(th.from_numpy(m)[None, None], 2 * skin_dilate + 1, stride=1, padding=skin_dilate)[0, 0].numpy()
    out_w = (1 - md) + 1e-6
    skin = (ref01 * out_w).sum((1, 2)) / out_w.sum()

    # 虹彩の位置: 楕円の中で暗い画素 (輝度の下位) の重心。縦は楕円の中心に置く
    lum = 0.299 * ref01[0] + 0.587 * ref01[1] + 0.114 * ref01[2]
    inner = m > 0.5
    thr = np.quantile(lum[inner], 0.3) if inner.any() else lum.min()
    dark = inner & (lum <= thr)
    ix = xx[dark].mean() if dark.any() else cx
    ix = float(np.clip(ix, cx - 0.5 * a, cx + 0.5 * a))
    iy = cy + (ix - cx) * np.tan(theta)  # 長軸の上
    r = min(iris_scale * b, 0.45 * a)
    near = (xx - ix) ** 2 + (yy - iy) ** 2 <= r ** 2
    iris = ref01[:, near & inner].mean(1) if (near & inner).any() else ref01[:, inner].mean(1)

    # SS 倍の格子で描く
    H, W = h * SS, w * SS
    Y, X = np.mgrid[:H, :W].astype(np.float64)
    X, Y = (X + 0.5) / SS - 0.5, (Y + 0.5) / SS - 0.5  # 元の画素座標
    c, s = np.cos(theta), np.sin(theta)
    u = (X - cx) * c + (Y - cy) * s  # 長軸方向
    v = -(X - cx) * s + (Y - cy) * c  # 短軸方向 (下が正)
    rho = np.sqrt((u / a) ** 2 + (v / b) ** 2)
    eye = rho <= 1
    img = np.empty((3, H, W))
    img[:] = skin[:, None, None]
    img[:, eye] = np.array(SCLERA_RGB)[:, None]

    di = np.sqrt((X - ix) ** 2 + (Y - iy) ** 2)
    in_iris = eye & (di <= r)
    # 虹彩: 上側ほど暗く (0.55 倍 -> 1.1 倍)、縁を少し暗く
    shade = np.clip(0.55 + 0.55 * ((Y - (iy - r)) / (2 * r)), 0.55, 1.1) * np.where(di > 0.8 * r, 0.8, 1.0)
    for k in range(3):
        img[k][in_iris] = np.clip(iris[k] * shade[in_iris], 0, 1)
    pupil = eye & (di <= 0.45 * r)
    img[:, pupil] = (iris * 0.3)[:, None]
    hl = (X - (ix - 0.35 * r)) ** 2 + (Y - (iy - 0.35 * r)) ** 2 <= (0.22 * r) ** 2
    img[:, hl & eye] = 1.0

    # まぶたの線: 楕円の縁からの距離 (短半径方向の画素数で近似)
    edge_px = (rho - 1) * b
    upper = (v < 0.25 * b) & (edge_px > -0.3 * lash_px) & (edge_px < 0.7 * lash_px)
    upper &= np.abs(u) <= 1.08 * a
    lower = (v > 0.4 * b) & (np.abs(edge_px) < 0.25 * lash_px)
    img[:, upper] = np.array(LASH_RGB)[:, None]
    img[:, lower & ~upper] = (0.5 * np.array(LASH_RGB) + 0.5 * skin)[:, None]

    small = img.reshape(3, h, SS, w, SS).mean((2, 4))
    params = {"cx": cx, "cy": cy, "a": a, "b": b, "theta_deg": float(np.degrees(theta)), "iris_x": ix,
              "iris_r": r, "iris_rgb": iris.tolist(), "skin_rgb": skin.tolist()}
    return th.from_numpy(small).float() * 2 - 1, params


def make_proxies(refs, masks, blend=1.0, **kw):
    """refs: [N x 3 x H x W]、masks: [N x 1 x H x W]。blend * 代理画像 + (1 - blend) * refs と、パラメータのリストを返す。"""
    outs, params = [], []
    for ref, m in zip(refs, masks):
        p, prm = draw_proxy(ref, m[0], **kw)
        outs.append(blend * p + (1 - blend) * ref.cpu())
        params.append(prm)
    return th.stack(outs), params
