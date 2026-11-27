"""
画像ごとの「アニメらしさ」の指標。Inception-v3 (torch-fidelity の FID/KID 用の重み) の 2048 次元のプール特徴で、
アニメ顔のデータと実写の顔 (FFHQ) を見分けるロジスティック回帰を学習し、アニメである確率を返す。

KID や FID は分布どうしの距離なので、16 枚程度の条件の比較には向かない。この指標は 1 枚ごとに求まるので、
条件ごとの平均で比べられる。ただし分類器が見分けやすい手がかり (質感・線・色) に引っ張られる点に注意。
学習とテストの分け方・正解率は fit の戻り値 (と保存したファイル) に残す。
"""

import os

import numpy as np
import torch as th

from .ilvr_eye import load_image
from .ilvr_metrics import InceptionFeatures


def list_dir(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith((".png", ".jpg", ".jpeg")))


def features(extractor, images, batch_size=64):
    """images: [N x 3 x H x W] in [-1, 1] (CPU 可) -> [N x 2048] (CPU)"""
    out = []
    for i in range(0, len(images), batch_size):
        out.append(extractor(images[i:i + batch_size]).float().cpu())
    return th.cat(out)


def load_feats(extractor, d, names, image_size=256, batch=256):
    feats = []
    for i in range(0, len(names), batch):
        x = th.stack([load_image(os.path.join(d, n), image_size) for n in names[i:i + batch]])
        feats.append(features(extractor, x))
    return th.cat(feats)


def fit(anime_dir, real_dir, device, n_train=2000, n_test=500, exclude_real=(), seed=0, weight_decay=1e-3,
        steps=500):
    """
    anime_dir / real_dir から乱数で n_train + n_test 枚ずつ選び (exclude_real は実写側から除く)、標準化した特徴で
    ロジスティック回帰 (L2 正則化、LBFGS) を学習する。戻り値は (モデルの dict, 情報の dict)。
    """
    rng = np.random.RandomState(seed)
    a_names = list_dir(anime_dir)
    r_names = [n for n in list_dir(real_dir) if n not in set(exclude_real)]
    a_pick = [a_names[i] for i in rng.choice(len(a_names), n_train + n_test, replace=False)]
    r_pick = [r_names[i] for i in rng.choice(len(r_names), n_train + n_test, replace=False)]
    ext = InceptionFeatures(device)
    fa = load_feats(ext, anime_dir, a_pick)
    fr = load_feats(ext, real_dir, r_pick)
    x_tr = th.cat([fa[:n_train], fr[:n_train]])
    y_tr = th.cat([th.ones(n_train), th.zeros(n_train)])
    x_te = th.cat([fa[n_train:], fr[n_train:]])
    y_te = th.cat([th.ones(n_test), th.zeros(n_test)])
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-6
    xs_tr, xs_te = ((x_tr - mu) / sd).to(device), ((x_te - mu) / sd).to(device)
    w = th.zeros(x_tr.shape[1], device=device, requires_grad=True)
    b = th.zeros(1, device=device, requires_grad=True)
    opt = th.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")
    y = y_tr.to(device)

    def closure():
        opt.zero_grad()
        loss = th.nn.functional.binary_cross_entropy_with_logits(xs_tr @ w + b, y) + weight_decay * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with th.no_grad():
        acc_tr = (((xs_tr @ w + b) > 0).float().cpu() == y_tr).float().mean().item()
        p_te = th.sigmoid(xs_te @ w + b).cpu()
        acc_te = ((p_te > 0.5).float() == y_te).float().mean().item()
    model = {"mu": mu, "sd": sd, "w": w.detach().cpu(), "b": b.detach().cpu()}
    info = {"anime_dir": anime_dir, "real_dir": real_dir, "n_train": n_train, "n_test": n_test, "seed": seed,
            "weight_decay": weight_decay, "train_acc": acc_tr, "test_acc": acc_te,
            "test_mean_prob_anime": p_te[:n_test].mean().item(), "test_mean_prob_real": p_te[n_test:].mean().item(),
            "excluded_real": list(exclude_real)}
    return model, info


class AnimeScore:
    def __init__(self, path, device):
        d = th.load(path, map_location="cpu", weights_only=False)
        self.m = {k: v.to(device) for k, v in d["model"].items()}
        self.info = d["info"]
        self.ext = InceptionFeatures(device)
        self.device = device

    def logit(self, images):
        """images: [N x 3 x H x W] in [-1, 1] -> ロジット [N] (CPU)。確率は 0 や 1 に張り付きやすいので比較はこちらで"""
        f = features(self.ext, images).to(self.device)
        return (((f - self.m["mu"]) / self.m["sd"]) @ self.m["w"] + self.m["b"]).cpu()

    def __call__(self, images):
        """アニメである確率 [N] (CPU)"""
        return th.sigmoid(self.logit(images))
