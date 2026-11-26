"""
ILVR の変換の不安定さの計測 (顔全体の DDPM でのマルチビュー変換の前段の調査)。

  seed     計測 A: 同じ参照画像でシードだけを変えた出力のばらつきを集計する。生成は既存の CLI で行う:
             scripts/ilvr_sample_eye.py --seeds 0,...,7 --out_dir <A_dir>/N<k>   (k = 0, 8, 16, ...)
           指標 (画像ごと):
             std_full / std_fg     シード間の画素ごとの標準偏差 ([0, 1] の値) の平均 (画像全体 / 前景)
             std_lp{f}             1/f に縮小して戻した低周波成分のシード間の標準偏差 (f = 8, 16, 32)。
                                   ILVR (down_N = k) は f >= k の成分を拘束するので、そこが小さくなるはず
             lpips_pair            シードどうしの LPIPS (AlexNet、256px のまま) の平均 (全ての組)
             lf_rmse_ref16         参照画像との低周波 (16x16) の RMSE の、シードについての平均
           前景は参照画像の白くない画素 (背景除去済みの白背景) を 4px 膨らませたもの。
  perturb  計測 B: 参照画像を少し動かして生成し (シード 0)、出力を逆向きに戻して元の出力と比べる。
           揺らし方: 横の平行移動 1, 2, 4, 8px (端は画素の複製で埋める)、0.5px (+0.25px と -0.25px の出力どうし)、
           中心を固定した拡大 0.98 / 1.02 倍、明るさ 0.95 / 1.05 倍。
           ノイズ: fixed は画素の位置に固定 (各視点を独立に変換する現在の方式)、roll は整数の平行移動のときだけ
           ノイズ (初期ノイズ・各ステップのノイズ・q_sample のノイズ) を同じだけずらす。
           乱数は CLI と同じ ilvr_eye.sample_seed(0, 参照画像名) なので、揺らさない出力は A のシード 0 と同じ。
           比較の指標 (固定の内側の領域 rows 16:240, cols 24:232 で):
             rmse, lpips, lf_rmse8 / lf_rmse16 (8 / 16 画素の平均プーリング後の RMSE)
           基準として、A のシード 0 とシード 1〜7 の同じ指標 (seed) と、補間を往復させただけの差 (floor) を並べる。

例 (Docker):
  docker compose run --rm -v <FFHQ>:/data/real_face:ro shell python diagnostics/ilvr_stability.py seed \
      --a_dir diagnostics/outputs/stability/A --conds 0,8,16,32
  docker compose run --rm -v <FFHQ>:/data/real_face:ro shell python diagnostics/ilvr_stability.py perturb \
      --model_path ddpm/anime-face256-mp128-ema_0.9995_100000.pt --cfg ddpm/anime-face256-mp128.cfg \
      --base_samples /data/real_face --ref_list diagnostics/outputs/stability/ref_list16.txt \
      --a_dir diagnostics/outputs/stability/A --out_dir diagnostics/outputs/stability/B --conds 8,16,32
"""

import argparse
import csv
import itertools
import json
import os
import sys
import time

import numpy as np
import torch as th
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guided_diffusion.ilvr_eye import (  # noqa: E402
    SampleRNG,
    create_diffusion_from_cfg,
    ilvr_sample,
    load_image,
    load_model,
    make_resizers,
    read_cfg,
    sample_seed,
    save_png,
)

LP_FACTORS = (8, 16, 32)
# 比較に使う内側の領域 (端を埋めた部分と、平行移動で外に出る部分を避ける。16 の倍数の大きさ)
ROWS = slice(16, 240)
COLS = (24, 232)
FG_THRESH = 0.94  # [0, 1] でこれより暗いチャネルがある画素を前景とする
FG_DILATE = 4


# ---------------------------------------------------------------- 共通


def read_png(path):
    """PNG -> [3 x H x W] の [-1, 1]"""
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 127.5 - 1
    return th.from_numpy(a.transpose(2, 0, 1))


def to01(x):
    return (x.clamp(-1, 1) + 1) / 2


def lowpass(x, f):
    """Resizer (ILVR の条件付けと同じ) で 1/f に縮小して戻した成分。x: [N x 3 x H x W]"""
    h, w = x.shape[-2:]
    down, up = make_resizers(h, w, f, device=x.device)
    return up(down(x))


def fg_mask(ref):
    """ref: [N x 3 x H x W] in [-1, 1] -> [N x 1 x H x W] の bool"""
    m = (to01(ref).min(1, keepdim=True).values < FG_THRESH).float()
    m = F.max_pool2d(m, 2 * FG_DILATE + 1, stride=1, padding=FG_DILATE)
    return m > 0.5


class LPIPSFull:
    """LPIPS (AlexNet)。顔は 256px のまま測る (ilvr_metrics.LPIPS は 32px の目用に 64px へ拡大するので使わない)。"""

    def __init__(self, device):
        import lpips

        self.model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    def __call__(self, a, b):
        with th.no_grad():
            return self.model(a.clamp(-1, 1), b.clamp(-1, 1)).flatten()


def load_run(run_dir):
    """run.json から {参照画像名: {seed: 出力のパス}} と、参照画像のパスを返す。"""
    with open(os.path.join(run_dir, "run.json")) as f:
        run = json.load(f)
    samples, refs = {}, {}
    for o in run["outputs"]:
        samples.setdefault(o["ref"], {})[o["seed"]] = os.path.join(run_dir, o["sample"])
        refs[o["ref"]] = os.path.join(run_dir, o["ref_png"])
    return run, samples, refs


def write_csv(path, rows):
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def mean_by(rows, group_keys, value_keys):
    out = {}
    for r in rows:
        k = tuple(r[g] for g in group_keys)
        out.setdefault(k, []).append(r)
    res = []
    for k, rs in out.items():
        d = dict(zip(group_keys, k))
        d["n"] = len(rs)
        for v in value_keys:
            vals = [r[v] for r in rs if r.get(v) is not None and r[v] == r[v]]
            d[v] = float(np.mean(vals)) if vals else None
            d[v + "_sd"] = float(np.std(vals)) if vals else None
        res.append(d)
    return res


def heat(x):
    """[H x W] の非負値 -> uint8 [H x W x 3] (黒 -> 赤 -> 黄 -> 白)"""
    t = np.clip(x, 0, 1)
    r = np.clip(3 * t, 0, 1)
    g = np.clip(3 * t - 1, 0, 1)
    b = np.clip(3 * t - 2, 0, 1)
    return (np.stack([r, g, b], -1) * 255).round().astype(np.uint8)


def tile(rows_of_imgs, pad=2):
    """uint8 [H x W x 3] の 2 次元リストを 1 枚に並べる"""
    h, w = rows_of_imgs[0][0].shape[:2]
    nr, nc = len(rows_of_imgs), max(len(r) for r in rows_of_imgs)
    g = np.full((nr * (h + pad) + pad, nc * (w + pad) + pad, 3), 255, np.uint8)
    for i, r in enumerate(rows_of_imgs):
        for j, im in enumerate(r):
            g[pad + i * (h + pad):pad + i * (h + pad) + h, pad + j * (w + pad):pad + j * (w + pad) + w] = im
    return g


def u8(x):
    """[3 x H x W] in [-1, 1] -> uint8 [H x W x 3]"""
    return (to01(x) * 255).round().byte().permute(1, 2, 0).cpu().numpy()


# ---------------------------------------------------------------- 計測 A


def run_seed(args):
    device = th.device(args.device)
    lp = LPIPSFull(device)
    rows, maps, grids = [], {}, {}
    for k in [int(c) for c in args.conds.split(",")]:
        run_dir = os.path.join(args.a_dir, f"N{k}")
        run, samples, refs = load_run(run_dir)
        names = sorted(samples)
        maps[k] = []
        grids[k] = []
        for name in names:
            seeds = sorted(samples[name])
            x = th.stack([read_png(samples[name][s]) for s in seeds]).to(device)  # [S x 3 x H x W]
            ref = read_png(refs[name])[None].to(device)
            fg = fg_mask(ref)[0, 0]
            x01 = to01(x)
            std = x01.std(0, unbiased=False).mean(0)  # [H x W]
            r = {"cond": f"N{k}", "down_N": k, "ref": name, "num_seeds": len(seeds),
                 "std_full": std.mean().item(), "std_fg": std[fg].mean().item(),
                 "fg_ratio": fg.float().mean().item()}
            for f in LP_FACTORS:
                s = lowpass(x01, f).std(0, unbiased=False).mean(0)
                r[f"std_lp{f}"] = s.mean().item()
                r[f"std_lp{f}_fg"] = s[fg].mean().item()
            pairs = list(itertools.combinations(range(len(seeds)), 2))
            ia = th.tensor([p[0] for p in pairs], device=device)
            ib = th.tensor([p[1] for p in pairs], device=device)
            r["lpips_pair"] = lp(x[ia], x[ib]).mean().item()
            d16 = make_resizers(256, 256, 16, device=device)[0]
            r["lf_rmse_ref16"] = ((d16(x01) - d16(to01(ref))) ** 2).flatten(1).mean(1).sqrt().mean().item()
            rows.append(r)
            maps[k].append(std.cpu().numpy())
            if len(grids[k]) < args.grid_refs:
                grids[k].append([u8(ref[0])] + [u8(xi) for xi in x] + [heat(std.cpu().numpy() / args.heat_max)])
        print(f"N{k}: {len(names)} 枚", flush=True)

    os.makedirs(args.a_dir, exist_ok=True)
    write_csv(os.path.join(args.a_dir, "seed_per_image.csv"), rows)
    vals = [k for k in rows[0] if k not in ("cond", "down_N", "ref", "num_seeds")]
    summary = mean_by(rows, ["cond"], vals)
    with open(os.path.join(args.a_dir, "seed_summary.json"), "w") as f:
        json.dump({"heat_max": args.heat_max, "summary": summary}, f, indent=2, ensure_ascii=False)
    for k, g in grids.items():
        Image.fromarray(tile(g)).save(os.path.join(args.a_dir, f"seed_grid_N{k}.png"))
    # 画像を通した標準偏差のマップの平均 (条件ごとに 1 枚)
    Image.fromarray(tile([[heat(np.mean(m, 0) / args.heat_max) for m in maps.values()]])).save(
        os.path.join(args.a_dir, "seed_std_mean_maps.png"))
    print_table(summary, ["cond", "n", "std_full", "std_fg", "std_lp8", "std_lp16", "std_lp32", "lpips_pair",
                          "lf_rmse_ref16"])


# ---------------------------------------------------------------- 計測 B


def shift_int(x, d):
    """右へ d 画素ずらす (左端は画素の複製で埋める)。x: [N x 3 x H x W]"""
    if d == 0:
        return x
    return F.pad(x, (d, 0, 0, 0), mode="replicate")[..., :x.shape[-1]]


def affine(x, scale=1.0, shift=0.0):
    """中心を固定して scale 倍し、右へ shift 画素ずらす (双三次補間、端は境界の値)。"""
    n, _, h, w = x.shape
    theta = th.tensor([[1 / scale, 0, -2 * shift / w / scale], [0, 1 / scale, 0]], dtype=x.dtype, device=x.device)
    grid = F.affine_grid(theta[None].expand(n, -1, -1), list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bicubic", padding_mode="border", align_corners=False).clamp(-1, 1)


def gain(x, g):
    return (to01(x) * g).clamp(0, 1) * 2 - 1


class RolledRNG:
    """もとの rng の乱数を横に d 画素ずらして返す (はみ出た分は反対側に回す)。"""

    def __init__(self, rng, d):
        self.rng, self.d = rng, d

    def __call__(self, shape):
        return th.roll(self.rng(shape), shifts=self.d, dims=-1)


SHIFTS = (1, 2, 4, 8)


def perturbations():
    """(名前, 入力の変換, 出力を元の位置に戻す変換, ノイズのずらし量 or None)"""
    ps = [("base", lambda x: x, lambda y: y, None)]
    for d in SHIFTS:
        ps.append((f"shift{d}_fixed", lambda x, d=d: shift_int(x, d), None, None))
        ps.append((f"shift{d}_roll", lambda x, d=d: shift_int(x, d), None, d))
    ps.append(("sub+0.25", lambda x: affine(x, shift=0.25), lambda y: affine(y, shift=-0.25), None))
    ps.append(("sub-0.25", lambda x: affine(x, shift=-0.25), lambda y: affine(y, shift=0.25), None))
    for s in (0.98, 1.02):
        ps.append((f"scale{s}", lambda x, s=s: affine(x, scale=s), lambda y, s=s: affine(y, scale=1 / s), None))
    for g in (0.95, 1.05):
        ps.append((f"gain{g}", lambda x, g=g: gain(x, g), lambda y: y, None))
    return ps


def crop_aligned(y, d=0):
    """内側の領域を切り出す。d は入力を右へずらした量 (出力の同じ内容は d だけ右にある)。"""
    return y[..., ROWS, COLS[0] + d:COLS[1] + d]


def dist(lp, a, b):
    """a, b: [N x 3 x h x w] in [-1, 1] -> 指標の dict (サンプルごとの list)"""
    a01, b01 = to01(a), to01(b)
    out = {"rmse": ((a01 - b01) ** 2).flatten(1).mean(1).sqrt(), "lpips": lp(a, b)}
    for f in (8, 16):
        out[f"lf_rmse{f}"] = ((F.avg_pool2d(a01, f) - F.avg_pool2d(b01, f)) ** 2).flatten(1).mean(1).sqrt()
    return {k: v.tolist() for k, v in out.items()}


def run_perturb(args):
    device = th.device(args.device)
    cfg = read_cfg(args.cfg)
    model = load_model(cfg, args.model_path, device)
    diffusion = create_diffusion_from_cfg(cfg, args.timestep_respacing)
    image_size = int(cfg["IMAGE_SIZE"])
    with open(args.ref_list) as f:
        names = [line.strip() for line in f if line.strip()]
    stems = [os.path.splitext(n)[0] for n in names]
    refs = th.stack([load_image(os.path.join(args.base_samples, n), image_size) for n in names]).to(device)
    lp = LPIPSFull(device)
    ps = perturbations()
    os.makedirs(args.out_dir, exist_ok=True)
    rows, timing = [], {}

    for k in [int(c) for c in args.conds.split(",")]:
        resizers = make_resizers(image_size, image_size, k, device=device)
        outs = {}
        t0 = time.time()
        for pname, fwd, _, roll in ps:
            d_out = os.path.join(args.out_dir, f"N{k}", pname)
            os.makedirs(d_out, exist_ok=True)
            res = []
            for st in range(0, len(names), args.batch_size):
                idx = list(range(st, min(st + args.batch_size, len(names))))
                rng = SampleRNG([sample_seed(args.seed, names[i]) for i in idx], device)
                if roll is not None:
                    rng = RolledRNG(rng, roll)
                res.append(ilvr_sample(diffusion, model, fwd(refs[idx]), resizers, range_t=args.range_t, rng=rng))
            y = th.cat(res)
            for stem, yi in zip(stems, y):
                save_png(yi, os.path.join(d_out, f"{stem}.png"))
            # 指標は保存した PNG と同じ量子化で測る (A と条件をそろえる)
            outs[pname] = th.stack([read_png(os.path.join(d_out, f"{s}.png")) for s in stems]).to(device)
            print(f"N{k} {pname}: {time.time() - t0:.0f}s", flush=True)
        timing[f"N{k}"] = round(time.time() - t0, 1)

        base = outs["base"]
        # 確認: base は A のシード 0 と同じはず
        a_dir = os.path.join(args.a_dir, f"N{k}")
        a0 = None
        if os.path.exists(os.path.join(a_dir, "run.json")):
            _, a_samples, _ = load_run(a_dir)
            a0 = th.stack([read_png(a_samples[n][args.seed]) for n in names]).to(device)
            print(f"N{k}: base と A のシード {args.seed} の最大差 {(a0 - base).abs().max().item() * 127.5:.1f}/255")

        def add(kind, pname, noise, a, b):
            m = dist(lp, a, b)
            for i, name in enumerate(names):
                rows.append({"cond": f"N{k}", "kind": kind, "perturb": pname, "noise": noise, "ref": name,
                             **{key: v[i] for key, v in m.items()}})

        base_c = crop_aligned(base)
        for d in SHIFTS:
            for noise in ("fixed", "roll"):
                add("shift", f"shift{d}", noise, crop_aligned(outs[f"shift{d}_{noise}"], d), base_c)
        add("subpixel", "shift0.5", "fixed",
            crop_aligned(affine(outs["sub+0.25"], shift=-0.25)), crop_aligned(affine(outs["sub-0.25"], shift=0.25)))
        add("floor", "shift0.5", "-",
            crop_aligned(affine(affine(base, shift=0.25), shift=-0.25)),
            crop_aligned(affine(affine(base, shift=-0.25), shift=0.25)))
        for s in (0.98, 1.02):
            add("scale", f"scale{s}", "fixed", crop_aligned(affine(outs[f"scale{s}"], scale=1 / s)), base_c)
            add("floor", f"scale{s}", "-", crop_aligned(affine(affine(base, scale=s), scale=1 / s)), base_c)
        for g in (0.95, 1.05):
            add("gain", f"gain{g}", "fixed", crop_aligned(outs[f"gain{g}"]), base_c)
            add("input", f"gain{g}", "-", crop_aligned(gain(refs, g)), crop_aligned(refs))
        # 基準: シードを変えたときの同じ指標 (A のシード 0 と 1〜7)
        if a0 is not None:
            for s in sorted(a_samples[names[0]]):
                if s == args.seed:
                    continue
                a_s = th.stack([read_png(a_samples[n][s]) for n in names]).to(device)
                add("seed", f"seed{s}", "-", crop_aligned(a_s), crop_aligned(a0))

        # 並べた画像 (出力を元の位置に戻して、同じ領域を切り出したもの)
        cols = [("ref", lambda: crop_aligned(refs)), ("base", lambda: base_c)]
        for d in (1, 4, 8):
            for noise in ("fixed", "roll"):
                cols.append((f"shift{d}_{noise}", lambda d=d, noise=noise: crop_aligned(outs[f"shift{d}_{noise}"], d)))
        cols.append(("scale1.02", lambda: crop_aligned(affine(outs["scale1.02"], scale=1 / 1.02))))
        cols.append(("gain1.05", lambda: crop_aligned(outs["gain1.05"])))
        if a0 is not None:
            a1 = th.stack([read_png(a_samples[n][1]) for n in names]).to(device)
            cols.append(("seed1", lambda: crop_aligned(a1)))
        imgs = {c: f() for c, f in cols}
        n_show = min(args.grid_refs, len(names))
        g = [[u8(imgs[c][i]) for c, _ in cols] for i in range(n_show)]
        # 差のマップ (base との画素の差の絶対値、チャネル平均)
        g_diff = [[heat(np.zeros(g[0][0].shape[:2]))] * 2
                  + [heat((to01(imgs[c][i]) - to01(base_c[i])).abs().mean(0).cpu().numpy() / args.diff_max)
                     for c, _ in cols[2:]] for i in range(n_show)]
        Image.fromarray(tile([r for pair in zip(g, g_diff) for r in pair])).save(
            os.path.join(args.out_dir, f"perturb_grid_N{k}.png"))
        with open(os.path.join(args.out_dir, f"perturb_grid_N{k}.txt"), "w") as f:
            f.write("列: " + ", ".join(c for c, _ in cols) + "\n")
            f.write(f"行: 参照画像 {names[:n_show]} の出力と、base との差 (|差| / {args.diff_max} を色に)\n")

    write_csv(os.path.join(args.out_dir, "perturb_per_image.csv"), rows)
    summary = mean_by(rows, ["cond", "kind", "perturb", "noise"], ["rmse", "lpips", "lf_rmse8", "lf_rmse16"])
    # 基準の seed は全シードを 1 行にまとめる
    seed_rows = [r for r in rows if r["kind"] == "seed"]
    summary = [s for s in summary if s["kind"] != "seed"] + mean_by(
        [{**r, "perturb": "seed1-7"} for r in seed_rows], ["cond", "kind", "perturb", "noise"],
        ["rmse", "lpips", "lf_rmse8", "lf_rmse16"])
    with open(os.path.join(args.out_dir, "perturb_summary.json"), "w") as f:
        json.dump({"args": vars(args), "region": {"rows": [ROWS.start, ROWS.stop], "cols": list(COLS)},
                   "elapsed_sec": timing, "summary": summary}, f, indent=2, ensure_ascii=False)
    print_table(summary, ["cond", "kind", "perturb", "noise", "n", "rmse", "lpips", "lf_rmse8", "lf_rmse16"])


def print_table(rows, keys):
    print("\t".join(keys))
    for r in rows:
        print("\t".join(f"{r[k]:.4f}" if isinstance(r[k], float) else str(r[k]) for k in keys))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["seed", "perturb"])
    p.add_argument("--a_dir", default="diagnostics/outputs/stability/A")
    p.add_argument("--out_dir", default="diagnostics/outputs/stability/B")
    p.add_argument("--conds", default="0,8,16,32", help="down_N の一覧 (seed は A のサブディレクトリ N<k> を読む)")
    p.add_argument("--model_path", default="")
    p.add_argument("--cfg", default="")
    p.add_argument("--base_samples", default="")
    p.add_argument("--ref_list", default="")
    p.add_argument("--range_t", type=int, default=20)
    p.add_argument("--timestep_respacing", default="100")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grid_refs", type=int, default=4)
    p.add_argument("--heat_max", type=float, default=0.3, help="標準偏差のマップでこの値を白にする")
    p.add_argument("--diff_max", type=float, default=0.5, help="差のマップでこの値を白にする")
    p.add_argument("--device", default="cuda" if th.cuda.is_available() else "cpu")
    args = p.parse_args()
    {"seed": run_seed, "perturb": run_perturb}[args.mode](args)


if __name__ == "__main__":
    main()
