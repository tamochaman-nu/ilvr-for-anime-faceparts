"""
SDEdit + ILVR の設定 (t0, down_N, ilvr_stop_ratio, sdedit_repeats) の格子状の探索。

  fit_score  アニメらしさの指標 (guided_diffusion/anime_score.py) の分類器を学習して保存する。
             評価に使う参照画像 (--ref_list) は実写側の学習・テストから除く。
  run        条件ごとに変換し、指標を求める。変換は scripts/sdedit_sample.py と同じライブラリ関数を同じ引数で
             呼ぶ (モデルを読み直さないよう同じプロセスで回す)。乱数は CLI の既定と同じ「全入力で同じシード」。
             各条件と同じ結果になる CLI の引数は summary の cli に残す。

指標 (条件ごとに参照画像について平均):
  lf16           入力との 16 画素の平均プーリング後の RMSE (構図・顔の向きの保持。小さいほど保持)
  lpips          入力との LPIPS (AlexNet、256px)
  anime_logit    アニメらしさの分類器のロジット (大きいほどアニメ寄り)。anime_prob はその確率
  saturation, laplacian_var  ilvr_metrics.style_stats
構図の保持 (lf16 が小さい) とアニメらしさ (anime_logit が大きい) のパレート最適な条件を選び、
それらをシード 1 でも回して、シード間の LPIPS (seed_lpips) と lf16 (seed_lf16) を求める (視点間の安定性の目安)。

出力 (--out_dir): runs/<条件>/<入力名>.png、grid_per_image.csv、grid_summary.json、pareto.png、pareto_grid.png
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from guided_diffusion.anime_score import AnimeScore, fit  # noqa: E402
from guided_diffusion.ilvr_eye import (  # noqa: E402
    SampleRNG,
    create_diffusion_from_cfg,
    load_image,
    load_model,
    make_resizers,
    read_cfg,
    save_png,
)
from guided_diffusion.ilvr_metrics import style_stats  # noqa: E402
from guided_diffusion.sdedit import sdedit_repeat, sdedit_t0, start_info, stop_ratio_to_range_t  # noqa: E402


def configs(args):
    """(名前, dict) のリスト。dict のキーは use_sdedit, t0_ratio, down_N, stop, range_t, K"""
    cs = [
        ("uncond", dict(use_sdedit=False, t0_ratio=None, down_N=0, stop=None, range_t=None, K=1)),
        ("ilvr_xT_N16_rt20", dict(use_sdedit=False, t0_ratio=None, down_N=16, stop=None, range_t=20, K=1)),
        ("ilvr_xT_N32_rt20", dict(use_sdedit=False, t0_ratio=None, down_N=32, stop=None, range_t=20, K=1)),
        ("sd_t0.25", dict(use_sdedit=True, t0_ratio=0.25, down_N=0, stop=None, range_t=None, K=1)),
        ("sd_t0.50", dict(use_sdedit=True, t0_ratio=0.50, down_N=0, stop=None, range_t=None, K=1)),
        ("sd_t0.25_K2", dict(use_sdedit=True, t0_ratio=0.25, down_N=0, stop=None, range_t=None, K=2)),
    ]
    for t0, n, stop, k in itertools.product(*(list(map(float, args.t0s.split(","))),
                                              list(map(int, args.Ns.split(","))),
                                              list(map(float, args.stops.split(","))),
                                              list(map(int, args.Ks.split(","))))):
        cs.append((f"sd_t{t0:.2f}_N{n}_stop{stop:.2f}_K{k}",
                   dict(use_sdedit=True, t0_ratio=t0, down_N=n, stop=stop, range_t=None, K=k)))
    return cs


def cli_args(c):
    a = []
    if c["use_sdedit"]:
        a += ["--use_sdedit", "--t0_ratio", str(c["t0_ratio"]), "--sdedit_repeats", str(c["K"])]
    if c["down_N"] > 0:
        a += ["--use_ilvr", "--down_N", str(c["down_N"])]
        a += ["--ilvr_stop_ratio", str(c["stop"])] if c["stop"] is not None else ["--range_t", str(c["range_t"])]
    else:
        a += ["--use_ilvr", "False"]
    return " ".join(a)


def quantize(x):
    """PNG に保存して読み直したのと同じ値にする"""
    return ((x.clamp(-1, 1) + 1) * 127.5).round() / 127.5 - 1


def run_one(diffusion, model, x0, c, seed, batch_size):
    T = diffusion.num_timesteps
    t0 = sdedit_t0(c["t0_ratio"], T) if c["use_sdedit"] else None
    resizers = make_resizers(256, 256, c["down_N"], device=x0.device) if c["down_N"] > 0 else None
    range_t = stop_ratio_to_range_t(c["stop"], T) if c["stop"] is not None else (c["range_t"] or 0)
    out = []
    for st in range(0, len(x0), batch_size):
        xb = x0[st:st + batch_size]
        rng = SampleRNG([seed] * len(xb), x0.device)
        out.append(sdedit_repeat(diffusion, model, xb, repeats=c["K"], t0=t0, resizers=resizers, range_t=range_t,
                                 rng=rng))
    info = start_info(diffusion, t0 - 1) if t0 is not None else start_info(diffusion, T - 1)
    return quantize(th.cat(out)), info


def lf16(a, b):
    a01, b01 = (a + 1) / 2, (b + 1) / 2
    return ((F.avg_pool2d(a01, 16) - F.avg_pool2d(b01, 16)) ** 2).flatten(1).mean(1).sqrt()


def pareto(rows, x="lf16", y="anime_logit"):
    """x は小さいほど、y は大きいほど良い。支配されない行を x の昇順で返す。"""
    front = [r for r in rows if not any((o[x] <= r[x] and o[y] >= r[y]) and (o[x] < r[x] or o[y] > r[y])
                                        for o in rows)]
    return sorted(front, key=lambda r: r[x])


def run(args):
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    cfg = read_cfg(args.cfg)
    model = load_model(cfg, args.model_path, device)
    diffusion = create_diffusion_from_cfg(cfg, args.timestep_respacing)
    with open(args.ref_list) as f:
        names = [line.strip() for line in f if line.strip()]
    stems = [os.path.splitext(n)[0] for n in names]
    x0 = th.stack([load_image(os.path.join(args.base_samples, n), 256) for n in names]).to(device)
    import lpips

    lp = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    scorer = AnimeScore(args.score_path, device)
    in_logit = scorer.logit(x0)

    rows, per_image, outs = [], [], {}
    cs = configs(args)
    t_all = time.time()
    for name, c in cs:
        t1 = time.time()
        y, info = run_one(diffusion, model, x0, c, args.seed, args.batch_size)
        sec = time.time() - t1
        d = os.path.join(args.out_dir, "runs", name)
        os.makedirs(d, exist_ok=True)
        for s, yi in zip(stems, y):
            save_png(yi, os.path.join(d, s + ".png"))
        outs[name] = y.cpu()
        with th.no_grad():
            lv = lp(y, x0).flatten().cpu()
        lf = lf16(y, x0).cpu()
        st = style_stats(y)
        lg = scorer.logit(y)
        for i, n in enumerate(names):
            per_image.append({"config": name, "input": n, "lf16": lf[i].item(), "lpips": lv[i].item(),
                              "anime_logit": lg[i].item(), "saturation": st["saturation"][i].item(),
                              "laplacian_var": st["laplacian_var"][i].item()})
        r = {"config": name, **c, "cli": cli_args(c), "start": info, "sec": round(sec, 1),
             "lf16": lf.mean().item(), "lpips": lv.mean().item(), "anime_logit": lg.mean().item(),
             "anime_prob": th.sigmoid(lg).mean().item(), "saturation": st["saturation"].mean().item(),
             "laplacian_var": st["laplacian_var"].mean().item()}
        rows.append(r)
        print(f"[{len(rows)}/{len(cs)}] {name}: lf16 {r['lf16']:.4f} lpips {r['lpips']:.3f} "
              f"anime {r['anime_logit']:+.2f} ({r['anime_prob']:.2f}) sat {r['saturation']:.3f} "
              f"lap {r['laplacian_var']:.4f} {sec:.0f}s", flush=True)

    # パレート最適 (入力に近い uncond は除く) と、シード 1 での安定性
    cand = [r for r in rows if r["config"] != "uncond"]
    front = pareto(cand)
    for r in front[:args.stability_top]:
        c = dict((k, r[k]) for k in ("use_sdedit", "t0_ratio", "down_N", "stop", "range_t", "K"))
        y1, _ = run_one(diffusion, model, x0, c, args.seed + 1, args.batch_size)
        y0 = outs[r["config"]].to(device)
        with th.no_grad():
            r["seed_lpips"] = lp(y0, y1).mean().item()
        r["seed_lf16"] = lf16(y0, y1).mean().item()
        print(f"stability {r['config']}: seed_lpips {r['seed_lpips']:.3f} seed_lf16 {r['seed_lf16']:.4f}",
              flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "grid_per_image.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
        w.writeheader()
        w.writerows(per_image)
    with open(os.path.join(args.out_dir, "grid_summary.json"), "w") as f:
        json.dump({"args": vars(args), "score_info": scorer.info, "input_anime_logit": in_logit.mean().item(),
                   "elapsed_sec": round(time.time() - t_all, 1), "rows": rows,
                   "pareto": [r["config"] for r in front]}, f, indent=2, ensure_ascii=False)

    # 散布図 (PIL で描く)
    W, H, M = 900, 600, 60
    img = Image.new("RGB", (W, H), "white")
    from PIL import ImageDraw

    dr = ImageDraw.Draw(img)
    xs = [r["lf16"] for r in rows]
    ys = [r["anime_logit"] for r in rows]
    x_lo, x_hi = 0, max(xs) * 1.05
    y_lo, y_hi = min(ys + [in_logit.mean().item()]) - 0.5, max(ys) + 0.5

    def px(xv, yv):
        return M + (xv - x_lo) / (x_hi - x_lo) * (W - 2 * M), H - M - (yv - y_lo) / (y_hi - y_lo) * (H - 2 * M)

    dr.line([(M, H - M), (W - M, H - M)], fill="black")
    dr.line([(M, M), (M, H - M)], fill="black")
    dr.text((W // 2 - 80, H - M + 25), "lf16 (structure change, lower = kept)", fill="black")
    dr.text((5, M - 30), "anime_logit (higher = more anime)", fill="black")
    for k in range(5):
        xv = x_lo + (x_hi - x_lo) * k / 4
        dr.text((px(xv, y_lo)[0] - 15, H - M + 5), f"{xv:.3f}", fill="gray")
        yv = y_lo + (y_hi - y_lo) * k / 4
        dr.text((5, px(x_lo, yv)[1] - 6), f"{yv:+.1f}", fill="gray")
    yi = in_logit.mean().item()
    dr.line([px(x_lo, yi), px(x_hi, yi)], fill=(180, 180, 180))
    dr.text((px(x_hi, yi)[0] - 60, px(x_hi, yi)[1] - 14), "input", fill="gray")
    colors = {8: (230, 120, 0), 16: (0, 110, 200), 32: (140, 0, 170), 0: (90, 90, 90)}
    fset = {r["config"] for r in front}
    for r in rows:
        cx, cy = px(r["lf16"], r["anime_logit"])
        col = colors.get(r["down_N"], (0, 0, 0))
        rad = 3 + 3 * (r["K"] - 1)
        dr.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline=col, fill=col if r["config"] in fset else None)
        if r["config"] in fset or not r["use_sdedit"] or r["down_N"] == 0:
            dr.text((cx + 6, cy - 6), r["config"].replace("sd_", ""), fill=col)
    dr.text((W - M - 260, M), "orange N8 / blue N16 / purple N32 / gray no ILVR", fill="black")
    dr.text((W - M - 260, M + 14), "filled = Pareto, large = K2", fill="black")
    img.save(os.path.join(args.out_dir, "pareto.png"))

    # パレート最適の条件の出力を並べる (行: 入力、列: [入力, 条件...])
    n_show = min(args.grid_refs, len(names))
    show = [r["config"] for r in front]

    def u8(x):
        return ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()

    g = np.full((n_show * 258, (len(show) + 1) * 258, 3), 255, np.uint8)
    for i in range(n_show):
        g[i * 258:i * 258 + 256, :256] = u8(x0[i].cpu())
        for j, name in enumerate(show):
            g[i * 258:i * 258 + 256, (j + 1) * 258:(j + 1) * 258 + 256] = u8(outs[name][i])
    Image.fromarray(g).save(os.path.join(args.out_dir, "pareto_grid.png"))
    with open(os.path.join(args.out_dir, "pareto_grid.txt"), "w") as f:
        f.write("列: 入力, " + ", ".join(show) + "\n")


def fit_score(args):
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    with open(args.ref_list) as f:
        exclude = [line.strip() for line in f if line.strip()]
    model, info = fit(args.anime_dir, args.base_samples, device, n_train=args.n_train, n_test=args.n_test,
                      exclude_real=exclude)
    os.makedirs(os.path.dirname(args.score_path) or ".", exist_ok=True)
    th.save({"model": model, "info": info}, args.score_path)
    print(json.dumps({k: v for k, v in info.items() if k != "excluded_real"}, indent=2, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["fit_score", "run"])
    p.add_argument("--model_path", default="")
    p.add_argument("--cfg", default="")
    p.add_argument("--base_samples", required=True)
    p.add_argument("--ref_list", required=True)
    p.add_argument("--anime_dir", default="")
    p.add_argument("--score_path", default="diagnostics/outputs/anime_score/anime_score.pt")
    p.add_argument("--n_train", type=int, default=2000)
    p.add_argument("--n_test", type=int, default=500)
    p.add_argument("--out_dir", default="diagnostics/outputs/sdedit_grid")
    p.add_argument("--timestep_respacing", default="100")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--t0s", default="0.5,0.7,0.9")
    p.add_argument("--Ns", default="8,16,32")
    p.add_argument("--stops", default="0.1,0.2,0.3,0.4")
    p.add_argument("--Ks", default="1,2")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--stability_top", type=int, default=12)
    p.add_argument("--grid_refs", type=int, default=4)
    args = p.parse_args()
    {"fit_score": fit_score, "run": run}[args.mode](args)


if __name__ == "__main__":
    main()
