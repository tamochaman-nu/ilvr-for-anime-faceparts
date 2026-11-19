"""
scripts/ilvr_sample_eye.py の実行結果 (run.json のあるディレクトリ) を評価して、1 行 1 実行の CSV にまとめる。
指標の定義は guided_diffusion/ilvr_metrics.py。

各実行について:
  - サンプルごとの指標の平均 (参照画像との対応: lowfreq_rmse_8 / lowfreq_rmse_4 / lpips / delta_e_eye / delta_e_img、
    アニメらしさ: saturation / laplacian_var / luminance)。サンプルごとの値は <run_dir>/per_sample.csv
  - KID: 全シードの出力をまとめたもの (kid, kid_std はサブセット間の標準偏差) と、シードごとの KID の平均と標準偏差
  - seed_mad: 同じ参照画像のシード違いの出力どうしの平均絶対差 (多様性)
  - 参照画像の色を写した実行 (run.json の outputs に cond_png がある) では、対応の指標を加工前の参照画像
    (refs/) と比べたものに加えて、モデルに渡した参照画像 (refs_cond/) と比べたものを <指標>_cond として出す
加えて、比較の基準として次の行を出す:
  - ref_images: 参照画像 (実写の目) そのもの。KID の上限の目安 (アニメにまったく寄せない場合)
  - anime_holdout: KID の比較対象に含めていないアニメ目の学習データ。KID の下限の目安 (同じ分布)
無条件生成 (--down_N 0 で作った実行) も同じ形で評価され、参照画像との対応の指標の「対応がない場合」の目安になる。

使い方: docker compose run --rm eval-ilvr  (-e RUN_DIRS="outputs/ilvr_eye/*" -e OUT_CSV=... で対象を指定)
"""

import argparse
import csv
import glob
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch as th

from guided_diffusion.ilvr_eye import list_images, load_image
from guided_diffusion.ilvr_metrics import (
    LPIPS,
    InceptionFeatures,
    delta_e,
    kid,
    lowfreq_rmse,
    style_stats,
)

PAIR_KEYS = ("lowfreq_rmse_8", "lowfreq_rmse_4", "lpips", "delta_e_eye", "delta_e_img")
STYLE_KEYS = ("saturation", "laplacian_var", "luminance")


def load_stack(paths, image_size=32):
    return th.stack([load_image(p, image_size) for p in paths])


def per_sample_metrics(x, ref, lpips_fn, batch_size=500):
    out = defaultdict(list)
    for i in range(0, len(x), batch_size):
        xb, rb = x[i:i + batch_size], ref[i:i + batch_size]
        out["lowfreq_rmse_8"].append(lowfreq_rmse(xb, rb, 8).cpu())
        out["lowfreq_rmse_4"].append(lowfreq_rmse(xb, rb, 4).cpu())
        out["lpips"].append(lpips_fn(xb, rb).cpu())
        d_eye, d_img = delta_e(xb, rb)
        out["delta_e_eye"].append(th.from_numpy(d_eye))
        out["delta_e_img"].append(th.from_numpy(d_img))
        for k, v in style_stats(xb).items():
            out[k].append(v.cpu())
    return {k: th.cat(v).double().numpy() for k, v in out.items()}


def style_only(x, batch_size=500):
    out = defaultdict(list)
    for i in range(0, len(x), batch_size):
        for k, v in style_stats(x[i:i + batch_size]).items():
            out[k].append(v.cpu())
    return {k: float(th.cat(v).mean()) for k, v in out.items()}


def anime_features(args, inception):
    """KID の比較対象 (anime_list) と、それに含まれない holdout の特徴。キャッシュする。"""
    with open(args.anime_list) as f:
        ref_names = [line.strip() for line in f if line.strip()]
    cache = os.path.join(args.cache_dir, f"inception_{os.path.splitext(os.path.basename(args.anime_list))[0]}"
                                         f"_holdout{args.num_holdout}.pt")
    if os.path.exists(cache):
        return th.load(cache)
    in_ref = set(ref_names)
    holdout = sorted(random.Random(1).sample([n for n in list_images(args.anime_dir) if n not in in_ref],
                                             args.num_holdout))
    feats = {}
    for key, names in (("ref", ref_names), ("holdout", holdout)):
        x = load_stack([os.path.join(args.anime_dir, n) for n in names])
        feats[key] = inception(x)
        feats[f"{key}_style"] = style_only(x)
    os.makedirs(args.cache_dir, exist_ok=True)
    th.save(feats, cache)
    return feats


def kid_row(feats, anime_ref, args):
    k, s, size = kid(feats, anime_ref, args.kid_subset_size, args.kid_subsets)
    return {"kid": k, "kid_std": s, "kid_subset_size": size}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dirs", required=True, help="run.json のあるディレクトリ (カンマ区切り・glob 可)")
    p.add_argument("--out_csv", required=True)
    p.add_argument("--anime_dir", default="/data/anime_eye")
    p.add_argument("--anime_list", default="lists/anime_eye_kidref10000.txt")
    p.add_argument("--num_holdout", type=int, default=3000)
    p.add_argument("--kid_subset_size", type=int, default=1000)
    p.add_argument("--kid_subsets", type=int, default=100)
    p.add_argument("--cache_dir", default="outputs/eval_cache")
    p.add_argument("--device", default="cuda" if th.cuda.is_available() else "cpu")
    args = p.parse_args()

    run_dirs = sorted({d for pat in args.run_dirs.split(",") for d in glob.glob(pat.strip())
                       if os.path.exists(os.path.join(d, "run.json"))})
    assert run_dirs, f"run.json のあるディレクトリが無い: {args.run_dirs}"
    inception = InceptionFeatures(args.device)
    lpips_fn = LPIPS(args.device)
    anime = anime_features(args, inception)

    rows = [{"run": "anime_holdout", **kid_row(anime["holdout"], anime["ref"], args),
             "n_samples": len(anime["holdout"]), **anime["holdout_style"]}]
    seen_refs = set()
    for d in run_dirs:
        with open(os.path.join(d, "run.json")) as f:
            run = json.load(f)
        outs = run["outputs"]
        x = load_stack([os.path.join(d, o["sample"]) for o in outs]).to(args.device)
        ref = load_stack([os.path.join(d, o["ref_png"]) for o in outs]).to(args.device)
        m = per_sample_metrics(x, ref, lpips_fn)
        has_cond = "cond_png" in outs[0]
        if has_cond:
            cref = load_stack([os.path.join(d, o["cond_png"]) for o in outs]).to(args.device)
            mc = per_sample_metrics(x, cref, lpips_fn)
            m.update({f"{k}_cond": mc[k] for k in PAIR_KEYS})
        pair_keys = PAIR_KEYS + tuple(f"{k}_cond" for k in PAIR_KEYS if has_cond)
        feats = inception(x)

        # 参照画像そのもの (参照画像の組ごとに 1 回)
        ref_key = tuple(sorted({o["ref"] for o in outs}))
        if ref_key not in seen_refs:
            seen_refs.add(ref_key)
            uniq = sorted({o["ref_png"] for o in outs})
            r = load_stack([os.path.join(d, u) for u in uniq])
            rows.append({"run": f"ref_images({len(uniq)})", **kid_row(inception(r), anime["ref"], args),
                         "n_samples": len(uniq), **style_only(r)})

        if has_cond:  # 色を写した参照画像そのもの (写像だけでどこまでアニメに近づくかの目安)
            uniq = sorted({o["cond_png"] for o in outs})
            r = load_stack([os.path.join(d, u) for u in uniq])
            rows.append({"run": f"ref_cond({os.path.relpath(d)})", **kid_row(inception(r), anime["ref"], args),
                         "n_samples": len(uniq), **style_only(r)})

        seeds = sorted({o["seed"] for o in outs})
        seed_idx = {s: [i for i, o in enumerate(outs) if o["seed"] == s] for s in seeds}
        kid_seeds = [kid(feats[idx], anime["ref"], args.kid_subset_size, args.kid_subsets)[0]
                     for idx in seed_idx.values()]
        # 同じ参照画像のシード違いの出力どうしの平均絶対差
        by_ref = defaultdict(list)
        for i, o in enumerate(outs):
            by_ref[o["ref"]].append(i)
        mads = [(x[a] - x[b]).abs().mean().item() for idx in by_ref.values()
                for k, a in enumerate(idx) for b in idx[k + 1:]]

        a = run["args"]
        row = {"run": os.path.relpath(d), "model": os.path.basename(run["model_path"]),
               "down_N": a["down_N"], "down_size": a.get("down_size", 0), "range_t": a["range_t"],
               "t_ref": a["t_ref"], "resizer": a["resizer"],
               "color_transfer": a.get("color_transfer", "none"), "color_strength": a.get("color_strength", ""),
               "mask": "eye_mask" if a.get("mask_dir") else "", "mask_dilate": a.get("mask_dilate", ""),
               "mask_mode": a.get("mask_mode", "") if a.get("mask_dir") else "",
               "timestep_respacing": a["timestep_respacing"], "n_refs": len(by_ref), "seeds": len(seeds),
               "n_samples": len(outs), **{k: float(m[k].mean()) for k in pair_keys + STYLE_KEYS},
               **kid_row(feats, anime["ref"], args),
               "kid_seed_mean": float(np.mean(kid_seeds)), "kid_seed_std": float(np.std(kid_seeds)),
               "seed_mad": float(np.mean(mads)) if mads else "",
               "sec_per_sample": run.get("sec_per_sample"), "peak_vram_mb": run.get("peak_vram_mb")}
        rows.append(row)
        with open(os.path.join(d, "per_sample.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ref", "seed", "sample", *pair_keys, *STYLE_KEYS])
            for i, o in enumerate(outs):
                w.writerow([o["ref"], o["seed"], o["sample"], *[f"{m[k][i]:.6g}" for k in pair_keys + STYLE_KEYS]])
        print(f"{row['run']}: kid={row['kid']:.4f} lpips={row['lpips']:.3f} lf8={row['lowfreq_rmse_8']:.3f} "
              f"dE_eye={row['delta_e_eye']:.1f} lap={row['laplacian_var']:.4f}")

    keys = list(dict.fromkeys(k for r in rows for k in r))
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"-> {args.out_csv}")


if __name__ == "__main__":
    main()
