"""
色の写像 (guided_diffusion/ilvr_ref_transform.ColorTransfer) に使う CIELAB の統計を求めて JSON に保存する。

  real   実写の目: make_ref_lists.py と同じ条件 (lists/ref_lists.json の args) で選別した画像のうち、
         評価用の参照画像 (real_eye_eval*.txt) を除いたものから num 枚
  anime  アニメ目の学習データ: KID の比較対象 (anime_eye_kidref*.txt) を除いたものから num 枚
評価に使う画像を統計に含めないので、写像が評価用の画像に合わせ込まれることはない。
--real_mask_dir / --anime_mask_dir を渡すと、マスクが 0.5 以上の画素 (マスク済みの画像では白い背景を除いた
目の部分) だけで統計を求める。ディレクトリに {side} を含めると左右別のフォルダから読む (ilvr_eye.image_path)。

使い方: docker compose run --rm fit-color-stats
"""

import argparse
import json
import os
import random

import torch as th

from guided_diffusion.ilvr_eye import image_path, list_images, load_image
from guided_diffusion.ilvr_ref_transform import lab_stats, load_masks


def read_list(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def selected_real(real_dir, parts_jsonl, a):
    """make_ref_lists.py と同じ選別条件 (a はその args) を満たす実写の目のファイル名。"""
    real_files = set(list_images(real_dir))
    kept = set()
    with open(parts_jsonl) as f:
        for line in f:
            r = json.loads(line)
            name = os.path.basename(r["file"])
            if (r["part"] == "eye" and name in real_files and r["glasses"] < a["max_glasses"]
                    and r["eye_px"] >= a["min_eye_px"] and r["vis"] >= a["min_vis"]
                    and abs(r["yaw"]) <= a["max_abs_yaw"]):
                kept.add(name)
    return kept


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real_dir", default="/data/real_eye")
    p.add_argument("--real_parts_jsonl", default="/data/real_eye_parts.jsonl")
    p.add_argument("--anime_dir", default="/data/anime_eye")
    p.add_argument("--real_mask_dir", default="")
    p.add_argument("--anime_mask_dir", default="")
    p.add_argument("--ref_lists_json", default="lists/ref_lists.json")
    p.add_argument("--eval_list", default="lists/real_eye_eval1000.txt")
    p.add_argument("--anime_list", default="lists/anime_eye_kidref10000.txt")
    p.add_argument("--num", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="lists/color_stats.json")
    args = p.parse_args()

    with open(args.ref_lists_json) as f:
        sel = json.load(f)["args"]
    real_pool = sorted(selected_real(args.real_dir, args.real_parts_jsonl, sel) - set(read_list(args.eval_list)))
    anime_pool = sorted(set(list_images(args.anime_dir)) - set(read_list(args.anime_list)))
    rng = random.Random(args.seed)
    out = {"args": vars(args), "selection": sel}
    for key, d, md, pool in (("real", args.real_dir, args.real_mask_dir, real_pool),
                             ("anime", args.anime_dir, args.anime_mask_dir, anime_pool)):
        names = rng.sample(pool, min(args.num, len(pool)))
        x = th.stack([load_image(image_path(d, n), 32) for n in names])
        masks = load_masks(md, names, 32) if md else None
        out[key] = {"num": len(names), "pool": len(pool), **lab_stats(x, masks)}
        print(f"{key}: {len(names)} / {len(pool)} 枚, L*a*b* 平均 {[round(v, 2) for v in out[key]['mean']]}, "
              f"標準偏差 {[round(v, 2) for v in out[key]['std']]}")
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
