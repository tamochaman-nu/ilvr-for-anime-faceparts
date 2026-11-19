"""
評価に使う画像のリストを作る (lists/ に書き出し、以降の実行は --ref_list でこれを読む)。

  real_eye_eval<N>.txt    ILVR の参照画像 (評価用。KID・LPIPS などはこの全件で測る)
  real_eye_grid<M>.txt    グリッド画像用 (eval の中から選ぶ)
  anime_eye_kidref<K>.txt KID の比較対象にするアニメ目の学習データ
  ref_lists.json          選別の条件と、条件ごとに除いた件数

実写の目は、パーツ切り出し (anime-face-alignment/util/part_crop/crop_parts.py) が parts.jsonl に残した
スコアで選別する。サングラスは切り出しの段階で除かれているが、透明な眼鏡・閉じた目・髪で隠れた目・
横向きの目は残っているので、ここで除く:
  glasses < max_glasses   目の凸包のうち、顔パーツ分割で眼鏡と判定された画素の割合 (0 か 1 付近に分かれる)
  eye_px >= min_eye_px    目の凸包のうち、目と判定された画素の割合。閉じた目・細めた目で小さい
                          (目視で 0.5 未満は大半が閉じた目か細めた目、0.5 以上は大半が開いた目)
  vis >= min_vis          パーツを囲む楕円のうち、髪・帽子などで隠れていない割合
  |yaw| <= max_abs_yaw    推定ヨー角 (度)

使い方: docker compose run --rm make-ref-lists
"""

import argparse
import json
import os
import random
from collections import Counter

from guided_diffusion.ilvr_eye import list_images, select_refs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real_dir", default="/data/real_eye")
    p.add_argument("--real_parts_jsonl", default="/data/real_eye_parts.jsonl")
    p.add_argument("--anime_dir", default="/data/anime_eye")
    p.add_argument("--max_glasses", type=float, default=0.05)
    p.add_argument("--min_eye_px", type=float, default=0.5)
    p.add_argument("--min_vis", type=float, default=0.9)
    p.add_argument("--max_abs_yaw", type=float, default=30.0)
    p.add_argument("--num_eval", type=int, default=1000)
    p.add_argument("--num_grid", type=int, default=16)
    p.add_argument("--num_anime_ref", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="lists")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    real_files = set(list_images(args.real_dir))
    excluded, kept = Counter(), set()
    with open(args.real_parts_jsonl) as f:
        for line in f:
            r = json.loads(line)
            name = os.path.basename(r["file"])
            if r["part"] != "eye" or name not in real_files:
                continue
            reasons = []
            if r["glasses"] >= args.max_glasses:
                reasons.append("glasses")
            if r["eye_px"] < args.min_eye_px:
                reasons.append("eye_px")
            if r["vis"] < args.min_vis:
                reasons.append("vis")
            if abs(r["yaw"]) > args.max_abs_yaw:
                reasons.append("yaw")
            excluded.update(reasons)
            if not reasons:
                kept.add(name)

    eval_names = select_refs(args.real_dir, args.num_eval, seed=args.seed, candidates=kept)
    grid_names = select_refs(args.real_dir, args.num_grid, seed=args.seed, candidates=eval_names)
    anime_names = sorted(random.Random(args.seed).sample(list_images(args.anime_dir), args.num_anime_ref))

    lists = {f"real_eye_eval{args.num_eval}.txt": eval_names, f"real_eye_grid{args.num_grid}.txt": grid_names,
             f"anime_eye_kidref{args.num_anime_ref}.txt": anime_names}
    for fname, names in lists.items():
        with open(os.path.join(args.out_dir, fname), "w") as f:
            f.write("\n".join(names) + "\n")
    meta = {"args": vars(args), "num_real_images": len(real_files), "num_kept": len(kept),
            "excluded_by_reason": dict(excluded), "lists": {k: len(v) for k, v in lists.items()}}
    with open(os.path.join(args.out_dir, "ref_lists.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
