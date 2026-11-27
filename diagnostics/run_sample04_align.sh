#!/bin/bash
# sample04 (avatar-bench のマルチビュー、input/sample04-12119910) で、切り出しの方式 x 変換の代表点を回し、
# 元のフレームへ貼り戻す。前段 (ランドマークと切り出し) は scripts/detect_face_landmarks.py と
# scripts/align_faces.py (--mode ffhq / head --rotation ffhq / head --rotation none) で作っておく。
# 使い方 (ホストで): bash diagnostics/run_sample04_align.sh [出力先 (既定 outputs/sample04_align)]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-outputs/sample04_align}
IN=input/sample04-12119910
MODEL=ddpm/anime-face256-mp128-ema_0.9995_250000.pt
CFG=ddpm/face256-mp128-ema_0.9995_250000.infer.cfg
# 名前: SDEdit + ILVR の引数 (格子状の探索 diagnostics/sdedit_grid_search.py の代表点)
declare -A SET=(
  [P1]="--t0_ratio 0.5 --down_N 16 --ilvr_stop_ratio 0.1 --sdedit_repeats 1"
  [P2]="--t0_ratio 0.7 --down_N 16 --ilvr_stop_ratio 0.2 --sdedit_repeats 2"
  [P3]="--t0_ratio 0.7 --down_N 32 --ilvr_stop_ratio 0.2 --sdedit_repeats 2"
)
RUNS=${RUNS:-"ffhq:P2 head_rotffhq:P2 head_rotnone:P2 head_rotffhq:P1 head_rotnone:P1 head_rotffhq:P3 head_rotnone:P3"}
for r in $RUNS; do
  A=${r%%:*}; S=${r##*:}
  D=$OUT/$A/${S}_ct
  docker compose run --rm shell bash -c "
    python scripts/sdedit_sample.py --model_path $MODEL --cfg $CFG --base_samples ${IN}_aligned256_$A \
      --out_dir $D --use_sdedit ${SET[$S]} --color_transfer lab_meanstd_l --seed 0 &&
    python scripts/paste_back_faces.py --run_dir $D --frames_dir $IN --quads ${IN}_aligned256_$A/quads.json \
      --grid Cam1_frame_00001,Cam1_frame_00008,Cam2_frame_00006,Cam3_frame_00005,Cam4_frame_00006,Cam5_frame_00006
  " 2>&1 | grep -v -i "warn\|license\|NVIDIA\|^$\|copyright\|container" || true
done
