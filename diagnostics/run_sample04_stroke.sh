#!/bin/bash
# sample04 の head 方式の切り出し (FFHQ と同じ回転) で、ストローク化 (face_input_transform.stroke_*) x 変換の設定を回し、
# 元のフレームへ貼り戻す。前段: scripts/align_faces.py --mode head と scripts/parse_faces.py。
# 使い方 (ホストで): bash diagnostics/run_sample04_stroke.sh [出力先 (既定 outputs/sample04_stroke)]
#   RUNS="generic:P1 parts2:P2" のように環境変数で組を絞れる
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-outputs/sample04_stroke}
IN=input/sample04-12119910
AL=${IN}_aligned256_head_rotffhq
MODEL=ddpm/anime-face256-mp128-ema_0.9995_250000.pt
CFG=ddpm/face256-mp128-ema_0.9995_250000.infer.cfg
declare -A SET=(
  [SD]="--t0_ratio 0.5 --use_ilvr False --sdedit_repeats 1"
  [SD30]="--t0_ratio 0.3 --use_ilvr False --sdedit_repeats 1"
  [SD40]="--t0_ratio 0.4 --use_ilvr False --sdedit_repeats 1"
  [SD60]="--t0_ratio 0.6 --use_ilvr False --sdedit_repeats 1"
  [SD70]="--t0_ratio 0.7 --use_ilvr False --sdedit_repeats 1"
  [P1]="--t0_ratio 0.5 --down_N 16 --ilvr_stop_ratio 0.1 --sdedit_repeats 1"
  [P2]="--t0_ratio 0.7 --down_N 16 --ilvr_stop_ratio 0.2 --sdedit_repeats 2"
)
declare -A VAR=(
  [none]=""
  [generic]="--stroke generic --stroke_segments 200 --stroke_merge 8"
  [parts1]="--stroke parts --stroke_parse_dir ${AL}_parse --stroke_shade_levels 1"
  [parts2]="--stroke parts --stroke_parse_dir ${AL}_parse --stroke_shade_levels 2"
  [parts2eye]="--stroke parts --stroke_parse_dir ${AL}_parse --stroke_shade_levels 2 --stroke_eye_scale_w 1.2 --stroke_eye_scale_h 1.8"
  [parts2iris]="--stroke parts --stroke_parse_dir ${AL}_parse --stroke_shade_levels 2 --stroke_eye_scale_w 1.2 --stroke_eye_scale_h 1.8 --stroke_eye_style iris"
  [parts2irisL]="--stroke parts --stroke_parse_dir ${AL}_parse --stroke_shade_levels 2 --stroke_eye_scale_w 1.3 --stroke_eye_scale_h 2.5 --stroke_eye_style iris"
  [parts1iris]="--stroke parts --stroke_parse_dir ${AL}_parse --stroke_shade_levels 1 --stroke_eye_scale_w 1.2 --stroke_eye_scale_h 1.8 --stroke_eye_style iris"
)
RUNS=${RUNS:-"none:SD generic:SD parts1:SD parts2:SD parts2eye:SD generic:P1 parts1:P1 parts2:P1 parts2eye:P1 generic:P2 parts1:P2 parts2:P2 parts2eye:P2"}
for r in $RUNS; do
  V=${r%%:*}; S=${r##*:}
  D=$OUT/$V/$S
  docker compose run --rm shell bash -c "
    python scripts/sdedit_sample.py --model_path $MODEL --cfg $CFG --base_samples $AL \
      --out_dir $D --use_sdedit ${SET[$S]} ${VAR[$V]} --color_transfer lab_meanstd_l --seed 0 &&
    python scripts/paste_back_faces.py --run_dir $D --frames_dir $IN --quads $AL/quads.json \
      --grid Cam1_frame_00001,Cam1_frame_00008,Cam2_frame_00006,Cam3_frame_00005,Cam4_frame_00006,Cam5_frame_00006
  " 2>&1 | grep -v -i "warn\|license\|NVIDIA\|^$\|copyright\|container\|CUDA\|===" || true
done
