#!/bin/bash
# 目DDPM (anime-eye32-mp64) × ILVR で、実写の目のクロップをアニメ調の目に変換する。
# モデルの設定は学習時に保存された train.cfg (MODEL_CFG) から読むので、ここではサンプリングの条件だけを決める。
# 値は環境変数で上書きできる (例: docker compose run --rm -e DOWN_N=16 -e RANGE_T=40 ilvr-eye)。
set -e
cd /app

MODEL_PATH="${MODEL_PATH:-ddpm/anime-eye32-mp64-ema_0.9999_200000.pt}"
MODEL_CFG="${MODEL_CFG:-ddpm/anime-eye32-mp64.cfg}"
BASE_SAMPLES="${BASE_SAMPLES:-/data/real_eye}"

DOWN_N="${DOWN_N:-8}"             # 0 なら条件付けしない (無条件生成。評価の比較基準)
RANGE_T="${RANGE_T:-20}"          # timestep_respacing=100 での間引き後のステップ数。i > RANGE_T のステップだけ補正する
T_REF="${T_REF:-t}"               # t (公式実装と同じ) / t-1 (論文の Algorithm 1 と同じ)
RESIZER="${RESIZER:-resizer}"     # resizer (公式の Resizer) / interpolate (F.interpolate, antialias)
DOWN_SIZE="${DOWN_SIZE:-0}"       # > 0 なら縮小後の一辺を直接指定 (DOWN_N より優先。Step 3 の推奨は 3)
COLOR_TRANSFER="${COLOR_TRANSFER:-none}"  # none / lab_meanstd / lab_quantile (Step 3 の推奨は lab_meanstd)
MASK_DIR="${MASK_DIR:-}"          # 目のマスクのディレクトリ (例: /data/real_eye_mask)。空ならマスクなし
TIMESTEP_RESPACING="${TIMESTEP_RESPACING:-100}"
SEEDS="${SEEDS:-0}"
NUM_REFS="${NUM_REFS:-16}"        # 0 なら BASE_SAMPLES の全画像
REF_SEED="${REF_SEED:-0}"
REF_LIST="${REF_LIST:-}"
BATCH_SIZE="${BATCH_SIZE:-64}"
OUT_DIR="${OUT_DIR:-outputs/ilvr_eye/N${DOWN_N}_S${DOWN_SIZE}_range${RANGE_T}_${T_REF}_${RESIZER}_${COLOR_TRANSFER}}"

python scripts/ilvr_sample_eye.py \
  --model_path "$MODEL_PATH" --cfg "$MODEL_CFG" \
  --ema_model_path "${EMA_MODEL_PATH:-}" --use_ema "${USE_EMA:-False}" \
  --base_samples "$BASE_SAMPLES" --ref_list "$REF_LIST" --num_refs "$NUM_REFS" --ref_seed "$REF_SEED" \
  --down_N "$DOWN_N" --down_size "$DOWN_SIZE" --range_t "$RANGE_T" --t_ref "$T_REF" --resizer "$RESIZER" \
  --color_transfer "$COLOR_TRANSFER" --mask_dir "$MASK_DIR" --mask_dilate "${MASK_DILATE:-0}" \
  --timestep_respacing "$TIMESTEP_RESPACING" --seeds "$SEEDS" --batch_size "$BATCH_SIZE" \
  --out_dir "$OUT_DIR" "$@"
