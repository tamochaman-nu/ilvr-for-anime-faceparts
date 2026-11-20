#!/bin/bash
# 左右の目を別々に学習した目DDPM (anime-eye32-mp64-left / -right) と、ランドマークの楕円の外を白で埋めた
# マスク済みの目 (crop_parts.py --split_eye_sides の eye_{l,r}_masked/) で、比較資料 (代理画像の有無) の条件を
# 実行し直す。参照画像の名前・評価の指標・乱数は Phase 3 / Step 3 と同じで、データとモデルだけが違う。
#   比較の基準: 無条件生成、N4 / N8 / N16 / 6x6 / 3x3 (range_t=40)
#   色の写像 (lab_meanstd_l。統計も適用もマスクの部分だけ): N8 / 3x3
#   代理画像 (色の写像 + 代理画像): N4 / 6x6 / 3x3 x 実写との混合比 {1.0, 0.5}
# 参照画像の l / r (画像の左側の目が l) に応じて left / right のモデルを使い分ける。
# KID の比較対象は、アニメ目の学習データ (左右別・マスク済み) のうち lists/anime_eye_kidref10000.txt の画像。
# 使い方: docker compose run --rm lr-masked   (環境変数 SEEDS, OUT_ROOT などで上書きできる)
set -e
cd /app

MODEL_PATH_L="${MODEL_PATH_L:-ddpm/anime-eye32-mp64-left-ema_0.9999_200000.pt}"
MODEL_PATH_R="${MODEL_PATH_R:-ddpm/anime-eye32-mp64-right-ema_0.9999_200000.pt}"
MODEL_CFG="${MODEL_CFG:-ddpm/train.cfg}"
OUT_ROOT="${OUT_ROOT:-outputs/lr_masked}"
SEEDS="${SEEDS:-0,1,2}"
REF_LIST="${REF_LIST:-lists/real_eye_eval1000.txt}"
GRID_LIST="${GRID_LIST:-lists/real_eye_grid16.txt}"
BATCH_SIZE="${BATCH_SIZE:-500}"
REAL_DIR="/data/real_eye_split/eye_{side}_masked"
REAL_MASK_DIR="/data/real_eye_split/eye_{side}_mask"
ANIME_DIR="/data/anime_eye_split/eye_{side}_masked"
ANIME_MASK_DIR="/data/anime_eye_split/eye_{side}_mask"
COLOR_STATS="${COLOR_STATS:-lists/color_stats_masked.json}"
RT=40

if [ ! -f "$COLOR_STATS" ]; then
  python scripts/fit_color_stats.py --real_dir "$REAL_DIR" --real_mask_dir "$REAL_MASK_DIR" \
    --anime_dir "$ANIME_DIR" --anime_mask_dir "$ANIME_MASK_DIR" --out "$COLOR_STATS"
fi

sample() {  # sample <出力名> <ilvr_sample_eye.py に追加で渡す引数...>
  local name="$1"; shift
  if [ -f "$OUT_ROOT/$name/run.json" ]; then echo "skip $name (run.json あり)"; return; fi
  python scripts/ilvr_sample_eye.py --model_path_l "$MODEL_PATH_L" --model_path_r "$MODEL_PATH_R" \
    --cfg "$MODEL_CFG" --base_samples "$REAL_DIR" --ref_list "$REF_LIST" --range_t "$RT" \
    --timestep_respacing 100 --seeds "$SEEDS" --batch_size "$BATCH_SIZE" --out_dir "$OUT_ROOT/$name" "$@"
}
color=(--color_transfer lab_meanstd_l --color_stats "$COLOR_STATS" --color_mask_dir "$REAL_MASK_DIR")

sample uncond --down_N 0
for N in 4 8 16; do sample "N${N}_rt${RT}" --down_N "$N"; done
for S in 6 3; do sample "S${S}_rt${RT}" --down_size "$S"; done
sample "N8_rt${RT}_color" --down_N 8 "${color[@]}"
sample "S3_rt${RT}_color" --down_size 3 "${color[@]}"
for B in 1.0 0.5; do
  sample "N4_rt${RT}_proxy${B}" --down_N 4 "${color[@]}" --proxy ellipse --proxy_mask_dir "$REAL_MASK_DIR" \
    --proxy_blend "$B"
  for S in 6 3; do
    sample "S${S}_rt${RT}_proxy${B}" --down_size "$S" "${color[@]}" --proxy ellipse \
      --proxy_mask_dir "$REAL_MASK_DIR" --proxy_blend "$B"
  done
done

python scripts/eval_ilvr_eye.py --run_dirs "$OUT_ROOT/*" --out_csv "$OUT_ROOT/metrics.csv" \
  --anime_dir "$ANIME_DIR" --anime_list lists/anime_eye_kidref10000.txt

grid() {  # grid <出力ファイル名> <シード> <見出し=ディレクトリ,...>
  python scripts/make_ilvr_grid.py --ref_list "$GRID_LIST" --seeds "$2" --runs "$3" --out "$OUT_ROOT/$1"
}
O="$OUT_ROOT"
grid 01_baseline.png 0 "N4=$O/N4_rt40,N8=$O/N8_rt40,N16=$O/N16_rt40,uncond=$O/uncond"
grid 02_color_transfer.png 0 "N8=$O/N8_rt40,N8 color=$O/N8_rt40_color,S3=$O/S3_rt40,S3 color=$O/S3_rt40_color"
grid 04_compare_seed0.png 0 "N8=$O/N8_rt40,S3 color=$O/S3_rt40_color,N4 proxy1.0=$O/N4_rt40_proxy1.0,N4 proxy0.5=$O/N4_rt40_proxy0.5,S6 proxy0.5=$O/S6_rt40_proxy0.5,S3 proxy0.5=$O/S3_rt40_proxy0.5"
grid 05_diversity_seeds.png "$SEEDS" "N4 proxy1.0=$O/N4_rt40_proxy1.0,N4 proxy0.5=$O/N4_rt40_proxy0.5,S3 color=$O/S3_rt40_color"
grid 06_proxy_all_conditions.png 0 "N4=$O/N4_rt40,N4 proxy1.0=$O/N4_rt40_proxy1.0,N4 proxy0.5=$O/N4_rt40_proxy0.5,S6=$O/S6_rt40,S6 proxy1.0=$O/S6_rt40_proxy1.0,S6 proxy0.5=$O/S6_rt40_proxy0.5,S3=$O/S3_rt40,S3 proxy1.0=$O/S3_rt40_proxy1.0,S3 proxy0.5=$O/S3_rt40_proxy0.5"
