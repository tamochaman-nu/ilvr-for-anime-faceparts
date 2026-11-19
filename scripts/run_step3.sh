#!/bin/bash
# Step 3: 改善案の比較。Phase 3 と同じ参照画像 (lists/real_eye_eval1000.txt) x 3 シードで生成し、同じ指標で評価する。
#   a. 色の写像: N ∈ {4,8,16} (range_t=40) x {lab_meanstd, lab_quantile}
#   b. マスク付き ILVR (composite): N ∈ {4,8} (range_t=40) x 膨張 {0,3} 画素 x 色の写像 {なし, lab_meanstd}
#      比較用に、補正量にマスクを掛けるだけの multiply (発散する) を N8 で 1 条件
#   c. 中間の縮小サイズ: 6x6 / 3x3 (range_t=40) x 色の写像 {なし, lab_meanstd}
# 比較の基準 (Phase 3 の N{4,8,16}_rt40 と uncond) は outputs/phase3 から評価に加える。
# 先に docker compose run --rm fit-color-stats で lists/color_stats.json を作っておく。
# 使い方: docker compose run --rm step3   (環境変数 SEEDS, OUT_ROOT, MODEL_PATH などで上書きできる)
set -e
cd /app

MODEL_PATH="${MODEL_PATH:-ddpm/anime-eye32-mp64-ema_0.9999_200000.pt}"
MODEL_CFG="${MODEL_CFG:-ddpm/anime-eye32-mp64.cfg}"
OUT_ROOT="${OUT_ROOT:-outputs/step3}"
PHASE3_ROOT="${PHASE3_ROOT:-outputs/phase3}"
SEEDS="${SEEDS:-0,1,2}"
REF_LIST="${REF_LIST:-lists/real_eye_eval1000.txt}"
GRID_LIST="${GRID_LIST:-lists/real_eye_grid16.txt}"
BATCH_SIZE="${BATCH_SIZE:-500}"
MASK_DIR="${MASK_DIR:-/data/real_eye_mask}"
RT=40

sample() {  # sample <出力名> <ilvr_sample_eye.py に追加で渡す引数...>
  local name="$1"; shift
  if [ -f "$OUT_ROOT/$name/run.json" ]; then echo "skip $name (run.json あり)"; return; fi
  python scripts/ilvr_sample_eye.py --model_path "$MODEL_PATH" --cfg "$MODEL_CFG" \
    --base_samples /data/real_eye --ref_list "$REF_LIST" --range_t "$RT" --timestep_respacing 100 \
    --seeds "$SEEDS" --batch_size "$BATCH_SIZE" --out_dir "$OUT_ROOT/$name" "$@"
}

# a. 色の写像
for N in 4 8 16; do
  for CT in lab_meanstd lab_quantile; do
    sample "N${N}_rt${RT}_${CT}" --down_N "$N" --color_transfer "$CT"
  done
done
# b. マスク付き ILVR
for N in 4 8; do
  for DIL in 0 3; do
    sample "N${N}_rt${RT}_mask${DIL}" --down_N "$N" --mask_dir "$MASK_DIR" --mask_dilate "$DIL"
    sample "N${N}_rt${RT}_mask${DIL}_lab_meanstd" --down_N "$N" --mask_dir "$MASK_DIR" --mask_dilate "$DIL" \
      --color_transfer lab_meanstd
  done
done
sample "N8_rt${RT}_mask0_multiply" --down_N 8 --mask_dir "$MASK_DIR" --mask_mode multiply
# c. 中間の縮小サイズ
for S in 6 3; do
  sample "S${S}_rt${RT}" --down_size "$S"
  sample "S${S}_rt${RT}_lab_meanstd" --down_size "$S" --color_transfer lab_meanstd
done

base="$PHASE3_ROOT/uncond,$PHASE3_ROOT/N4_rt40,$PHASE3_ROOT/N8_rt40,$PHASE3_ROOT/N16_rt40"
python scripts/eval_ilvr_eye.py --run_dirs "$base,$OUT_ROOT/*" --out_csv "$OUT_ROOT/metrics.csv" \
  --anime_dir /data/anime_eye --anime_list lists/anime_eye_kidref10000.txt

# グリッド (シード 0): 案ごとに、Phase 3 の基準と並べる
grid() {  # grid <出力ファイル名> <見出し=ディレクトリ,...>
  python scripts/make_ilvr_grid.py --ref_list "$GRID_LIST" --seeds 0 --runs "$2" --out "$OUT_ROOT/$1"
}
runs=""
for N in 4 8 16; do
  runs="${runs:+$runs,}N${N}=$PHASE3_ROOT/N${N}_rt40,N${N} meanstd=$OUT_ROOT/N${N}_rt40_lab_meanstd"
  runs="$runs,N${N} quantile=$OUT_ROOT/N${N}_rt40_lab_quantile"
done
grid grid_a_color.png "$runs"
runs=""
for N in 4 8; do
  runs="${runs:+$runs,}N${N}=$PHASE3_ROOT/N${N}_rt40"
  for DIL in 0 3; do
    runs="$runs,N${N} mask${DIL}=$OUT_ROOT/N${N}_rt40_mask${DIL},N${N} mask${DIL} ms=$OUT_ROOT/N${N}_rt40_mask${DIL}_lab_meanstd"
  done
done
runs="$runs,N8 mask0 multiply=$OUT_ROOT/N8_rt40_mask0_multiply"
grid grid_b_mask.png "$runs"
grid grid_c_size.png "N4 (8x8)=$PHASE3_ROOT/N4_rt40,S6=$OUT_ROOT/S6_rt40,S6 ms=$OUT_ROOT/S6_rt40_lab_meanstd,N8 (4x4)=$PHASE3_ROOT/N8_rt40,S3=$OUT_ROOT/S3_rt40,S3 ms=$OUT_ROOT/S3_rt40_lab_meanstd,N16 (2x2)=$PHASE3_ROOT/N16_rt40"
