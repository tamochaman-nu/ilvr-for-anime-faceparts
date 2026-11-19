#!/bin/bash
# Phase 3: down_N x range_t の総当たりと評価・グリッド画像。
#   - 本体: down_N ∈ {4,8,16,32} x range_t ∈ {0,10,20,40} (t_ref=t, Resizer)
#   - 比較基準: 無条件生成 (down_N=0)
#   - 補足: N8 rt20 で t_ref=t-1 / interpolate
# 参照画像は lists/real_eye_eval1000.txt (評価) 、グリッドはその中の lists/real_eye_grid16.txt。
# 使い方: docker compose run --rm phase3   (環境変数 SEEDS, OUT_ROOT, MODEL_PATH などで上書きできる)
set -e
cd /app

MODEL_PATH="${MODEL_PATH:-ddpm/anime-eye32-mp64-ema_0.9999_200000.pt}"
MODEL_CFG="${MODEL_CFG:-ddpm/anime-eye32-mp64.cfg}"
OUT_ROOT="${OUT_ROOT:-outputs/phase3}"
SEEDS="${SEEDS:-0,1,2}"
REF_LIST="${REF_LIST:-lists/real_eye_eval1000.txt}"
GRID_LIST="${GRID_LIST:-lists/real_eye_grid16.txt}"
BATCH_SIZE="${BATCH_SIZE:-500}"

sample() {  # sample <出力名> <down_N> <range_t> <t_ref> <resizer>
  if [ -f "$OUT_ROOT/$1/run.json" ]; then echo "skip $1 (run.json あり)"; return; fi
  python scripts/ilvr_sample_eye.py --model_path "$MODEL_PATH" --cfg "$MODEL_CFG" \
    --base_samples /data/real_eye --ref_list "$REF_LIST" \
    --down_N "$2" --range_t "$3" --t_ref "$4" --resizer "$5" --timestep_respacing 100 \
    --seeds "$SEEDS" --batch_size "$BATCH_SIZE" --out_dir "$OUT_ROOT/$1"
}

sample uncond 0 0 t resizer
for N in 4 8 16 32; do
  for RT in 0 10 20 40; do
    sample "N${N}_rt${RT}" "$N" "$RT" t resizer
  done
done
sample N8_rt20_t-1 8 20 t-1 resizer
sample N8_rt20_interp 8 20 t interpolate

python scripts/eval_ilvr_eye.py --run_dirs "$OUT_ROOT/*" --out_csv "$OUT_ROOT/metrics.csv" \
  --anime_dir /data/anime_eye --anime_list lists/anime_eye_kidref10000.txt

# グリッド: 全条件 (シード 0) と、down_N ごとの range_t x シード
runs="uncond=$OUT_ROOT/uncond"
for N in 4 8 16 32; do for RT in 0 10 20 40; do runs="$runs,N${N} rt${RT}=$OUT_ROOT/N${N}_rt${RT}"; done; done
runs="$runs,N8 rt20 t-1=$OUT_ROOT/N8_rt20_t-1,N8 rt20 interp=$OUT_ROOT/N8_rt20_interp"
python scripts/make_ilvr_grid.py --ref_list "$GRID_LIST" --seeds 0 --runs "$runs" --out "$OUT_ROOT/grid_all_seed0.png"
for N in 4 8 16 32; do
  runs=""
  for RT in 0 10 20 40; do runs="${runs:+$runs,}rt${RT}=$OUT_ROOT/N${N}_rt${RT}"; done
  python scripts/make_ilvr_grid.py --ref_list "$GRID_LIST" --seeds "$SEEDS" --runs "$runs" \
    --out "$OUT_ROOT/grid_N${N}_seeds.png"
done
