#!/usr/bin/env bash
# =====================================================================
# 多数据集噪声鲁棒性基准（4090 GPU）
# 用法:
#   bash scripts/run_bench_all.sh                 # 完整 9 数据集
#   bash scripts/run_bench_all.sh --quick         # 精简: ETTh1,ETTh2,ETTm1,ETTm2 + exchange + electricity
#   bash scripts/run_bench_all.sh --epochs 10 --seeds 0,1,2
# =====================================================================
set -e
cd "$(dirname "$0")/.."

EPOCHS=10
SEEDS="0"
DATASETS="ETTh1,ETTh2,ETTm1,ETTm2,exchange,weather,electricity,traffic,illness"
DEVICE="auto"
WORKERS=4

[ "$1" = "--quick" ] && DATASETS="ETTh1,ETTh2,ETTm1,ETTm2,exchange,electricity"
[ "$1" = "--epochs" ] && { EPOCHS="$2"; shift 2; }

mkdir -p results
# 清掉上次残留的 jsonl，保证汇总表只来自本次运行
rm -f results/*.jsonl results/multi_summary.md

# --- 依赖检查 ---
python - <<'PY'
import importlib
missing = [m for m in ('numpy', 'pandas', 'torch', 'sklearn') if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit('missing deps: ' + ', '.join(missing))
print('[check] deps ok')
PY

# --- 数据检查（缺哪个提示补哪个）---
python - <<'PY'
import os
need = {
    'dataset/ETT-small/ETTh1.csv': 'ETTh1/ETTh2/ETTm1/ETTm2(放 ETT-small)',
    'dataset/ETT-small/ETTm1.csv': '',
    'dataset/ETT-small/ETTm2.csv': '',
    'dataset/ETT-small/ETTh2.csv': '',
    'dataset/exchange_rate/exchange_rate.csv': 'Exchange',
    'dataset/weather/weather.csv': 'Weather',
    'dataset/electricity/electricity.csv': 'Electricity',
    'dataset/traffic/traffic.csv': 'Traffic',
    'dataset/illness/national_illness.csv': 'ILI',
}
missing = [p for p in need if not os.path.exists(p)]
if missing:
    for m in missing:
        print('[data] MISSING:', m, need[m])
    print('请先从 https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy 下载缺失数据集放到 dataset/')
    exit(0)
print('[check] data ok')
PY

echo "==== run: datasets=$DATASETS epochs=$EPOCHS seeds=$SEEDS device=$DEVICE ===="
python scripts/noise_robustness.py \
  --models "DLinear,FuzzyDLinear:membership" \
  --protocols clean,noisy \
  --seeds "$SEEDS" \
  --epochs "$EPOCHS" \
  --amps 0,1,3,5,10 \
  --datasets "$DATASETS" \
  --out ./results/bench_multi.jsonl \
  --device "$DEVICE" \
  --num_workers "$WORKERS" \
  2>&1 | tee results/bench_multi.log

echo "==== done. 生成汇总表 ===="
python scripts/summarize_bench.py --input_dir results --out results/multi_summary.md
echo "==== summary ===="
cat results/multi_summary.md