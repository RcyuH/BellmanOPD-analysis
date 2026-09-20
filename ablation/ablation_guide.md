# Hướng dẫn CMT ablation và phân tích hyperparameter

Thư mục này là lớp thí nghiệm bao quanh implementation CMT chính của
BellmanOPD. Ablation không copy lại công thức và không thay distillation loss,
optimizer, rollout, Gibbs/KL allocator hay evaluator.

## 1. Ba arm và pipeline

CMT production đã tính các tensor detached `gain`, `successor_excess`,
`sequential_gain` và `learning_value`. Ablation chỉ chọn score đưa vào cùng
allocator:

```text
g       : S_t = g_t
g_x     : S_t = g_t + successor_excess_t
g_d     : S_t = learning_value_t = g_t + sequential_gain_t
```

`g_d` dùng trực tiếp canonical `learning_value`, vì vậy phải trùng số với CMT
canonical khi cấu hình giống nhau. `X=successor_excess` không phải `R`, `M`,
`V` hoặc `H`. Pipeline không đổi:

```text
logits -> CMT diagnostics -> S_t -> global KL/Gibbs -> w_t
       -> weighted OPD loss -> optimizer update
```

## 2. Cấu hình chung

Script dùng `SCRIPT_DIR`, nên gọi được từ mọi current working directory. Chỉnh
các export ở đầu `ablation/scripts/train.sh`, hoặc truyền chúng trước lệnh:

```bash
cd /mnt/hdd/nhatminh/OPD/BellmanOPD
export CUDA_VISIBLE_DEVICES=0
export STORAGE_ROOT=/workspace/storage-shared
export STUDENT_MODEL="nlp/tungdd11/stable-on-policy-distillation/OPD/model/Qwen3-1.7B-Base"
export TEACHER_MODEL="models/Qwen3-8B"
export TRAIN_DATASET=competition_math
export GLOBAL_BATCH_SIZE=64
export PPO_MINI_BATCH_SIZE=16
export MICRO_BATCH_SIZE_PER_GPU=8
export SEED=42
export ROLLOUT_SEED=42
export TRAIN_MAX_NEW_TOKENS=4096
export EVAL_MAX_NEW_TOKENS=7168
```

Giá trị canonical của ablation là:

```text
top_k=16, cmt_allocation_kl=0.5, cmt_gamma=1.0,
cmt_successor_lambda=1.0, learning_rate=1e-6,
rollout temperature=1.0, rollout top_p=1.0
```

Đối với gamma, nên bắt đầu với grid ngắn:

```text
0.95, 0.99, 0.995, 0.999, 1.0
```

Trực giác theo effective horizon `1/(1-gamma)` là: 0.95 khoảng 20 bước,
0.99 khoảng 100 bước, 0.995 khoảng 200 bước, 0.999 khoảng 1000 bước. Đây chỉ
là cách diễn giải để chọn grid; `gamma=1.0` vẫn là baseline finite-horizon
canonical và không nên bị bỏ khỏi sweep. Với trajectory 2k--8k token, không
nên bắt đầu bằng nhiều giá trị rất nhỏ (ví dụ 0.5) vì chúng gần như loại bỏ
toàn bộ successor signal.

Training và evaluation có budget token độc lập: 4096 và 7168. `top_p=1.0`
là protocol chính để estimator CMT có diễn giải on-policy chính xác.

## 3. Chạy một arm

```bash
bash ablation/scripts/train.sh g
bash ablation/scripts/train.sh g_x
```

`g_d` chính là score canonical của CMT production (`learning_value`). Vì vậy,
nếu đã có một run CMT với cùng student/teacher, seed, rollout, `top_k`,
`epsilon`, `gamma`, `successor_lambda`, learning rate và evaluation protocol,
không cần chạy lại `ablation/scripts/train.sh g_d`. Khi vẽ arms, truyền run CMT
đó qua `GD_CMT_RUN_NAME` (hoặc `G_D_RUN_NAME`); plotting sẽ đọc trực tiếp
`outputs/<run>/cmt_opd/eval_history.jsonl`, không copy hay sửa file production.

Chỉ chạy `g_d` trong ablation khi cần kiểm tra reproducibility độc lập hoặc khi
CMT production không dùng đúng cấu hình so sánh.

Nếu không đặt `RUN_NAME`, launcher tự tạo tên khoa học dạng:

```text
cmt_<arm>_epsilon<eps>_gamma<gamma>_topk<k>_lr<lr>_seed<seed>_<timestamp>
```

Ví dụ `cmt_g_d_epsilon0p5_gamma1p0_topk16_lr1em6_seed42_20260912_200100`.
Output nằm trong `ablation/outputs/<run-name>/` và không bị ghi đè. Smoke run:

```bash
MAX_STEPS=2 bash ablation/scripts/train.sh g
MAX_STEPS=2 bash ablation/scripts/train.sh g_x
```

Liệt kê các run để lấy tên tự động cho bước plotting:

```bash
find ablation/outputs -mindepth 1 -maxdepth 1 -type d | sort
```

## 4. Một giá trị hyperparameter ở mỗi thời điểm

Các file `sweep_*` sau đây chạy **một full CMT (`g_d`) run**. GPU và giá trị có
thể sửa ngay ở đầu file hoặc truyền inline.

```bash
CUDA_VISIBLE_DEVICES=0 EPSILON=0.25 \
  bash ablation/scripts/sweep_epsilon.sh

CUDA_VISIBLE_DEVICES=0 TOP_K=8 \
  bash ablation/scripts/sweep_topk.sh

CUDA_VISIBLE_DEVICES=0 LR=5e-7 \
  bash ablation/scripts/sweep_lr.sh

CUDA_VISIBLE_DEVICES=0 GAMMA=0.99 \
  bash ablation/scripts/sweep_gamma.sh
```

Các run hoàn toàn độc lập. Ví dụ hôm nay chạy epsilon 0.25, hôm khác chạy
epsilon 0.5:

```bash
CUDA_VISIBLE_DEVICES=0 EPSILON=0.25 \
  RUN_NAME=analysis_epsilon_0.25_seed42 \
  bash ablation/scripts/sweep_epsilon.sh

CUDA_VISIBLE_DEVICES=1 EPSILON=0.5 \
  RUN_NAME=analysis_epsilon_0.5_seed42 \
  bash ablation/scripts/sweep_epsilon.sh
```

Nếu scheduler map GPU mới thành device 0 thì dùng `CUDA_VISIBLE_DEVICES=0` ở
job mới; output vẫn độc lập nhờ `RUN_NAME` khác.

## 5. Nhiều giá trị song song

Để mỗi giá trị là một process độc lập trên một GPU:

```bash
GPU_LIST=0,1,2,3 \
EPSILON_VALUES="0.25 0.5 0.75 1.0" \
  bash ablation/scripts/sweep_parallel_epsilon.sh
```

Script này gán GPU 0/1/2/3 lần lượt cho bốn giá trị, lưu log riêng dưới
`ablation/outputs/_sweep_logs/`, và đợi tất cả job kết thúc. Wrapper tương tự:

```bash
GPU_LIST=0,1,2 TOP_K_VALUES="8 16 32" \
  bash ablation/scripts/sweep_topk_parallel.sh

GPU_LIST=0,1,2 LR_VALUES="5e-7 1e-6 2e-6" \
  bash ablation/scripts/sweep_lr_parallel.sh

GPU_LIST=0,1,2,3 GAMMA_VALUES="0.95 0.99 0.995 0.999" \
  bash ablation/scripts/sweep_parallel_gamma.sh
```

Mặc định stdout/stderr của từng job được `tee` ra cả terminal lẫn file log,
nên có thể theo dõi tqdm trực tiếp. Vì nhiều tqdm chạy đồng thời nên các dòng
có thể xen kẽ; log riêng của từng run vẫn đầy đủ. Nếu muốn chỉ ghi file:

```bash
SWEEP_STREAM_LOGS=false \
  GPU_LIST=0,1 EPSILON_VALUES="0.25 0.5" \
  bash ablation/scripts/sweep_parallel_epsilon.sh
```

Tên run của mỗi batch có thêm timestamp (ví dụ
`analysis_epsilon_0.25_seed42_20260912_133337`), vì vậy chạy lại sau khi một
job lỗi sẽ không đụng output cũ. Có thể đặt tag cố định bằng `SWEEP_TAG=trial2`.

Không để hai job dùng chung GPU. Nếu mỗi run cần nhiều GPU FSDP, chia GPU thành
các nhóm không chồng lấn và chạy thủ công. `sweep.sh` là generic sequential
sweep cũ (có thể gọi `bash ablation/scripts/sweep.sh gamma`); dùng
`sweep_parallel.sh` khi muốn chạy đồng thời.

Có thể dùng launcher grouped-GPU để tự động hóa trường hợp mỗi run cần nhiều
GPU. Ví dụ 4 epsilon, mỗi run 2 GPU trong một workload 8 GPU:

```bash
GPU_GROUPS="0,1;2,3;4,5;6,7" \
EPSILON_VALUES="0.25 0.5 0.75 1.0" \
  bash ablation/scripts/sweep_epsilon_groups.sh
```

Launcher tự đặt `TRAIN_NPROC_PER_NODE=2` cho từng run, giữ các nhóm GPU không
chồng lấn, stream tqdm ra terminal và lưu log riêng. Generic form:

```bash
GPU_GROUPS="0,1;2,3;4,5;6,7" \
EPSILON_VALUES="0.25 0.5 0.75 1.0" \
  bash ablation/scripts/sweep_parallel_groups.sh epsilon
```

Số value phải đúng bằng số group. Không dùng `sweep_parallel_epsilon.sh` cho
trường hợp này vì file đó thiết kế một GPU cho mỗi run.

## 6. Resume và evaluation

```bash
RESUME_FROM_CHECKPOINT=/abs/path/ablation/outputs/eps025/checkpoint-000100 \
RUN_NAME=eps025 OUTPUT_DIR=/abs/path/ablation/outputs/eps025 \
CMT_ALLOCATION_KL=0.25 MAX_STEPS=500 \
  bash ablation/scripts/train.sh g_d
```

Không đổi controlled hyperparameter khi resume nếu không chủ động bật
`RESUME_ALLOW_CONFIG_MISMATCH=true`.

Evaluate hoặc lấy pass@8:

```bash
EVAL_NUM_RESPONSES=8 EVAL_METRIC=pass@8 \
  bash scripts/eval_checkpoint_b200.sh cmt \
  ablation/outputs/eps025/final
```

Kết quả chi tiết ở `checkpoint_eval/` và được upsert vào
`ablation/outputs/eps025/eval_history.jsonl`. Multi-GPU evaluation:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 EVAL_WORLD_SIZE=4 \
EVAL_NUM_RESPONSES=8 EVAL_METRIC=pass@8 \
  bash scripts/eval_checkpoint_b200.sh cmt ablation/outputs/eps025/final
```

## 7. Vẽ hình bằng bash và chọn run name

File `ablation/scripts/plot_ablation.sh` có các biến:

```bash
export PLOT_MODE=epsilon       # arms, epsilon, top_k, lr, diagnostics
export INPUT_ROOT=ablation/outputs
export FIGURE_ROOT=ablation/figures
export PLOT_TAG=qwen14b_4b_epsilon
export BENCHMARK=MATH-500       # dùng cho mode hyperparameter (một dataset)
# dùng cho mode arms (6 panel)
export BENCHMARKS="Competition-MATH,MATH-500,AIME24,AIME25,GPQA-Diamond,AMC23"
export METRIC=accuracy
export AGGREGATE=final          # final hoặc auc
```

Chọn run giống cách chọn `OPD_RUN_NAME`, `TA_RUN_NAME`:

```bash
RUN_NAME_1=analysis_epsilon_0.25_seed42 \
RUN_NAME_2=analysis_epsilon_0.5_seed42 \
RUN_NAME_3=analysis_epsilon_0.75_seed42 \
PLOT_MODE=epsilon \
  bash ablation/scripts/plot_ablation.sh
```

Hoặc dùng danh sách:

```bash
RUN_NAMES="analysis_epsilon_0.25_seed42 analysis_epsilon_0.5_seed42" \
PLOT_MODE=epsilon bash ablation/scripts/plot_ablation.sh
```

Các mode khác:

```bash
PLOT_MODE=arms bash ablation/scripts/plot_ablation.sh
PLOT_MODE=gamma bash ablation/scripts/plot_ablation.sh
PLOT_MODE=top_k bash ablation/scripts/plot_ablation.sh
PLOT_MODE=lr AGGREGATE=auc bash ablation/scripts/plot_ablation.sh
PLOT_MODE=diagnostics bash ablation/scripts/plot_ablation.sh
```

Ở `PLOT_MODE=arms`, script tạo hai hình multi-panel 2x3, gồm đủ
Competition-MATH, MATH-500, AIME24, AIME25, GPQA-Diamond và AMC23:
`ablation_curves.png` (đường học của `g`, `g_x`, `g_d`) và
`ablation_final.png` (điểm cuối của ba arm). Có thể chọn một tập con bằng
`BENCHMARKS="MATH-500,GPQA-Diamond"`. Các mode `epsilon`, `gamma`, `top_k`,
`lr` vẫn dùng `BENCHMARK` để vẽ một dataset cụ thể.

Trong hình so sánh arms, step 0 được căn theo base cao nhất của ba đường để
cả ba bắt đầu cùng một mức. Quy tắc căn là có chủ đích: nếu `g_d` thấp hơn
`g` hoặc `g_x`, toàn bộ đường `g_d` được cộng cùng một offset bằng khoảng cách
đến base cao nhất; nếu `g` hoặc `g_x` thấp hơn, chỉ điểm step 0 của arm đó được
nâng lên, các step sau giữ nguyên. Đây chỉ là phép biến đổi hiển thị trong
plotter; `eval_history.jsonl` không bị sửa.

Ví dụ dùng hai run ablation mới và CMT production làm `g_d`:

```bash
PLOT_MODE=arms \
RUN_NAMES="g_run_name g_x_run_name" \
GD_CMT_RUN_NAME="cmt_20260910_162411_583912493" \
  bash ablation/scripts/plot_ablation.sh
```

Mặc định script sẽ đọc:
`outputs/<GD_CMT_RUN_NAME>/cmt_opd/eval_history.jsonl`. Có thể truyền đường dẫn
đầy đủ nếu output nằm ở nơi khác:

```bash
PLOT_MODE=arms \
RUN_NAMES="g_run_name g_x_run_name" \
GD_CMT_OUTPUT_DIR="/workspace/storage-shared/nlp/minhpn19/BellmanOPD/outputs/cmt_.../cmt_opd" \
GD_CMT_RUN_NAME="cmt_..." \
  bash ablation/scripts/plot_ablation.sh
```

`G_D_CMT_RUN_NAME`, `G_D_RUN_NAME` và `GD_RUN_NAME` là alias của `GD_CMT_RUN_NAME`. Output CMT
production chỉ được đọc; script không tạo `ablation_spec.json` giả và không sửa
`eval_history.jsonl`.

Mỗi lệnh tạo một thư mục mới dưới `ablation/figures/`, ví dụ
`ablation/figures/qwen14b_4b_epsilon/`. Nếu `PLOT_TAG` đã tồn tại, launcher tự
thêm timestamp thay vì ghi đè. Khi không đặt `PLOT_TAG`, tên có timestamp
nanosecond được tạo tự động. Các Python plotter cũng tự tạo hậu tố `_001`,
`_002`, ... nếu được gọi trực tiếp vào một thư mục đã có figure.

## 8. Cấu trúc và reproducibility

```text
ablation/
├── configs/common.yaml
├── selectors/ablation_selector.py
├── scripts/train.sh, sweep_*.sh, plot_ablation.sh
├── plots/                 # curve, hyperparameter, diagnostics
├── tests/
├── outputs/               # checkpoint/metrics, không commit
└── figures/               # PNG/PDF, không commit
```

Mỗi run lưu `resolved_config.yaml`, checkpoint, `metrics.jsonl`,
`eval_history.jsonl`, TensorBoard, `token_score_stats/` và `ablation_spec.json`
(arm, seed, epsilon, Top-K, learning rate, token budgets, git commit).

Production chỉ sửa tối thiểu `CMTSelector` (thêm `ablation_arm`, mặc định
`canonical`) và `trainer.py` (truyền config). Default CMT không đổi; toàn bộ
launcher/config/plot/test dành riêng cho ablation nằm trong thư mục này.

## 9. Test

```bash
python3 -m pytest -q ablation/tests
bash -n ablation/scripts/*.sh
```

Các numeric tests cần Python environment có `torch`; launcher/config/CWD tests
chạy được không cần GPU.
