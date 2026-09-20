# Hướng dẫn chạy BellmanOPD trên B200

File này là runbook thực hành cho các method trong repository hiện tại: OPD thuần,
TA-OPD, CMT-OPD, GRPO và IW-OPD (các script Bellman-RAC/PGT legacy vẫn được giữ tương thích).
Mọi lệnh đều chạy từ thư mục:

```bash
cd /mnt/hdd/nhatminh/OPD/BellmanOPD
```

`RUN_B200.md` chứa phần giải thích hạ tầng và tuning chi tiết hơn; file này tập trung vào các
lệnh thường dùng có thể copy-paste.

## 1. Chuẩn bị môi trường

Nếu cluster đã có PyTorch/vLLM environment chuẩn cho B200, dùng environment đó. Nếu chưa:

```bash
cd /mnt/hdd/nhatminh/OPD/BellmanOPD
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python scripts/check_b200_env.py
```

Đặt model, dataset và tokenizer protocol. Các path tương đối được resolve dưới `STORAGE_ROOT`.
Mặc định là Qwen3-8B teacher, Qwen3-1.7B-Base student và Competition-MATH:

```bash
export STORAGE_ROOT=/workspace/storage-shared
export TEACHER_MODEL=models/Qwen3-8B
export STUDENT_MODEL=nlp/tungdd11/stable-on-policy-distillation/OPD/model/Qwen3-1.7B-Base
export TRAIN_DATA=nlp/minhpn19/data/competition_math/data/train-00000-of-00001.parquet
export PROMPT_KEY=problem
```

### Đổi model family chỉ bằng hai path

Code tự nhận diện `Qwen3DecoderLayer`, `Qwen3_5DecoderLayer` hoặc
`LlamaDecoderLayer`; không cần sửa YAML/FSDP. Teacher và student bắt buộc có
cùng ánh xạ token-ID (preflight sẽ dừng sớm nếu không đúng).

Qwen3.5-9B teacher, Qwen3.5-4B student:

```bash
export TEACHER_MODEL=/workspace/storage-shared/models/Qwen3.5-9B
export STUDENT_MODEL=/workspace/storage-shared/models/Qwen3.5-4B
```

Llama-3.1-8B-Instruct teacher, Llama-3.2-3B-Instruct student:

```bash
export TEACHER_MODEL=/workspace/storage-shared/models/Llama-3.1-8B-Instruct
export STUDENT_MODEL=/workspace/storage-shared/models/Llama-3.2-3B-Instruct
```

Muốn quay lại Qwen3 thì chỉ đặt lại đúng hai biến như block mặc định phía trên.
Qwen3.5 cần `transformers>=4.57,<5`; chạy lại `pip install -r requirements.txt`
nếu environment cũ chưa nhận diện `qwen3_5`.

Preset DAPO-Math:

```bash
unset TRAIN_DATA TRAIN_DATA_PATH PROMPT_KEY TRAIN_PROMPT_KEY TRAIN_DATA_SPLIT
export TRAIN_DATASET=dapo_math
```

Dataset custom:

```bash
unset TRAIN_DATA TRAIN_DATA_PATH PROMPT_KEY TRAIN_PROMPT_KEY TRAIN_DATA_SPLIT
export TRAIN_DATASET=custom
export TRAIN_DATA_PATH=/absolute/path/to/train.parquet
export TRAIN_DATA_SPLIT=null
export TRAIN_PROMPT_KEY=problem
```

Kiểm tra preflight asset/GPU trước khi chạy full:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/smoke_test_b200.sh
```

Smoke FSDP thật (tùy chọn, nhưng nên chạy trước full run):

```bash
CUDA_VISIBLE_DEVICES=0,1 METHOD=opd bash scripts/smoke_test_fsdp_multigpu.sh
CUDA_VISIBLE_DEVICES=0,1 METHOD=cmt bash scripts/smoke_test_fsdp_multigpu.sh
```

Preflight report và autotune được namespace theo model/data fingerprint và số GPU. Không dùng
chung một `OUTPUT_DIR` cho hai experiment khác nhau.

## 2. Shared configuration cho một comparison công bằng

Tạo run name riêng nhưng giữ mọi shared hyperparameter giống nhau:

```bash
PAIR=$(date +%Y%m%d_%H%M%S)
export OPD_RUN_NAME="opd_${PAIR}"
export TA_RUN_NAME="ta_${PAIR}"
export RAC_RUN_NAME="rac_${PAIR}"
export PGT_RUN_NAME="pgt_${PAIR}"
export CMT_RUN_NAME="cmt_${PAIR}"
# GRPO has no teacher--student pair; keep its label independent of PAIR.
export GRPO_RUN_NAME="grpo_qwen3_1p7b_compmath_seed42_${PAIR}"
export IW_RUN_NAME="iw_qwen3_1p7b_8b_compmath_seed42_${PAIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export DISTRIBUTED_STRATEGY=fsdp
export BATCH_SIZE=64
export NUM_RESPONSES=4
export MICRO_BATCH_SIZE_PER_GPU=16
export PPO_MINI_BATCH_SIZE=64
export LR=5e-6
export MAX_PROMPT_LENGTH=1024
export MAX_RESPONSE_LENGTH=4096
export TOP_K=16
export SAVE_INTERVAL=150
export EVAL_INTERVAL=150
export ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION=0.6
export ROLLOUT_VLLM_MAX_MODEL_LEN=5200
export TRAIN_EVAL_ENABLED=true
export TRAIN_EVAL_NUM_RESPONSES=8
export TRAIN_EVAL_SEED=42
export TRAIN_EVAL_SYNC_TIMEOUT_SEC=86400
export ROLLOUT_TEMPERATURE=1.0
export ROLLOUT_TOP_P=1.0
```

`MAX_STEPS` là tổng số optimizer steps mục tiêu. Không đặt hoặc đặt `MAX_STEPS=-1` để chạy hết
epoch đã cấu hình; đặt `MAX_STEPS=1` hoặc `2` cho debug.

`BATCH_SIZE` là số prompt global. Cấu hình trên sinh `64 x 4 = 256` trajectory trong
một lần rollout, sau đó chia thành bốn PPO group global, mỗi group 64 trajectory.
Với CMT, group thứ `r` chứa response thứ `r` của cả 64 prompt và được giải một Gibbs
riêng ngay trước đúng một optimizer update. Vì vậy có đúng bốn Gibbs allocation và bốn
optimizer step cho mỗi full rollout; không còn allocation chung trên 256 trajectory.
Trên 4 GPU, mỗi rank nhận 16 trajectory thật trong mỗi PPO group.
`MICRO_BATCH_SIZE_PER_GPU=16` chỉ là chunk local; giảm giá trị này khi OOM không làm
tăng số Gibbs allocation hay optimizer step.

Launcher tự chọn `NUM_EPOCHS=3` cho Competition-MATH và `NUM_EPOCHS=2` cho DAPO.
Có thể override `NUM_EPOCHS` từ command line nếu thực nghiệm cần khác.

## 3. Training

### OPD thuần

```bash
RUN_NAME="$OPD_RUN_NAME" bash scripts/train_opd_b200.sh
```

OPD dùng weight uniform trên mọi response token hợp lệ.

### CMT-OPD

```bash
RUN_NAME="$CMT_RUN_NAME" bash scripts/train_cmt_b200.sh
```

CMT mặc định dùng `CMT_ALLOCATION_KL=0.5`, `CMT_GAMMA=1.0`,
`CMT_SUCCESSOR_LAMBDA=1.0`, union support cho CMT score, Student Top-16 cho OPD loss và `top_p=1`.
Gibbs được chuẩn hoá độc lập trong từng PPO group 64 trajectory (one Gibbs = one update).
Full-vocabulary CMT diagnostics
không bật mặc định:

```bash
CMT_FULL_VOCAB_DIAGNOSTICS=true RUN_NAME="$CMT_RUN_NAME" \
  bash scripts/train_cmt_b200.sh
```

### TA-OPD

```bash
RUN_NAME="$TA_RUN_NAME" bash scripts/train_ta_b200.sh
```

TA hard-select fraction mặc định `TA_RHO=0.10`:

### GRPO thuần (teacher-free)

GRPO không tải hoặc dùng teacher. Mỗi prompt được rollout nhiều lần (mặc định `G=8`),
reward outcome được chấm bằng `math_verify`/boxed-answer, chuẩn hoá theo group rồi tối ưu
clipped PPO surrogate. Script riêng đặt mặc định eval và checkpoint mỗi 100 optimizer steps;
`TRAIN_EVAL_NUM_RESPONSES` vẫn là số mẫu eval, độc lập với `GRPO_GROUP_SIZE`:

`GRPO_RUN_NAME` chỉ là tên thư mục/nhãn thí nghiệm, không biểu diễn cặp teacher--student.
Vì vậy dùng tên như `grpo_qwen3_1p7b_compmath_seed42`; tên cũ dạng `grpo_14b_4b` (nếu có)
chỉ là nhãn đặt nhầm, không làm GRPO tải teacher 14B.

`PPO_MINI_BATCH_SIZE` cũng là global, giống OPD/TA/CMT. Mặc định GRPO là `16`: 1 GPU nhận
16 trajectory thực mỗi optimizer step, 2 GPU nhận 8+8, 4 GPU nhận 4+4+4+4. `MICRO_BATCH_SIZE_PER_GPU`
chỉ điều khiển chia nhỏ local batch. Vì GRPO giữ nhiều response dài trong graph của
backward, launcher GRPO mặc định `MICRO_BATCH_SIZE_PER_GPU=1`; đây chỉ là gradient
accumulation và không đổi objective/effective PPO minibatch. Chỉ tăng lên `2`, `4`, ...
sau khi đã kiểm tra peak VRAM với đúng model và `MAX_RESPONSE_LEN`.

```bash
CUDA_VISIBLE_DEVICES=0,1 \
GRPO_GROUP_SIZE=8 PPO_MINI_BATCH_SIZE=16 \
GRPO_RUN_NAME="grpo_qwen3_1p7b_compmath_seed42" \
bash scripts/train_grpo_b200.sh
```

Đổi group size (phải lớn hơn hoặc bằng 2), batch, số GPU hoặc giới hạn debug:

```bash
CUDA_VISIBLE_DEVICES=0 \
GRPO_GROUP_SIZE=4 BATCH_SIZE=16 PPO_MINI_BATCH_SIZE=16 \
MAX_STEPS=10 RUN_NAME=grpo_debug bash scripts/train_grpo_b200.sh
```

GRPO dùng `rollout.temperature=1.0` để log-prob hành vi từ vLLM là đúng mẫu số PPO;
không đặt `ROLLOUT_TEMPERATURE` khác 1.0.

#### Giữ cùng số rollout/step với các baseline

Nếu `BATCH_SIZE=64`, `GRPO_GROUP_SIZE=4` thì mỗi rollout tạo
`64 * 4 = 256` trajectory. Số optimizer step trong một rollout là:

```text
ceil(256 / PPO_MINI_BATCH_SIZE)
```

Đây là lý do cấu hình tương đương với CMT phải là:

```text
CMT : ceil(64 * 4 / 64) = 4 step/rollout
GRPO: ceil(64 * 4 / 64) = 4 step/rollout
```

Nếu GRPO dùng `PPO_MINI_BATCH_SIZE=16` mặc định thì sẽ có
`ceil(64 * 4 / 16)=16` step/rollout, tức khoảng 4 lần nhiều step (gần 3,000
thay vì 750). Vì vậy hãy dùng `PPO_MINI_BATCH_SIZE=64` để giữ cùng số step một
epoch với OPD/TA/CMT:

```bash
# Competition-MATH: target 750 optimizer steps
CUDA_VISIBLE_DEVICES=0,1 \
TRAIN_DATASET=competition_math \
BATCH_SIZE=64 GLOBAL_BATCH_SIZE=64 \
GRPO_GROUP_SIZE=4 PPO_MINI_BATCH_SIZE=64 \
MICRO_BATCH_SIZE_PER_GPU=1 MAX_STEPS=750 \
GRPO_RUN_NAME=grpo_compmath_g4_rolloutstep_seed42 \
bash scripts/train_grpo_b200.sh

# DAPO-Math: target 1087 optimizer steps
CUDA_VISIBLE_DEVICES=0,1 \
TRAIN_DATASET=dapo_math \
BATCH_SIZE=64 GLOBAL_BATCH_SIZE=64 \
GRPO_GROUP_SIZE=4 PPO_MINI_BATCH_SIZE=64 \
MICRO_BATCH_SIZE_PER_GPU=1 MAX_STEPS=1087 \
GRPO_ANSWER_KEY=solution \
GRPO_RUN_NAME=grpo_dapo_g4_rolloutstep_seed42 \
bash scripts/train_grpo_b200.sh
```

Với cấu hình production hiện tại, GRPO `G=4, PPO=64` và CMT `n=4, PPO=64`
đều dùng 64 trajectory cho mỗi optimizer update và đều có bốn update/full rollout.
CMT khác ở chỗ bốn group được tạo theo response index và mỗi group có Gibbs
allocation độc lập; GRPO dùng group-relative reward objective riêng của nó.

#### Chẩn đoán OOM GRPO

OOM tại `loss.backward()` với thông báo PyTorch đã cấp phát khoảng 165--171 GiB
không phải do vLLM eval/rollout giữ toàn bộ GPU: rollout server được khởi tạo với
`--enable-sleep-mode` và ngủ trước backward; trong log lỗi, tiến trình vLLM phụ chỉ
dùng khoảng 1.5 GiB. Nguyên nhân là activation của các response dài trong một
micro-batch lớn. Giữ nguyên global batch, PPO minibatch và `MAX_RESPONSE_LEN`, rồi
giảm micro-batch:

```bash
MICRO_BATCH_SIZE_PER_GPU=1 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
... bash scripts/train_grpo_b200.sh
```

`expandable_segments` chỉ hỗ trợ phân mảnh allocator; nó không thay thế việc giảm
micro-batch. Nếu vẫn OOM với micro-batch 1, giảm `MAX_RESPONSE_LEN`/`MAX_NEW_TOKENS`
theo cùng protocol cho cả các baseline, hoặc dùng student nhỏ hơn; không giảm
`PPO_MINI_BATCH_SIZE` để chữa lỗi vì điều đó làm thay đổi số optimizer step.

Với `TRAIN_DATASET=dapo_math`, DAPO-Math-17k dùng trường đáp án `solution`
(và bản sao `reward_model.ground_truth`), không dùng `answer`. Launcher GRPO tự
chọn `GRPO_ANSWER_KEY=solution`; `GRPO_REWARD_BENCHMARK` vẫn nên để
`Competition-MATH` vì biến này chọn math grader, không phải tên dataset:

```bash
TRAIN_DATASET=dapo_math \
GRPO_REWARD_BENCHMARK=Competition-MATH \
GRPO_RUN_NAME="grpo_dapo_math_seed42" \
  bash scripts/train_grpo_b200.sh
```

Có thể override thủ công bằng `GRPO_ANSWER_KEY=solution`. Code cũng fallback
an toàn qua các field `answer`, `solution`, `ground_truth` và
`reward_model.ground_truth` nếu dataset custom không dùng schema mặc định.

Trong workflow tuần tự, GRPO mặc định tắt để không vô tình phát sinh thêm GPU-hours;
bật rõ ràng như sau:

```bash
RUN_GRPO_TRAIN=true GRPO_GROUP_SIZE=8 bash scripts/train_all_b200.sh
```

Resume GRPO trên topology khác (ví dụ chạy đầu bằng 1 GPU, tiếp tục bằng 4 GPU):

```bash
# Lần đầu
CUDA_VISIBLE_DEVICES=0 \
GRPO_RUN_NAME=grpo_qwen3_1p7b_compmath_seed42 \
GRPO_GROUP_SIZE=8 PPO_MINI_BATCH_SIZE=16 \
bash scripts/train_grpo_b200.sh

# Tiếp tục cùng run; MAX_STEPS là tổng target cuối cùng
CUDA_VISIBLE_DEVICES=0,1,2,3 \
GRPO_RUN_NAME=grpo_qwen3_1p7b_compmath_seed42 RESUME=auto MAX_STEPS=750 \
bash scripts/train_grpo_b200.sh
```

Checkpoint GRPO dùng cùng FSDP full-state/standard optimizer checkpoint và bộ chuyển đổi hai
chiều của các baseline, nên 1↔N↔M GPU được hỗ trợ. Để resume đúng cùng objective, giữ nguyên
student, dataset/order, seed, `GRPO_GROUP_SIZE`, global `BATCH_SIZE`, `PPO_MINI_BATCH_SIZE`,
learning rate và các PPO setting; chỉ thay `CUDA_VISIBLE_DEVICES` và microbatch theo VRAM.

```bash
TA_RHO=0.10 RUN_NAME="$TA_RUN_NAME" bash scripts/train_ta_b200.sh
```

### IW-OPD (Importance-Weighted On-Policy Distillation)

IW-OPD là baseline teacher-based độc lập, không phải GRPO. Bản cài đặt dùng đúng
thuật toán chính chủ: với mỗi sampled response token, advantage OPD là
`log p_teacher - log p_student`; `d_t=abs(advantage)` (mặc định), rồi nhân advantage
bằng prefix remaining-discrepancy weight với `weight_max=1.5`. Weight được stop-gradient,
không thêm critic/counterfactual rollout và PPO clipping vẫn được áp dụng sau khi nhân weight.
IW dùng singleton sampled-action support trong loss để khớp objective chính chủ; các method cũ
không đi qua nhánh này.

Launcher IW mặc định công bằng theo yêu cầu: `BATCH_SIZE=64`, `PPO_MINI_BATCH_SIZE=16`,
`LR=5e-6`, `MAX_RESPONSE_LEN=4096`, `MICRO_BATCH_SIZE_PER_GPU=8`,
`ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION=0.6`, eval/checkpoint mỗi 100 step, FSDP và vLLM đa GPU.
IW hỗ trợ cùng preset dữ liệu với các method khác: `competition_math` (mặc định) và
`dapo_math`/`dapo` (DAPO-Math-17k-Processed). Có thể dùng `TRAIN_DATASET=custom` cùng
`TRAIN_DATA_PATH`, `TRAIN_PROMPT_KEY` và tùy chọn `TRAIN_DATA_SPLIT` cho dữ liệu riêng.

```bash
CUDA_VISIBLE_DEVICES=0,1 \
IW_RUN_NAME="${IW_RUN_NAME}" \
BATCH_SIZE=64 PPO_MINI_BATCH_SIZE=16 MICRO_BATCH_SIZE_PER_GPU=8 \
LR=5e-6 MAX_RESPONSE_LEN=4096 ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION=0.6 \
bash scripts/train_iw_b200.sh
```

Chạy IW trên DAPO-Math:

```bash
TRAIN_DATASET=dapo_math CUDA_VISIBLE_DEVICES=0,1 \
IW_RUN_NAME="iw_qwen3_1p7b_8b_dapo_seed42_$(date +%Y%m%d_%H%M%S)" \
BATCH_SIZE=64 PPO_MINI_BATCH_SIZE=16 MICRO_BATCH_SIZE_PER_GPU=8 \
LR=5e-6 MAX_RESPONSE_LEN=4096 ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION=0.6 \
bash scripts/train_iw_b200.sh
```

Có thể đổi cặp model/data bằng các biến shared ở mục 1. `IW_OPD_WEIGHT_MAX=1.5` và
`IW_OPD_WEIGHT_USE_ABS=true` là giá trị chính chủ; chỉ đổi khi làm ablation.

Resume cùng hoặc khác số GPU:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
IW_RUN_NAME="${IW_RUN_NAME}" RESUME=auto MAX_STEPS=750 \
bash scripts/train_iw_b200.sh
```

Đánh giá checkpoint hoặc re-eval toàn bộ checkpoint:

```bash
IW_RUN_NAME="${IW_RUN_NAME}" bash scripts/eval_iw_b200.sh
REEVAL_NUM_RESPONSES=8 REEVAL_METRIC=avg@8 \
  bash scripts/reeval_method_checkpoints_b200.sh iw "${IW_RUN_NAME}"
```

Vẽ IW cùng các baseline:

```bash
PLOT_METHODS="opd ta cmt grpo iw" \
OPD_RUN_NAME="${OPD_RUN_NAME}" TA_RUN_NAME="${TA_RUN_NAME}" \
CMT_RUN_NAME="${CMT_RUN_NAME}" GRPO_RUN_NAME="${GRPO_RUN_NAME}" \
IW_RUN_NAME="${IW_RUN_NAME}" \
bash scripts/plot_training_progress.sh --plot-name opd_ta_cmt_grpo_iw
```

### Bellman-RAC và PGT (nếu cần baseline đầy đủ)

```bash
RUN_NAME="$RAC_RUN_NAME" bash scripts/train_rac_b200.sh
RUN_NAME="$PGT_RUN_NAME" bash scripts/train_pgt_b200.sh
```

Các launcher dùng chung rollout, checkpoint, TensorBoard và evaluation pipeline; teacher scoring
chỉ áp dụng cho OPD/TA/CMT, không áp dụng cho GRPO.

Trong CMT ablation, `g_d` chính là score canonical `learning_value` của CMT. Nếu
đã có CMT production run cùng cấu hình, không cần train lại `g_d`; chỉ train
`g` và `g_x`, sau đó truyền `GD_CMT_RUN_NAME` khi vẽ trong
`ablation/scripts/plot_ablation.sh`.

Training-time evaluation tự dùng toàn bộ GPU training khi `world_size>1`: mỗi rank chạy một
vLLM replica độc lập với `tensor_parallel_size=1`, nhận shard deterministic của benchmark, rồi
rank 0 merge lại thành đúng `summary.json`, prediction files và `model_outputs_detailed.jsonl.gz`
trong `training_eval/step-*`. Trong lúc generation/grade kéo dài, các rank chờ bằng filesystem
sentinel chứ không giữ NCCL barrier; chỉ có collective ngắn sau khi mọi shard đã hoàn tất.
Nếu một rank lỗi, sentinel lỗi được phát hiện và toàn job dừng với thông báo rõ. `sync_timeout_sec`
(mặc định 24 giờ) điều khiển timeout này. Với một GPU, pipeline cũ vẫn được giữ nguyên.
Multi-GPU training-time evaluation yêu cầu `training_evaluation.backend=vllm`; backend `hf` vẫn
dùng được cho single-GPU.
Có thể ghi pass@8 ngay trong periodic evaluation bằng
`TRAIN_EVAL_NUM_RESPONSES=8 TRAIN_EVAL_METRIC=pass@8`.
Các launcher `ablation/scripts/train.sh g`, `g_x` và (nếu thực sự chạy độc lập)
`g_d` mặc định đánh giá đủ 6 benchmark: Competition-MATH, MATH-500, AIME24,
AIME25, GPQA-Diamond và AMC23.
Có thể chủ động chạy một subset bằng `TRAIN_EVAL_BENCHMARKS="MATH-500,GPQA-Diamond"`.
Seed sampling của vLLM trong periodic evaluation được điều khiển riêng bằng
`TRAIN_EVAL_SEED` (ví dụ `TRAIN_EVAL_SEED=42`); biến này không thay đổi
`SEED` của optimizer/data hoặc `ROLLOUT_SEED` của student rollout. Nếu không đặt,
training-time evaluation giữ mặc định `1234` để tương thích các run cũ.
Các rank phải cùng nhìn thấy `experiment.output_dir` (filesystem dùng chung) để đọc sentinel và
merge shard.
Output mặc định:

```text
outputs/<run-name>/opd/
outputs/<run-name>/ta_opd/
outputs/<run-name>/rac_opd/
outputs/<run-name>/pgt_opd/
outputs/<run-name>/cmt_opd/
outputs/<run-name>/iw/
```

### Chạy ngắn để kiểm tra launch/config

```bash
MAX_STEPS=1 TRAIN_EVAL_ENABLED=false RUN_NAME="debug_opd" \
  bash scripts/train_opd_b200.sh

MAX_STEPS=1 TRAIN_EVAL_ENABLED=false RUN_NAME="debug_cmt" \
  bash scripts/train_cmt_b200.sh
```

Nếu OOM, giảm cùng một cách cho các method cần so sánh:

```bash
BATCH_SIZE=64 MICRO_BATCH_SIZE_PER_GPU=4 SCORE_MICRO_BATCH_SIZE=4 \
  RUN_NAME="$CMT_RUN_NAME" bash scripts/train_cmt_b200.sh
```

## 4. Resume

### Tự tìm checkpoint mới nhất

Giữ nguyên `RUN_NAME`; `RESUME=auto` sẽ tìm checkpoint hoàn chỉnh mới nhất trong output tương ứng:

```bash
RUN_NAME="$OPD_RUN_NAME" RESUME=auto \
  bash scripts/train_opd_b200.sh

RUN_NAME="$TA_RUN_NAME" RESUME=auto \
  bash scripts/train_ta_b200.sh

RUN_NAME="822192681" RESUME=auto \
  bash scripts/train_cmt_b200_14b.sh

RUN_NAME="601241037" RESUME=auto \
  bash scripts/train_cmt_b200_14b_dapo.sh

RUN_NAME="734372756" RESUME=auto \
  bash scripts/train_cmt_b200_4b_dapo.sh

RUN_NAME="$GRPO_RUN_NAME" RESUME=auto \
  bash scripts/train_grpo_b200.sh
```

`MAX_STEPS=200` ở đây là target cuối cùng, không phải chạy thêm 200 steps.

### Chỉ rõ checkpoint

```bash
RUN_NAME="$CMT_RUN_NAME" \
RESUME_FROM_CHECKPOINT="outputs/${CMT_RUN_NAME}/cmt_opd/checkpoint-000100" \
MAX_STEPS=200 \
  bash scripts/train_cmt_b200.sh
```

Nếu resume với config khác có chủ ý, cần bật rõ:

```bash
RESUME_ALLOW_CONFIG_MISMATCH=true \
  RUN_NAME="$CMT_RUN_NAME" RESUME=auto MAX_STEPS=200 \
  bash scripts/train_cmt_b200.sh
```

Không nên đổi model, tokenizer, dataset, seed, rollout protocol hoặc shared optimizer settings
giữa các lần resume nếu mục tiêu là tiếp tục cùng một experiment.

Số GPU khi resume có thể khác lúc tạo checkpoint. Checkpoint FSDP (`fsdp_full_v1`) được chuyển
về state integer-ID khi chạy single-GPU/non-FSDP; checkpoint standard được chuyển sang full
name-keyed state trước khi FSDP scatter. Mapping được kiểm tra theo tên, thứ tự và số lượng
parameter của optimizer; nếu model/param-group topology khác, chương trình sẽ báo lỗi thay vì
khôi phục nhầm state. Ví dụ chuyển từ 1 GPU sang 2 GPU:

```bash
CUDA_VISIBLE_DEVICES=0,1 RUN_NAME="$CMT_RUN_NAME" RESUME=auto MAX_STEPS=200 \
  bash scripts/train_cmt_b200.sh
```

## 5. TensorBoard và artifact chính

```bash
tensorboard --logdir_spec \
  "OPD:outputs/${OPD_RUN_NAME}/opd/tensorboard,TA:outputs/${TA_RUN_NAME}/ta_opd/tensorboard,CMT:outputs/${CMT_RUN_NAME}/cmt_opd/tensorboard,GRPO:outputs/${GRPO_RUN_NAME}/grpo/tensorboard,IW:outputs/${IW_RUN_NAME}/iw/tensorboard,RAC:outputs/${RAC_RUN_NAME}/rac_opd/tensorboard" \
  --bind_all --port 6006
```

Các file cần kiểm tra:

```text
outputs/<run>/<method>/resolved_config.yaml
outputs/<run>/<method>/metrics.jsonl
outputs/<run>/<method>/train_metrics.csv
outputs/<run>/<method>/tensorboard/
outputs/<run>/<method>/checkpoint-*/
outputs/<run>/<method>/final/
```

CMT có thêm selector diagnostics như support coverage, raw/conditional common mass, bounded
transition weight, `R`, `M`, `H`, successor excess và sequential gain.

## 6. Evaluation checkpoint cuối

Đánh giá riêng từng method; mặc định dùng vLLM, `temperature=0.7`, `top_p=0.95`, `n=8`:

```bash
OPD_RUN_NAME="$OPD_RUN_NAME" bash scripts/eval_opd_b200.sh
TA_RUN_NAME="$TA_RUN_NAME" bash scripts/eval_ta_b200.sh
CMT_RUN_NAME="$CMT_RUN_NAME" bash scripts/eval_cmt_b200.sh
IW_RUN_NAME="$IW_RUN_NAME" bash scripts/eval_iw_b200.sh
```

Đánh giá accuracy một response:

```bash
EVAL_NUM_RESPONSES=1 EVAL_TEMPERATURE=1 \
  CMT_RUN_NAME="$CMT_RUN_NAME" bash scripts/eval_cmt_b200.sh
```

Đổi checkpoint/output trực tiếp:

```bash
CMT_CHECKPOINT="outputs/${CMT_RUN_NAME}/cmt_opd/checkpoint-000100" \
CMT_EVAL_OUTPUT="results/manual_eval/cmt_step100" \
  bash scripts/eval_cmt_b200.sh
```

Kết quả gồm `summary.json`, prediction files và detailed JSONL dưới `results/<comparison>/eval/`
(hoặc thư mục được chỉ định bởi `*_EVAL_OUTPUT`).

### Eval và aggregate tất cả method

`eval_all_b200.sh` luôn chạy Base, OPD, TA và RAC; bật thêm PGT/CMT khi cần:

```bash
OPD_RUN_NAME="$OPD_RUN_NAME" \
TA_RUN_NAME="$TA_RUN_NAME" \
RAC_RUN_NAME="$RAC_RUN_NAME" \
CMT_RUN_NAME="$CMT_RUN_NAME" \
RUN_CMT_EVAL=true \
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_all_b200.sh
```

Thêm PGT:

```bash
RUN_PGT_EVAL=true RUN_CMT_EVAL=true \
OPD_RUN_NAME="$OPD_RUN_NAME" TA_RUN_NAME="$TA_RUN_NAME" \
RAC_RUN_NAME="$RAC_RUN_NAME" PGT_RUN_NAME="$PGT_RUN_NAME" \
CMT_RUN_NAME="$CMT_RUN_NAME" bash scripts/eval_all_b200.sh
```

Giới hạn evaluation để debug:

```bash
EVAL_LIMIT=20 EVAL_NUM_RESPONSES=1 EVAL_TEMPERATURE=1 \
  CMT_RUN_NAME="$CMT_RUN_NAME" bash scripts/eval_cmt_b200.sh
```

### Eval một checkpoint bất kỳ

Mặc định mỗi lần train/eval mới dùng đủ **6 benchmark** theo thứ tự:
`Competition-MATH`, `MATH-500`, `AIME24`, `AIME25`, `GPQA-Diamond`, `AMC23`.
GPQA-Diamond đọc từ `nlp/minhpn19/data/GPQA-Diamond/gpqa_diamond.jsonl`. Bản dữ liệu
hiện tại có các cột `id`, `prompt`, `ground_truth`; bốn lựa chọn A./B./C./D. đã nằm
trong `prompt` và `ground_truth` là chữ cái đáp án đúng. Loader cũng tương thích với
export cũ có các cột `Question`, `Correct Answer`, `Incorrect Answer 1/2/3`.
AMC23 đọc từ `nlp/minhpn19/data/amc23/test-00000-of-00001.parquet`, dùng cột `question`,
`answer` và `id`, rồi được render/chấm bằng cùng math prompt/verifier như các benchmark
toán còn lại. GPQA chỉ xáo trộn đáp án khi lựa chọn nằm ở các cột riêng; với prompt đã
nhúng lựa chọn, thứ tự A--D được giữ nguyên để không làm sai `ground_truth`. Các lần
train mới vì vậy sẽ
tự ghi thêm hai cột/biểu đồ này mà không thay đổi protocol của bốn bộ cũ.
Muốn debug nhanh chỉ hai bộ mới trong lúc train (không khuyến nghị cho comparison
chính), thêm `TRAIN_EVAL_BENCHMARKS="GPQA-Diamond,AMC23"`; mặc định biến này bỏ
trống để luôn chạy đủ sáu bộ.

```bash
EVAL_NUM_RESPONSES=1 EVAL_TEMPERATURE=1 \
  bash scripts/eval_checkpoint_b200.sh cmt \
  "outputs/${CMT_RUN_NAME}/cmt_opd/checkpoint-000100"
```

Khi checkpoint nằm dưới `outputs/<run>/<method>/` và bỏ qua `OUTPUT_DIR`, script ghi artifact
chi tiết vào `<method-output>/checkpoint_eval/<checkpoint-name>/` và tự upsert kết quả vào
`<method-output>/eval_history.jsonl` (đồng thời cập nhật `eval_metrics.csv`). Thay `cmt` bằng
`opd`, `ta`, `rac` hoặc `pgt`. Có thể truyền `OUTPUT_DIR` thứ ba nếu muốn giữ artifact chi tiết
ở một thư mục khác; history của run vẫn được cập nhật nếu đường dẫn checkpoint có layout chuẩn.

Pass@8 cho một checkpoint bất kỳ:

```bash
EVAL_NUM_RESPONSES=8 EVAL_METRIC=pass@8 EVAL_TEMPERATURE=0.7 \
  CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/eval_checkpoint_b200.sh cmt \
  "outputs/${CMT_RUN_NAME}/cmt_opd/checkpoint-000100"
```

Chạy lại cùng lệnh cho cùng checkpoint sẽ thay đúng row `(step, method)` trong
`eval_history.jsonl`, không tạo bản ghi trùng.

Chỉ re-evaluate hai benchmark mới cho một checkpoint (bốn benchmark cũ vẫn giữ nguyên
trong `eval_history.jsonl` và `eval_metrics.csv`):

```bash
EVAL_BENCHMARKS="GPQA-Diamond,AMC23" \
EVAL_NUM_RESPONSES=8 EVAL_METRIC=avg@8 EVAL_TEMPERATURE=0.7 \
  bash scripts/eval_checkpoint_b200.sh cmt \
  "outputs/${CMT_RUN_NAME}/cmt_opd/checkpoint-000100"
```

Re-evaluate **toàn bộ checkpoint** của một run chỉ với hai bộ mới:

```bash
REEVAL_BENCHMARKS="Competition-MATH,MATH-500,AIME24,AIME25,GPQA-Diamond,AMC23" \
REEVAL_NUM_RESPONSES=8 REEVAL_METRIC=avg@8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 REEVAL_WORLD_SIZE=8 \
  bash scripts/reeval_method_checkpoints_b200.sh cmt "cmt_20260918_111418_822192681"
```

Các run tạo trước khi cập nhật (resolved config chỉ có 4 bộ) cũng chạy được: script tự
bổ sung spec của hai dataset trong bộ nhớ, không sửa `resolved_config.yaml` hay checkpoint.

Lệnh trên merge theo `(step, method, benchmark)`: kết quả cũ của
`Competition-MATH/MATH-500/AIME24/AIME25` không bị xóa. Chạy lại cùng protocol cho
cùng checkpoint/benchmark sẽ thay đúng kết quả benchmark đó. Nếu dùng `pass@8`, các
file metric-specific `eval_history_pass_at_8.jsonl` vẫn được tách như trước.
Nếu history cũ còn dòng IFEval, plotting chỉ bỏ qua dòng benchmark legacy đó; nó không
được gán nhãn lại thành AMC23.

Nếu mục tiêu là pass@8 thay vì bổ sung vào history avg@8, chạy biến thể sau; kết quả
được lưu ở bộ file `*_pass_at_8` riêng và không trộn metric với đường avg@8:

```bash
REEVAL_BENCHMARKS="GPQA-Diamond,AMC23" \
REEVAL_NUM_RESPONSES=8 REEVAL_METRIC=pass@8 \
  bash scripts/reeval_method_checkpoints_b200.sh cmt "$CMT_RUN_NAME"
```

Với `CUDA_VISIBLE_DEVICES` có nhiều GPU, evaluator tự chia benchmark deterministic thành các
shard, chạy một vLLM `TP=1` trên mỗi GPU rồi merge lại; với một GPU kết quả/protocol không đổi,
nhưng mỗi checkpoint được chạy trong subprocess riêng để engine cũ không giữ VRAM cho
checkpoint kế tiếp. Có thể chọn số worker bằng `EVAL_WORLD_SIZE`.

## 7. Re-evaluate toàn bộ checkpoint

Dry-run trước để chỉ kiểm tra danh sách checkpoint, không ghi file:

```bash
REEVAL_METHODS=cmt REEVAL_DRY_RUN=true \
  CMT_RUN_NAME="$CMT_RUN_NAME" \
  bash scripts/reeval_method_checkpoints_b200.sh cmt
```

Chạy thật toàn bộ checkpoint của CMT:

```bash
REEVAL_METHODS=cmt \
REEVAL_NUM_RESPONSES=8 REEVAL_TEMPERATURE=0.7 REEVAL_TOP_P=0.95 \
  CMT_RUN_NAME="$CMT_RUN_NAME" \
  bash scripts/reeval_method_checkpoints_b200.sh cmt
```

Shortcut re-eval pass@8 (hoạt động cho `opd`, `ta`, `rac`, `pgt`, `cmt`):

```bash
REEVAL_DRY_RUN=true CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/reeval_pass8_b200.sh cmt "$CMT_RUN_NAME"
REEVAL_WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/reeval_pass8_b200.sh cmt "$CMT_RUN_NAME"
```

Pass@8 được lưu riêng trong `eval_history_pass_at_8.jsonl`,
`eval_metrics_pass_at_8.csv`, `checkpoint_reevaluation_manifest_pass_at_8.json` và
`training_eval_pass_at_8/`; history metric khác không bị ghi đè. Chạy lại đúng pass@8 sẽ replace
bộ artifact pass@8 hiện có.

Re-evaluate nhiều method cùng protocol:

```bash
REEVAL_METHODS="opd ta cmt" \
OPD_RUN_NAME="$OPD_RUN_NAME" TA_RUN_NAME="$TA_RUN_NAME" CMT_RUN_NAME="$CMT_RUN_NAME" \
  bash scripts/reeval_all_checkpoints_b200.sh
```

Script này ghi lại `training_eval/step-*`, `eval_history.jsonl` và `eval_metrics.csv` của method
được chọn. Dùng `REEVAL_DRY_RUN=true` trước khi ghi đè.

## 8. Vẽ hình

### Accuracy/loss theo training step

Các run được chọn phải có `eval_history.jsonl`; khi vẽ nhiều method, cần thêm `metrics.jsonl`.

```bash
PLOT_METHODS="opd ta cmt" \
OPD_RUN_NAME="$OPD_RUN_NAME" TA_RUN_NAME="$TA_RUN_NAME" CMT_RUN_NAME="$CMT_RUN_NAME" \
  bash scripts/plot_training_progress.sh --plot-name opd_ta_cmt
```

So sánh thêm GRPO (mỗi benchmark một panel, tổng cộng 6 panel khi history có đủ dữ liệu):

```bash
PLOT_METHODS="opd ta cmt grpo" \
OPD_RUN_NAME="$OPD_RUN_NAME" TA_RUN_NAME="$TA_RUN_NAME" \
CMT_RUN_NAME="$CMT_RUN_NAME" GRPO_RUN_NAME="$GRPO_RUN_NAME" \
  bash scripts/plot_training_progress.sh --plot-name opd_ta_cmt_grpo
```

So sánh đầy đủ:

```bash
PLOT_METHODS="opd ta rac pgt cmt" \
OPD_RUN_NAME="$OPD_RUN_NAME" TA_RUN_NAME="$TA_RUN_NAME" \
RAC_RUN_NAME="$RAC_RUN_NAME" PGT_RUN_NAME="$PGT_RUN_NAME" CMT_RUN_NAME="$CMT_RUN_NAME" \
  bash scripts/plot_training_progress.sh --plot-name all_methods
```

Khi các history đã có đủ sáu benchmark, cùng lệnh này tự tạo **6 biểu đồ accuracy**
(mỗi benchmark một panel), gồm thêm `GPQA-Diamond` và `AMC23`. Để vẽ riêng các run
đã re-evaluate pass@8, trỏ script tới history pass@8 tương ứng hoặc copy/symlink file
đó thành `eval_history.jsonl` trong thư mục plot input; plotting không trộn các metric
khác nhau trong một hình.

Chỉ vẽ CMT:

```bash
PLOT_METHODS=cmt CMT_RUN_NAME="$CMT_RUN_NAME" \
  bash scripts/plot_training_progress.sh --plot-name cmt_progress
```

Ảnh và manifest được ghi dưới:

```text
results/<comparison-name>/plots/<plot-name>/
```

Smoothing mặc định là 10 step; đổi bằng:

```bash
SMOOTHING_WINDOW=1 PLOT_METHODS="opd cmt" \
  OPD_RUN_NAME="$OPD_RUN_NAME" CMT_RUN_NAME="$CMT_RUN_NAME" \
  bash scripts/plot_training_progress.sh --plot-name raw_opd_cmt
```

### Histogram CMT theo token và learning value

Trong lúc train CMT, logger compact ghi histogram/mean/quantile của các score trên
toàn bộ response-token hợp lệ tại các rollout endpoint được cấu hình (mặc định:
step 1, mỗi 50 step và step cuối). Không cần load checkpoint hay chạy GPU để vẽ
các biểu đồ này. Chạy:

```bash
cd /mnt/hdd/nhatminh/OPD/BellmanOPD
CMT_RUN_NAME="cmt_..." bash scripts/plot_cmt_scores.sh
```

Lệnh trên tạo một thư mục mới, không ghi đè các lần vẽ trước:

```text
outputs/<run-name>/cmt_opd/plots/cmt_scores_<run-name>_<timestamp>/
```

Trong đó có cả PNG và PDF:

- `*_token_score_histograms`: histogram của `g_t` (`gain`),
  `X_t` (`successor_excess`), `D_t` (`sequential_gain`), learning value
  (`learning_value`) và supervision weight (`w`) tại snapshot đầu/giữa/cuối.
- `*_token_score_histogram_heatmaps`: cùng năm histogram nhưng giữ **mọi** step
  đã log (trục dọc là training step, trục ngang là score).
- `*_learning_value_quantiles`: mean, median, q05--q95 và q25--q75 của learning
  value theo training step.
- `*_score_means`: mean của năm trường score theo training step.
- `*_plot_manifest.json`: run, source, các step và tên trường được vẽ.

Có thể chỉ định trực tiếp thư mục output hoặc tên file logic:

```bash
CMT_OUTPUT_DIR="/abs/path/to/outputs/cmt_xxx/cmt_opd" \
CMT_SCORE_RUN_NAME="cmt_14b_4b_eps025" \
  bash scripts/plot_cmt_scores.sh
```

Nếu muốn đặt tên thư mục plot cố định cho dễ tìm, dùng `CMT_SCORE_PLOT_NAME`.
Khi tên đó đã tồn tại, script tự tạo hậu tố `_02`, `_03`, ... để không mất ảnh cũ:

```bash
CMT_RUN_NAME="cmt_..." CMT_SCORE_PLOT_NAME="epsilon_025" \
  bash scripts/plot_cmt_scores.sh
```

Các file nguồn nằm trong `cmt_opd/token_score_stats/step-*.json`. Nếu thư mục này
không tồn tại (ví dụ run cũ tắt logger), cần train lại với:

```bash
bash scripts/train_cmt_b200.sh \
  --set logging.token_score_stats_enabled=true \
  --set logging.token_score_interval=50
```

Hoặc truyền trực tiếp override tương ứng trong config. Các biểu đồ dùng histogram
đã ghi sẵn, nên không thể khôi phục phân phối token-level của một run không lưu
`token_score_stats`; dữ liệu compact này cũng không chứa token text/ID.

Các biểu đồ accuracy tự động zoom trục Y theo miền giá trị quan sát (làm tròn theo
5 điểm phần trăm và chừa 2 điểm phần trăm đệm), thay vì luôn hiển thị 0--100%.
Step-0 của `Competition-MATH` và `MATH-500` được phép lệch tối đa 2 điểm phần trăm;
đường `Base student` trong biểu đồ nhiều method dùng giá trị đã căn chỉnh. Khi cùng
vẽ OPD và CMT-OPD, nếu OPD có Base cao hơn thì toàn bộ đường CMT được nâng theo độ lệch;
nếu CMT cao hơn thì chỉ điểm Step-0 của OPD được nâng, các step sau giữ nguyên.
Chênh lệch Step-0 của AIME không được dùng làm điều kiện từ chối biểu đồ.

### Vẽ final evaluation bar chart

Sau `eval_all_b200.sh`, dùng:

```bash
RUN_NAME="$OPD_RUN_NAME" \
OPD_RUN_NAME="$OPD_RUN_NAME" TA_RUN_NAME="$TA_RUN_NAME" \
RAC_RUN_NAME="$RAC_RUN_NAME" CMT_RUN_NAME="$CMT_RUN_NAME" \
RESULTS_DIR="results/${OPD_RUN_NAME}_vs_${TA_RUN_NAME}_vs_${RAC_RUN_NAME}" \
  bash scripts/plot_results.sh --plot-name final_comparison
```

`plot_results.sh` là plot final aggregate; `plot_training_progress.sh` là plot diễn biến theo
checkpoint. Nếu có CMT/GRPO/PGT trong aggregate, cần truyền đúng output directory và đã chạy eval cho
method đó.
Comparison chỉ gồm OPD/TA/CMT/GRPO cũng được; `plot_results.sh` tự bỏ qua RAC nếu
`RAC_RUN_OUTPUT/metrics.jsonl` không tồn tại, còn `plot_training_progress.sh` là lựa chọn
trực tiếp và rõ ràng nhất.

### Eval và re-eval GRPO

Eval checkpoint cuối (vẫn dùng đủ 6 benchmark mặc định):

```bash
GRPO_RUN_NAME="$GRPO_RUN_NAME" \
  bash scripts/eval_grpo_b200.sh
```

Re-eval toàn bộ checkpoint, ghi/ghi đè đúng các row cùng `(step, method)` trong history GRPO:

```bash
REEVAL_NUM_RESPONSES=8 REEVAL_METRIC=avg@8 \
GRPO_RUN_NAME="$GRPO_RUN_NAME" \
  bash scripts/reeval_method_checkpoints_b200.sh grpo
```

Chỉ chạy pass@8 trên subset benchmark mà không ảnh hưởng các dataset còn lại:

```bash
REEVAL_BENCHMARKS="GPQA-Diamond,AMC23" \
REEVAL_NUM_RESPONSES=8 REEVAL_METRIC=pass@8 \
GRPO_RUN_NAME="$GRPO_RUN_NAME" \
  bash scripts/reeval_method_checkpoints_b200.sh grpo
```

## 9. Một số override thường dùng

```bash
# Chạy trên một GPU để debug DDP/FSDP protocol
CUDA_VISIBLE_DEVICES=0 DISTRIBUTED_STRATEGY=ddp MAX_STEPS=1 \
  RUN_NAME=debug_one_gpu bash scripts/train_cmt_b200.sh

# Không chạy periodic evaluation trong lúc debug
TRAIN_EVAL_ENABLED=false MAX_STEPS=2 \
  RUN_NAME=debug_no_eval bash scripts/train_opd_b200.sh

# Chạy eval nhanh trên một subset
EVAL_LIMIT=20 EVAL_NUM_RESPONSES=1 EVAL_TEMPERATURE=1 \
  bash scripts/eval_checkpoint_b200.sh opd \
  "outputs/${OPD_RUN_NAME}/opd/checkpoint-000050"

# Bật sanity check log-prob HF/vLLM cho một rollout
VLLM_LOGPROB_SANITY_ENABLED=true VLLM_LOGPROB_SANITY_FAIL=true \
  MAX_STEPS=1 RUN_NAME=debug_logprob bash scripts/train_cmt_b200.sh
```

Trong comparison chính, không thay đổi riêng một method về batch, rollout length, seed,
temperature, dataset, optimizer hoặc evaluation protocol. CMT training nên giữ `ROLLOUT_TOP_P=1`
để bounded raw-kernel estimator đúng theo scored student distribution; evaluation có thể dùng
`top_p=0.95` như protocol benchmark riêng.

## 10. Troubleshooting nhanh

- **Missing preflight**: chạy lại `bash scripts/smoke_test_b200.sh` với đúng model/data và đúng số GPU.
- **OOM**: giảm `MICRO_BATCH_SIZE_PER_GPU`, sau đó `SCORE_MICRO_BATCH_SIZE`; giữ `BATCH_SIZE` và
  các knob này giống nhau giữa các baseline khi so sánh.
- **`Only ... GiB VRAM is free, below ... headroom` khi re-eval**: đây là thiếu VRAM trên
  GPU đang được chọn, không phải thiếu RAM hệ thống và cũng không phải lỗi pass@8/checkpoint.
  `gpu_memory_utilization=auto` cố ý dừng trước khi khởi tạo vLLM nếu không còn tối thiểu
  `EVAL_VLLM_GPU_HEADROOM_GIB` (mặc định 4 GiB) cộng thêm 2 GiB workspace reserve cho
  CUDA graph/attention transient allocations. Kiểm tra process đang giữ GPU:

  ```bash
  nvidia-smi
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
  ```

  Dừng các process cũ do chính bạn sở hữu hoặc chọn GPU còn trống. Các launcher eval/re-eval
  hiện tự phát hiện và dùng toàn bộ GPU khi `CUDA_VISIBLE_DEVICES` chưa được đặt; nếu muốn
  giới hạn một GPU hoặc một nhóm GPU thì đặt biến này rõ ràng. Script tự tạo một replica vLLM
  `TP=1` trên mỗi GPU. Những lệnh eval độc lập cùng trỏ vào một GPU sẽ được xếp hàng bằng
  filesystem lock, không khởi tạo hai engine chồng lên nhau:

  ```bash
  CUDA_VISIBLE_DEVICES=0,1 REEVAL_WORLD_SIZE=2 \
    bash scripts/reeval_pass8_b200.sh cmt "$CMT_RUN_NAME"
  ```

  Sau một lỗi vLLM, launcher cũng dọn cả process group của EngineCore/worker; vì vậy worker
  mồ côi không còn giữ VRAM cho các lệnh sau. Nếu `nvidia-smi` vẫn cho thấy process training
  hoặc vLLM khác đang chiếm GPU, không có cách an toàn để ép re-eval dùng chung VRAM đó: hãy
  chờ tiến trình kết thúc, chọn GPU khác, hoặc giảm workload của tiến trình đang chạy.

  Không bọc script re-eval này trong `torchrun`; nó tự quản lý các worker vLLM. Có thể truyền
  `EVAL_VLLM_GPU_MEMORY_UTILIZATION=0.90` (hoặc `REEVAL_VLLM_GPU_MEMORY_UTILIZATION=0.90`) chỉ
  sau khi đã xác nhận GPU đủ chỗ; tùy chọn số sẽ bỏ qua kiểm tra headroom và không làm mô hình
  vừa vào GPU chỉ còn 0.7 GiB.
  Nếu GPU vẫn chịu tải khác, tăng phần đệm bằng `VLLM_GPU_WORKSPACE_HEADROOM_GIB=4` khi train,
  `EVAL_VLLM_GPU_WORKSPACE_HEADROOM_GIB=4` khi eval một checkpoint, hoặc
  `REEVAL_VLLM_GPU_WORKSPACE_HEADROOM_GIB=4` khi re-eval.
- **Resume báo config mismatch**: kiểm tra `resolved_config.yaml`; chỉ dùng
  `RESUME_ALLOW_CONFIG_MISMATCH=true` khi thay đổi là có chủ ý.
- **`FileExistsError: Refusing to overwrite checkpoint path`**: checkpoint hoàn chỉnh là bất
  biến; dùng `RESUME=auto` hoặc `RESUME_FROM_CHECKPOINT` để tiếp tục từ checkpoint gần nhất,
  không chạy lại đúng cùng step. Nếu lần chạy trước bị ngắt khi đang ghi, rank 0 sẽ tự dọn đúng
  thư mục staging `.checkpoint-*.incomplete` rồi ghi lại an toàn. Các rank phụ không còn kiểm tra
  filesystem trước barrier nên không phát sinh race với thư mục staging của rank 0.
- **Plot thiếu method**: kiểm tra `PLOT_METHODS`, `*_RUN_NAME`, `eval_history.jsonl` và
  `metrics.jsonl` trong output tương ứng.
- **CMT cảnh báo top-p**: đây là cảnh báo đúng; `top_p<1` vẫn bounded nhưng estimator không còn
  unbiased cho raw truncated kernel. Không thêm importance correction thủ công.
