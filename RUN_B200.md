# Full B200 runbook

## Fresh environment

```bash
cd /workspace/storage-shared/nlp/minhpn19/TA-OPD-B200
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Không pin thủ công một CUDA wheel khác trước `requirements.txt`: image B200 cần PyTorch/vLLM phù hợp
driver thực tế. Nếu cluster cung cấp module/container PyTorch đã kiểm thử, dùng stack đó và cài phần
requirements còn lại theo chính sách cluster.

## Validate environment and schemas

```bash
export STORAGE_ROOT=/workspace/storage-shared
python scripts/check_b200_env.py
# Optional diagnostic only:
CUDA_VISIBLE_DEVICES=0 bash scripts/smoke_test_b200.sh
```

`smoke_test_b200.sh` chạy unit tests ở chế độ không chiếm GPU rồi preflight model/data/GPU; nó là
diagnostic tùy chọn, không còn là điều kiện để launcher train khởi động. Mỗi tổ hợp
teacher/student/data tự nhận một fingerprint và báo cáo riêng dưới
`results/preflight/asset-<fingerprint>-<N>gpu.json`, nên nhiều workload dùng chung project không
ghi đè preflight của nhau. Hậu tố số GPU bảo đảm mỗi topology được kiểm tra đúng các GPU visible;
file autotune cũng dùng cùng namespace vì batch layout phụ thuộc số rank. Đặt
`SMOKE_RUN_GPU_UNIT_TESTS=true` chỉ khi chủ động muốn chạy thêm unit test CUDA.

### Chọn model pair và train dataset

Ba launcher có block `USER CONFIG` riêng ở đầu file. Có thể override tại command line mà không sửa YAML:

```bash
export TEACHER_MODEL=models/Qwen3-8B
export STUDENT_MODEL=nlp/tungdd11/stable-on-policy-distillation/OPD/model/Qwen3-1.7B-Base
export TRAIN_DATA=nlp/minhpn19/data/competition_math/data/train-00000-of-00001.parquet
export PROMPT_KEY=problem
```

`competition_math` tương ứng file train Competition-MATH, `split=null`, `prompt_key=problem`;
`dapo_math` tương ứng `nlp/minhpn19/data/DAPO-Math-17k-Processed`, `split=all`,
`prompt_key=prompt`. Dataset tùy ý:

```bash
export TRAIN_DATASET=custom
export TRAIN_DATA_PATH=/absolute/or/storage-relative/train.parquet
export TRAIN_DATA_SPLIT=null
export TRAIN_PROMPT_KEY=problem
export TRAIN_PREFER_SOURCE_PROMPT=false
```

Path model/data nhận cả absolute path và path tương đối dưới `STORAGE_ROOT`. Selection mới tự dùng
preflight filename mới; không cần đặt `PREFLIGHT_REPORT` thủ công. Token-ID mapping giữa
teacher/student phải hoàn toàn giống nhau; chỉ
`special_tokens_map` được phép khác. Student tokenizer render prompt với `enable_thinking=false`,
và cùng tensor integer IDs được đưa thẳng vào teacher scoring, không có teacher re-tokenization.

Chạy smoke FSDP thật hai step trên mọi GPU đang visible trước full run (đổi `METHOD` để kiểm tra
từng method):

```bash
CUDA_VISIBLE_DEVICES=0,1 METHOD=opd bash scripts/smoke_test_fsdp_multigpu.sh
CUDA_VISIBLE_DEVICES=0,1 METHOD=ta  bash scripts/smoke_test_fsdp_multigpu.sh
CUDA_VISIBLE_DEVICES=0,1 METHOD=rac bash scripts/smoke_test_fsdp_multigpu.sh
CUDA_VISIBLE_DEVICES=0,1 METHOD=pgt bash scripts/smoke_test_fsdp_multigpu.sh

# Bốn rank FSDP:
CUDA_VISIBLE_DEVICES=0,1,2,3 METHOD=opd bash scripts/smoke_test_fsdp_multigpu.sh
```

Có thể chỉ inspect schema/path qua preflight CLI, nhưng cách này chủ động chọn output và không dùng
cơ chế filename tự động của shell launcher:

```bash
CUDA_VISIBLE_DEVICES=0 python -m b200_experiment.cli preflight \
  --config configs/qwen3_b200_base.yaml \
  --set paths.storage_root="$STORAGE_ROOT" \
  --output results/preflight.json
```

## Controlled full OPD, TA-OPD, Bellman-RAC, PGT và CMT runs

Tạo một comparison ID rồi giữ mọi shared knob giống hệt nhau:

```bash
PAIR=$(date +%Y%m%d_%H%M%S)
export OPD_RUN_NAME="opd_qwen3_8b_to_1p7b_base_${PAIR}"
export TA_RUN_NAME="ta_qwen3_8b_to_1p7b_base_${PAIR}"
export RAC_RUN_NAME="rac_bellman_qwen3_8b_to_1p7b_base_${PAIR}"
export CMT_RUN_NAME="cmt_qwen3_8b_to_1p7b_base_${PAIR}"

export CUDA_VISIBLE_DEVICES=0,1
export DISTRIBUTED_STRATEGY=fsdp
export BATCH_SIZE=64
export MICRO_BATCH_SIZE_PER_GPU=8
export NUM_RESPONSES=1
export LR=5e-6
export NUM_EPOCHS=1
export MAX_PROMPT_LENGTH=1024
export OVERLONG_PROMPT_POLICY=filter
export PPO_MINI_BATCH_SIZE=16
export MAX_RESPONSE_LENGTH=4096
export TOP_K=16
export TA_RHO=0.10
export RAC_GAMMA=0.995
export RAC_W_MIN=0.10
export RAC_BETA=2.0
export CMT_ALLOCATION_KL=0.5
export CMT_GAMMA=1.0
export CMT_SUCCESSOR_LAMBDA=1.0
export EVAL_INTERVAL=100
export SAVE_INTERVAL=50
export SEED=42

RUN_NAME="$OPD_RUN_NAME" bash scripts/train_opd_b200.sh
RUN_NAME="$TA_RUN_NAME"  bash scripts/train_ta_b200.sh
RUN_NAME="$RAC_RUN_NAME" bash scripts/train_rac_b200.sh
RUN_NAME="$CMT_RUN_NAME" bash scripts/train_cmt_b200.sh
```

`BATCH_SIZE` và `PPO_MINI_BATCH_SIZE` là global, không đổi theo world size. Ví dụ PPO batch 16
được chia thành 16/8/4 real trajectories mỗi GPU trên 1/2/4 GPU. Các rank dùng cùng global PPO
minibatch theo thứ tự interleaved deterministic; `MICRO_BATCH_SIZE_PER_GPU` chỉ điều khiển chunk
local và được cap ở local PPO share. Final partial minibatch được giữ nguyên sample count; nếu một
rank không có real sample để tham gia collective, training dừng với validation error thay vì âm
thầm dùng filler như một phần effective batch.

Các YAML OPD/TA/CMT khóa cùng `LR=5e-6`, rollout 4096 và eval interval 100. OPD thuần dùng uniform
weight `1` trên mọi valid response token; mọi launcher đi qua cùng `common_b200.sh`, do đó dùng
cùng Top-K OPD core, vLLM rollout, data order, model, seed, batch/micro-batch, LR, optimizer,
checkpoint và lịch
eval mặc định step 0 / mỗi 100 step / final. Nên chạy tuần tự trên cùng GPU layout để tránh nhiễu
tài nguyên giữa các run.

Training-time evaluation trên nhiều GPU được shard tự động: mỗi rank chạy một vLLM replica
`tensor_parallel_size=1`, xử lý một shard benchmark, rồi rank 0 merge về cùng format
`training_eval/step-*` như single-GPU. Trong thời gian generation không có rank nào chờ NCCL
barrier; filesystem sentinel phát hiện completion/failure, và chỉ dùng collective ngắn sau merge.
Vì vậy multi-GPU periodic evaluation yêu cầu `training_evaluation.backend=vllm`; HF fallback chỉ
được dùng khi chạy một GPU.

Full eval có 1.060 problem và `n=8`, vì vậy progress của vLLM hiển thị 8.480 generated responses
(`1.060 * 8`); đây không phải rollout train `n=1`. Để không lặp lại lượt base tốn thời gian, step 0
được cache theo fingerprint của model, dataset, evaluator và toàn bộ protocol. OPD sinh lần đầu;
TA-OPD/RAC copy đúng cùng prediction. Một run bị lỗi trước step train đầu tiên cũng tái sử dụng
`training_eval/step-000000` hợp lệ khi chạy lại. Có thể tắt bằng
`TRAIN_EVAL_REUSE_BASE=false`, hoặc đổi chỗ lưu qua `TRAIN_EVAL_BASE_CACHE_DIR`.

Không export `MAX_STEPS`, hoặc đặt `MAX_STEPS=-1`, để consume full configured epoch. `MAX_STEPS=N`
chỉ dành cho debug/explicit total target; nó không phải số step chạy thêm sau resume.

Một override hợp lệ phải áp giống nhau cho các method đang so sánh.

```bash
LR=5e-7 BATCH_SIZE=4 EVAL_INTERVAL=40 \
RUN_NAME="$OPD_RUN_NAME" bash scripts/train_opd_b200.sh
LR=5e-7 BATCH_SIZE=4 EVAL_INTERVAL=40 \
RUN_NAME="$TA_RUN_NAME" bash scripts/train_ta_b200.sh
LR=5e-7 BATCH_SIZE=4 EVAL_INTERVAL=40 \
RUN_NAME="$RAC_RUN_NAME" bash scripts/train_rac_b200.sh
```

Các OOM knob chính là `MICRO_BATCH_SIZE_PER_GPU`, `SCORE_MICRO_BATCH_SIZE`, `MAX_RESPONSE_LENGTH`,
`ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION`, `ROLLOUT_VLLM_MAX_NUM_SEQS` và
`ROLLOUT_VLLM_MAX_MODEL_LEN`. Chưa có peak-memory measurement trên B200 local, nên không coi default
là fit guarantee.

## Fast path không đổi protocol

Các launcher hiện mặc định bật cùng một nhóm tối ưu chính xác cho các method:

- vLLM rollout persistent, CUDA-IPC weight sync, prefix cache, chunked prefill, async scheduler và
  throughput scheduling;
- bỏ LM-head projection ở prompt positions của Qwen3 khi chỉ cần response logits;
- crop mọi suffix sau EOS và bucket trajectory theo response length ở scoring lẫn backward;
- scoring inference micro-batch 8 thay vì 1, đồng thời tái sử dụng `logsumexp` khi temperature bằng 1;
- TA/RAC/PGT/CMT lấy hai chiều student/teacher Top-K trong hai forward thay vì forward student lần thứ ba;
- CMT dùng conditional student/teacher Top-K union cho local geometry, raw-mass truncated common-mass
  transition và local-baseline excess scan; full-vocabulary reductions chỉ chạy khi
  `CMT_FULL_VOCAB_DIAGNOSTICS=true`;
- `top_p=1` là cấu hình exact khuyến nghị cho raw-kernel estimator; launcher không còn hard-fail
  với `top_p<1`, nhưng sẽ cảnh báo vì khi đó rollout distribution khác scored student `p`;
- không yêu cầu/serialize vLLM rollout token log-probs khi sync sanity check đang tắt.

Các mục này không đổi prompt, response, seed, sampling, Top-K OPD loss, selector hay optimizer step.
Có thể kiểm tra resolved config của một run tại `outputs/<run>/<method>/resolved_config.yaml`.

Trên đúng máy B200, speed knob có tác động lớn nhất tiếp theo là tăng trajectory micro-batch. Hãy
probe RAC (đường có peak memory lớn nhất) một step với cùng full length trước; mỗi probe phải dùng
tên output mới:

```bash
CUDA_VISIBLE_DEVICES=0,1 RUN_NAME=probe_rac_mb8 MAX_STEPS=1 \
  TRAIN_EVAL_ENABLED=false BATCH_SIZE=64 NUM_RESPONSES=1 \
  MICRO_BATCH_SIZE_PER_GPU=8 SCORE_MICRO_BATCH_SIZE=8 \
  bash scripts/train_rac_b200.sh
```

Nếu peak trong `outputs/probe_rac_mb8/rac_opd/metrics.jsonl` còn xa giới hạn, thử
`MICRO_BATCH_SIZE_PER_GPU=16`; nếu OOM thì lùi về `4`. Không đổi global `BATCH_SIZE=64` trong quá
trình tuning này. Có thể tune `SCORE_MICRO_BATCH_SIZE` riêng sau khi chọn training micro-batch.
Micro-batch chỉ chia cùng global weighted objective thành các lượt accumulate; hãy khóa đúng cùng
hai giá trị đã chọn cho OPD, TA và RAC. Sai khác floating-point ở mức thứ tự cộng gradient vẫn có
thể xảy ra khi đổi micro-batch, vì vậy không trộn các giá trị giữa ba run trong một comparison.

DDP chỉ còn là regression/debug option rõ ràng:

```bash
DISTRIBUTED_STRATEGY=ddp CUDA_VISIBLE_DEVICES=0,1 \
  BATCH_SIZE=64 NUM_RESPONSES=1 MICRO_BATCH_SIZE_PER_GPU=8 \
  bash scripts/train_opd_b200.sh
```

## TensorBoard

Rank 0 ghi event vào `outputs/<run>/<method>/tensorboard`. Theo dõi riêng một run:

```bash
tensorboard --logdir "outputs/$RAC_RUN_NAME/rac_opd/tensorboard" \
  --bind_all --port 6006
```

So sánh ba run trong cùng giao diện:

```bash
tensorboard --logdir_spec \
  "OPD:outputs/$OPD_RUN_NAME/opd/tensorboard,TA:outputs/$TA_RUN_NAME/ta_opd/tensorboard,RAC:outputs/$RAC_RUN_NAME/rac_opd/tensorboard" \
  --bind_all --port 6006
```

Production event giữ metric cũ và thêm Top-K loss/entropy/mass/overlap/divergence proxy, PPO ratio/clip/
advantage, response length min/mean/max/clip/throughput, VRAM allocated/reserved và step time.
TA thêm D/C/teachability/selected fraction; RAC thêm local teachability/alignment/V/weight stats;
CMT thêm support coverage/common mass/local-excess/sequential-gain/learning-value diagnostics.
Debug vLLM/HF log-prob MAE chỉ xuất hiện khi chủ động bật sanity validation. Mọi giá trị đã được
reduce toàn cục trước khi rank 0 ghi.

Full avg@8 trên 1.060 problem bắt buộc sinh 8.480 response cho mỗi checkpoint khác nhau; không thể
giảm con số này mà vẫn giữ nguyên metric. Nếu ưu tiên thời gian train không bị chặn bởi eval, có thể
lưu đúng mỗi 50 step, tắt eval inline, rồi chạy evaluator trên các checkpoint sau train (kết quả
checkpoint không đổi, nhưng về mặt vận hành đây không còn là eval đồng bộ bên trong train):

```bash
PAIR=$(date +%Y%m%d_%H%M%S)
export OPD_RUN_NAME="opd_qwen3_8b_to_1p7b_base_${PAIR}"
export TA_RUN_NAME="ta_qwen3_8b_to_1p7b_base_${PAIR}"
export RAC_RUN_NAME="rac_bellman_qwen3_8b_to_1p7b_base_${PAIR}"
export TRAIN_EVAL_TEMPERATURE=1
TRAIN_EVAL_ENABLED=false SAVE_INTERVAL=50 bash scripts/train_all_b200.sh

REEVAL_TEMPERATURE="$TRAIN_EVAL_TEMPERATURE" \
  OPD_RUN_NAME="$OPD_RUN_NAME" TA_RUN_NAME="$TA_RUN_NAME" RAC_RUN_NAME="$RAC_RUN_NAME" \
  CUDA_VISIBLE_DEVICES=0 bash scripts/reeval_all_checkpoints_b200.sh
```

Step 0 vẫn chỉ generate một lần nhờ shared fingerprint cache. Không đổi training `NUM_RESPONSES=1`,
`TRAIN_EVAL_NUM_RESPONSES=8` hoặc `MAX_RESPONSE_LENGTH` nếu mục tiêu là so sánh đúng protocol hiện
tại; các thay đổi đó nhanh hơn nhưng là thí nghiệm khác.

### Accuracy một response thay cho avg@8

Nếu không cần avg@8, đặt `TRAIN_EVAL_NUM_RESPONSES=1`. Mỗi checkpoint chỉ sinh một response cho
mỗi problem, tức 1.060 generation trên sáu bộ thay vì 8.480. Output, CSV và biểu đồ tự ghi nhãn
`accuracy`, không còn ghi nhầm `avg@8`:

```bash
TRAIN_EVAL_NUM_RESPONSES=1 TRAIN_EVAL_TEMPERATURE=1 \
  bash scripts/train_all_b200.sh
```

Eval final thủ công dùng `EVAL_NUM_RESPONSES=1`; eval lại và ghi đè mọi checkpoint dùng
`REEVAL_NUM_RESPONSES=1`:

```bash
OPD_RUN_NAME="$OPD_RUN_NAME" TA_RUN_NAME="$TA_RUN_NAME" RAC_RUN_NAME="$RAC_RUN_NAME" \
  EVAL_NUM_RESPONSES=1 EVAL_TEMPERATURE=1 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/eval_all_b200.sh

OPD_RUN_NAME="$OPD_RUN_NAME" TA_RUN_NAME="$TA_RUN_NAME" RAC_RUN_NAME="$RAC_RUN_NAME" \
  REEVAL_NUM_RESPONSES=1 REEVAL_TEMPERATURE=1 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/reeval_all_checkpoints_b200.sh
```

Với temperature 1 đây là sampled accuracy@1 theo seed cố định, không phải greedy decoding. Chế độ
n=1 nhanh hơn nhiều nhưng có variance lớn hơn n=8; phải dùng cùng n/temperature/top-p/seed cho cả
các method đang so sánh.

## Resume after interruption

Explicit checkpoint:

```bash
RUN_NAME="$OPD_RUN_NAME" \
RESUME_FROM_CHECKPOINT="outputs/$OPD_RUN_NAME/opd/checkpoint-000100" \
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_opd_b200.sh

RUN_NAME="$TA_RUN_NAME" \
RESUME_FROM_CHECKPOINT="outputs/$TA_RUN_NAME/ta_opd/checkpoint-000100" \
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_ta_b200.sh

RUN_NAME="$RAC_RUN_NAME" \
RESUME_FROM_CHECKPOINT="outputs/$RAC_RUN_NAME/rac_opd/checkpoint-000100" \
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_rac_b200.sh
```

Automatic latest complete checkpoint:

```bash
RUN_NAME="$OPD_RUN_NAME" RESUME=auto CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/train_opd_b200.sh
RUN_NAME="$TA_RUN_NAME" RESUME=auto CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/train_ta_b200.sh
RUN_NAME="$RAC_RUN_NAME" RESUME=auto CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/train_rac_b200.sh
```

`RESUME=auto` đọc `latest.json`, rồi fallback sang checkpoint hoàn chỉnh mới nhất. Resume validator
từ chối đổi method/scientific config. Nếu output đã có log/checkpoint mới hơn checkpoint được chọn,
resume sẽ tự rewind output về đúng step đó rồi ghi lại các step tiếp theo.

## Manual evaluation

Một launcher chung có thể eval checkpoint bất kỳ của các method. Khi checkpoint nằm trong
`outputs/<run>/<method>/`, nên bỏ qua đối số thứ ba: script sẽ ghi artifact chi tiết vào
`<method-output>/checkpoint_eval/<checkpoint-name>/` và upsert kết quả vào
`<method-output>/eval_history.jsonl` (cùng `eval_metrics.csv`).

```bash
CUDA_VISIBLE_DEVICES=0 EVAL_TEMPERATURE=1 EVAL_NUM_RESPONSES=1 \
  bash scripts/eval_checkpoint_b200.sh opd \
  outputs/my_opd_run/opd/checkpoint-000050

CUDA_VISIBLE_DEVICES=1 EVAL_TEMPERATURE=1 EVAL_NUM_RESPONSES=8 \
  bash scripts/eval_checkpoint_b200.sh ta-opd \
  outputs/my_ta_run/ta_opd/checkpoint-000100

CUDA_VISIBLE_DEVICES=2 EVAL_TEMPERATURE=1 EVAL_NUM_RESPONSES=1 \
  bash scripts/eval_checkpoint_b200.sh rac \
  outputs/my_rac_run/rac_opd/checkpoint-000150
```

`METHOD` nhận `opd`, `ta-opd` (hoặc `ta`), `rac`, `pgt` hoặc `cmt`. Script dùng vLLM mặc định; có thể đặt
`EVAL_BACKEND=hf`. Mặc định riêng của launcher chung là `temperature=1`, `n=8`; đặt
`EVAL_NUM_RESPONSES=1` để lấy accuracy một response thay vì avg@n.

Nếu truyền `OUTPUT_DIR` thứ ba, artifact vẫn được ghi vào thư mục đó; với layout checkpoint
chuẩn, history của run vẫn được cập nhật. Chạy lại cùng checkpoint sẽ thay row `(step, method)`
hiện có, không tạo duplicate.

Mỗi thư mục eval chứa `summary.json`, sáu file `*_predictions.jsonl.gz` (Competition-MATH,
MATH-500, AIME24, AIME25, GPQA-Diamond, AMC23). GPQA-Diamond dùng export
`id,prompt,ground_truth`, trong đó prompt đã có sẵn các lựa chọn A--D; loader cũng nhận
export cũ với các cột lựa chọn riêng. AMC23 dùng parquet
`nlp/minhpn19/data/amc23/test-00000-of-00001.parquet` (cột `id`, `question`, `answer`) và
cùng math prompt/verifier với các benchmark toán khác. File gộp
`model_outputs_detailed.jsonl.gz`. File gộp lưu dataset, ID, đề bài, đáp án chuẩn, prompt thực tế,
toàn bộ response, đúng/sai từng response và generation parameters. Đọc nhanh bằng:

```bash
gzip -cd outputs/my_opd_run/opd/checkpoint_eval/checkpoint-000050/model_outputs_detailed.jsonl.gz | less
```

Eval một RAC checkpoint trên đủ sáu benchmark Competition-MATH/MATH-500/AIME24/AIME25/
GPQA-Diamond/AMC23:

```bash
RUN_NAME="$RAC_RUN_NAME" \
RAC_CHECKPOINT="outputs/$RAC_RUN_NAME/rac_opd/checkpoint-000100" \
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_rac_b200.sh
```

Để bổ sung chỉ GPQA-Diamond và AMC23 vào history của một checkpoint cũ:

```bash
EVAL_BENCHMARKS="GPQA-Diamond,AMC23" \
  bash scripts/eval_checkpoint_b200.sh rac \
  outputs/my_rac_run/rac_opd/checkpoint-000100
```

Eval Base, OPD, TA và RAC final rồi aggregate:

```bash
OPD_RUN_NAME="$OPD_RUN_NAME" TA_RUN_NAME="$TA_RUN_NAME" RAC_RUN_NAME="$RAC_RUN_NAME" \
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_all_b200.sh
```

Evaluation mặc định dùng vLLM sampling (`n=8`, `temperature=0.7`, `top_p=0.95`,
`max_new_tokens=7168`) và báo cáo `avg@8`; mỗi problem lưu đủ 8 generation/correctness.
Fallback: `EVAL_BACKEND=hf`. Tensor parallel example:

```bash
OPD_RUN_NAME="$OPD_RUN_NAME" TA_RUN_NAME="$TA_RUN_NAME" RAC_RUN_NAME="$RAC_RUN_NAME" \
CUDA_VISIBLE_DEVICES=0,1 EVAL_VLLM_TENSOR_PARALLEL_SIZE=2 \
bash scripts/eval_all_b200.sh
```

## Re-eval mọi checkpoint đã lưu theo protocol avg@8

Script dưới đây tìm `checkpoint-<step>` và `final/` trong các output đã chọn. Nó cũng eval lại base ở
step 0 để toàn bộ đường avg@8 dùng cùng protocol. Mỗi `training_eval/step-*` tương ứng,
`eval_history.jsonl` và `eval_metrics.csv` cũ sẽ bị thay thế; raw prediction và `summary.json`
trong từng step cũng bị thay thế. Base được generate một lần rồi tái sử dụng cho hai method còn
lại; checkpoint sau train vẫn được eval độc lập. Đặt `REEVAL_REUSE_BASE=false` nếu cần cố ý sinh
lại base ba lần.

Kiểm tra danh sách checkpoint trước, không ghi file:

```bash
OPD_RUN_NAME="$OPD_RUN_NAME" TA_RUN_NAME="$TA_RUN_NAME" RAC_RUN_NAME="$RAC_RUN_NAME" \
  REEVAL_DRY_RUN=true \
  CUDA_VISIBLE_DEVICES=0 \
  bash scripts/reeval_all_checkpoints_b200.sh
```

Chạy thật:

```bash
OPD_RUN_NAME="$OPD_RUN_NAME" TA_RUN_NAME="$TA_RUN_NAME" RAC_RUN_NAME="$RAC_RUN_NAME" \
  CUDA_VISIBLE_DEVICES=0 \
  bash scripts/reeval_all_checkpoints_b200.sh
```

Tensor parallel cho từng evaluator vLLM:

```bash
OPD_RUN_NAME="$OPD_RUN_NAME" TA_RUN_NAME="$TA_RUN_NAME" RAC_RUN_NAME="$RAC_RUN_NAME" \
  CUDA_VISIBLE_DEVICES=0,1 \
  REEVAL_VLLM_TENSOR_PARALLEL_SIZE=2 \
  bash scripts/reeval_all_checkpoints_b200.sh
```

Script chạy tuần tự một subprocess vLLM cho mỗi checkpoint để trả VRAM sau từng lượt. Nó từ chối
ghi lịch sử mới nếu phát hiện thư mục eval cũ không có checkpoint tương ứng, tránh trộn kết quả
protocol cũ và mới. Sau khi hoàn tất, chạy lại lệnh plot periodic training evaluation bên dưới.

### Re-eval checkpoint của riêng một method

Launcher riêng nhận `METHOD` và `RUN_NAME`, không yêu cầu output của hai method còn lại. Luôn nên
dry-run trước để xem chính xác step 0, các `checkpoint-*` và `final` sẽ được xử lý:

```bash
REEVAL_DRY_RUN=true REEVAL_TEMPERATURE=1 REEVAL_NUM_RESPONSES=1 \
  CUDA_VISIBLE_DEVICES=0 \
  bash scripts/reeval_method_checkpoints_b200.sh opd "$OPD_RUN_NAME"
```

Chạy thật và ghi đè evaluation/history cũ của riêng OPD:

```bash
REEVAL_TEMPERATURE=1 REEVAL_NUM_RESPONSES=1 \
  CUDA_VISIBLE_DEVICES=0 \
  bash scripts/reeval_method_checkpoints_b200.sh opd "$OPD_RUN_NAME"
```

Đổi `opd` thành `ta`, `rac`, `pgt` hoặc `cmt` và truyền run name tương ứng. Với avg@8, đặt
`REEVAL_NUM_RESPONSES=8`. Nếu đã export `OPD_RUN_NAME`, có thể bỏ đối số run name. Cũng có thể
chỉ trực tiếp output không theo layout mặc định:

```bash
OPD_OUTPUT_DIR=/absolute/path/to/run/opd \
  REEVAL_TEMPERATURE=1 REEVAL_NUM_RESPONSES=1 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/reeval_method_checkpoints_b200.sh opd
```

## Plot periodic training evaluation

So sánh các phương pháp đã có output:

```bash
OPD_RUN_NAME="$OPD_RUN_NAME" TA_RUN_NAME="$TA_RUN_NAME" RAC_RUN_NAME="$RAC_RUN_NAME" \
  PLOT_METHODS="opd ta rac" \
  bash scripts/plot_training_progress.sh

# Khi đã train PGT:
OPD_RUN_NAME="$OPD_RUN_NAME" TA_RUN_NAME="$TA_RUN_NAME" RAC_RUN_NAME="$RAC_RUN_NAME" \
  PGT_RUN_NAME="$PGT_RUN_NAME" PLOT_METHODS="opd ta rac pgt" \
  bash scripts/plot_training_progress.sh

# Khi đã train CMT:
OPD_RUN_NAME="$OPD_RUN_NAME" TA_RUN_NAME="$TA_RUN_NAME" RAC_RUN_NAME="$RAC_RUN_NAME" \
  PGT_RUN_NAME="$PGT_RUN_NAME" CMT_RUN_NAME="$CMT_RUN_NAME" \
  PLOT_METHODS="opd ta rac pgt cmt" bash scripts/plot_training_progress.sh
```

So sánh hai phương pháp bất kỳ (đổi danh sách theo nhu cầu):

```bash
OPD_RUN_NAME="$OPD_RUN_NAME" RAC_RUN_NAME="$RAC_RUN_NAME" \
  PLOT_METHODS="opd rac" \
  bash scripts/plot_training_progress.sh --plot-name opd_vs_rac

OPD_RUN_NAME="$OPD_RUN_NAME" TA_RUN_NAME="$TA_RUN_NAME" \
  PLOT_METHODS="opd ta" \
  bash scripts/plot_training_progress.sh --plot-name opd_vs_ta

TA_RUN_NAME="$TA_RUN_NAME" RAC_RUN_NAME="$RAC_RUN_NAME" \
  PLOT_METHODS="ta rac" \
  bash scripts/plot_training_progress.sh --plot-name ta_vs_rac
```

Vẽ riêng một phương pháp để báo cáo, với sáu đường benchmark trong cùng một ảnh:

```bash
RUN_NAME="$OPD_RUN_NAME" PLOT_METHODS=opd \
  bash scripts/plot_training_progress.sh --plot-name opd_report

RUN_NAME="$TA_RUN_NAME" PLOT_METHOD=ta \
  bash scripts/plot_training_progress.sh --plot-name ta_report

RUN_NAME="$RAC_RUN_NAME" PLOT_METHOD=rac \
  bash scripts/plot_training_progress.sh --plot-name rac_report
```

`PLOT_METHODS` nhận danh sách cách nhau bởi dấu cách hoặc dấu phẩy. Chế độ riêng chỉ cần
`eval_history.jsonl`; chế độ so sánh còn đọc `metrics.jsonl` để vẽ loss. Mọi chế độ sinh cả
PNG/PDF cùng CSV/JSON số liệu. `PLOT_METHOD=both` cũ vẫn có nghĩa `ta rac`.

Sau final aggregate:

```bash
OPD_RUN_NAME="$OPD_RUN_NAME" TA_RUN_NAME="$TA_RUN_NAME" RAC_RUN_NAME="$RAC_RUN_NAME" \
  bash scripts/plot_results.sh
```

Chế độ so sánh tạo `results/<run1>_vs_<run2>.../plots/plot_YYYYMMDD_HHMMSS/`; chế độ riêng tạo
`results/<run>/plots/plot_YYYYMMDD_HHMMSS/`. Override tên folder:

```bash
TA_RUN_NAME="$TA_RUN_NAME" RAC_RUN_NAME="$RAC_RUN_NAME" \
  bash scripts/plot_training_progress.sh --plot-name paper_v1
```

## Output map

```text
outputs/<run>/<method>/
  resolved_config.yaml
  run_metadata.json
  metrics.jsonl
  train_metrics.csv
  eval_metrics.csv
  latest.json
  checkpoints or final/
  eval_history.jsonl
  training_eval/step-*/
  selector_scores/          # optional TA selected-token audit log (off by default)
  token_score_stats/        # compact all-valid-token histograms/quantiles/samples
```

Không kết luận method nào tốt hơn cho tới khi các controlled B200 run và full evaluation hoàn tất.
