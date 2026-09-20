# Pure OPD vs TA-OPD vs CMT vs GRPO trên NVIDIA B200

Project độc lập này hỗ trợ bốn baseline chính trên cùng student Qwen3-1.7B-Base:

- **OPD thuần**: mọi valid response token có uniform weight `1`.
- **TA-OPD gốc**: local teachability và hard top-`rho` token budget.
- **Bellman-RAC**: cùng local teachability, truyền thông tin tương lai theo transition thực tế mà
  teacher ủng hộ, rồi weight mềm mọi response token.
- **PGT (Projected-Gradient Teachability)**: xếp hạng theo natural-gradient energy của local OPD
  trên union Student-Top-K và Teacher-Top-K; không dùng Bellman recurrence, critic hoặc
  counterfactual rollout. Đây là local baseline đang chờ one-step/held-out validation.
- **CMT-OPD (Coupled Marginal Teachability)**: dùng conditional student/teacher distributions trên
  union Student-Top-K và Teacher-Top-K để tính score, cộng vào local PGT gain một
  *local-baseline excess* successor-opportunity derivative qua **raw truncated** common-mass kernel
  `min(p,q)` trên union này, ước lượng trên đúng một
  full-policy student rollout bằng bounded acceptance factor (không inverse coverage), rồi phân bổ
  weight bằng global KL-constrained allocation. CMT
  không claim là causal task value hay teacher-policy value; xem
  [`CMT_REFINEMENT_DECISION.md`](CMT_REFINEMENT_DECISION.md) cho formulation hiện hành.
- **GRPO thuần (Group Relative Policy Optimization)**: teacher-free; mỗi prompt sinh `G` response,
  chấm outcome reward, chuẩn hoá advantage trong group và tối ưu clipped PPO surrogate. GRPO chỉ
  tải student, không tải teacher/reference model. Bellman-RAC/PGT vẫn còn trong code để đọc và
  tái lập các run legacy.

OPD/TA/CMT dùng chung data order, vLLM rollout, teacher scoring, **Top-K OPD core**, optimizer,
checkpoint và evaluation; GRPO dùng data order, rollout, optimizer, checkpoint và evaluation chung
nhưng không có teacher/Top-K distillation core. Core được port
từ [`thunlp/OPD`](https://github.com/thunlp/OPD) tại commit
`ac26e38d6f1572eb027597b48a9f4e01f6915ef8`.
Xem lệnh đầy đủ từ shell mới trong [`RUN_B200.md`](RUN_B200.md), hoặc runbook tiếng Việt có lệnh
copy-paste cho train/resume/eval/plot trong [`HUONG_DAN_CHAY.md`](HUONG_DAN_CHAY.md).

## Runtime paths

Mọi asset được resolve từ đúng một giá trị:

```yaml
paths:
  storage_root: /workspace/storage-shared
```

Có thể đổi ở launch time bằng `STORAGE_ROOT=/mount/khac`. Các path tương đối còn lại là:

| Asset | Relative path dưới `STORAGE_ROOT` |
|---|---|
| Teacher | `models/Qwen3-8B` |
| Student | `nlp/tungdd11/stable-on-policy-distillation/OPD/model/Qwen3-1.7B-Base` |
| Competition-MATH train | `nlp/minhpn19/data/competition_math/data/train-00000-of-00001.parquet` |
| Competition-MATH test | `nlp/minhpn19/data/competition_math/data/test-00000-of-00001.parquet` |
| MATH-500 | `nlp/minhpn19/data/eval/math500` |
| AIME 2024 | `nlp/minhpn19/data/eval/aime24` |
| AIME 2025 | `nlp/minhpn19/data/eval/aime25` |
| GPQA-Diamond | `nlp/minhpn19/data/GPQA-Diamond/gpqa_diamond.jsonl` |
| AMC23 | `nlp/minhpn19/data/amc23/test-00000-of-00001.parquet` |

Không cần sửa YAML khi đổi cặp model hoặc train dataset. Đầu mỗi launcher có block `USER CONFIG`
và nhận `TEACHER_MODEL`, `STUDENT_MODEL`, `TRAIN_DATA`, `PROMPT_KEY`; các tên legacy vẫn tương thích.
Teacher/student phải có cùng exact token-to-ID vocab,
added vocab, model vocab size và tokenizer length. Khác biệt `special_tokens_map` giữa Base và
Instruct không còn bị coi là lỗi vì student tokenizer là tokenizer duy nhất, và teacher scoring nhận
trực tiếp cùng integer IDs. Prompt dùng Qwen3 `enable_thinking=false`, nên teacher luôn no-think.

Training đọc toàn bộ file Competition-MATH train đã cấu hình (`split: null`). Nếu batch cuối không chia hết cho hai rank, rank ngắn hơn
dùng một trajectory filler xác định chỉ để giữ lịch collective FSDP giống nhau; filler bị loại khỏi
loss, selector, metric và log nên mỗi sample thật vẫn được dùng đúng một lần. Loader hỗ trợ parquet,
JSON và JSONL.

## Common Top-K OPD core

Tại mỗi prefix, student chọn `S_t = StudentTopK(p_t, K=16)`. Teacher được evaluate trên chính các
ID trong `S_t`. Theo recipe upstream `only_stu + student_p + token_reward_direct`, core tính:

```text
alpha_t,k = softmax_k(log p_t(k))
A_t,k     = (log q_t(k) - log p_old,t(k)) * alpha_t,k
ell_t     = sum_k PPOClippedLoss(log p_t(k), log p_old,t(k), A_t,k)
```

`K=16` là optimization support thật, không phải diagnostic và không còn objective sampled-token-only.
Sau đó các method dùng cùng global weighted-token normalization trên **global rollout batch**, bao gồm mọi FSDP rank:

```text
L = sum_t w_t * ell_t / sum_t w_t
```

OPD đặt `w_t=1`; TA đặt hard mask top-`rho`; RAC đặt continuous Bellman weight; PGT đặt hard mask
top-`rho` theo projected-gradient gain. OPD, TA và CMT đều dùng đúng Student-Top-K làm candidate support.
Vì vậy per-position signal và denominator convention hoàn toàn chung trong từng method.

## Định nghĩa TA-OPD

Tại mọi response position hợp lệ `t`, TA lấy
`U_t=TopK(student_t) union TopK(teacher_t)`, rồi renormalize cả hai phân phối trên `U_t`:

```text
D_t = KL(qbar_t^U || pbar_t^U)
C_t = sum_{v in TopK(student)} q_t(v)
```

`D` và `C` được robust-normalize trên **toàn global rollout batch**, kể cả khi dùng FSDP:

```text
Norm_B(z) = clip((z - Q05(z)) / (Q95(z) - Q05(z) + eps), 0, 1)
s_teach   = Norm_B(D) * Norm_B(C)
```

TA chọn chính xác `ceil(rho * N_valid)` vị trí có `s_teach` lớn nhất. Default `K=16`,
`rho=0.10`. Tie được giải ổn định theo flat global token index.

## Định nghĩa Bellman-RAC

Bellman-RAC không sinh token counterfactual và không tạo branch rollout. Nó tái sử dụng original
student trajectory và các score đã cần cho TA:

```text
g_t = s_teach_t
a_t = exp(clamp(log q_t(y_t) - log p_t(y_t), -20, 0))

R_t = g_t + gamma * a_t * R_{t+1}
M_t = 1   + gamma * a_t * M_{t+1}
V_t = R_t / (M_t + eps)

z_t = Norm_B(V_t)
w_t = w_min + (1 - w_min) * z_t^beta
```

Recurrence reset độc lập tại terminal/padding của từng response. Default `gamma=0.995`,
`w_min=0.10`, `beta=2.0`. Mọi statistic selector đều detached; gradient chỉ đi qua OPD token loss
trên original rollout:

```text
L_RAC = sum_t w_t * ell_t_OPD / (sum_t w_t + eps)
```

Vì `w_t >= w_min`, mọi valid response token đều nhận gradient. `rho` chỉ được TA dùng; nó vẫn nằm
trong shared config để audit fairness.

## Định nghĩa CMT-OPD

CMT dùng union conditional cho local score, tách biệt với Student Top-16 của loss:

```text
U_t       = TopK(student_t) union TopK(teacher_t)
p_U,q_U   = conditional p,q on U_t
r_U       = log q_U - log p_U
g_t       = Var_{p_U}[r_U]                         # local PGT gain
# sequential accessibility retains original p/q mass on U (no tail lookup)
# for y_t in U: p_raw=m_p*p_U, q_raw=m_q*q_U; otherwise k_hat_t=0
k_hat_t   = 1[y_t in U_t] * min(1,(m_q*q_U(y_t))/(m_p*p_U(y_t)))
R_t       = g_t + gamma*k_hat_t*R_{t+1}
M_t       = 1   + gamma*k_hat_t*M_{t+1}
H_t       = R_t - g_t*M_t
flux_t    = 1[y_t in U_t,m_p*p_U(y_t)<m_q*q_U(y_t)] * (r_U(y_t)-E_pU[r_U])
L_CMT,t   = g_t + lambda * gamma * flux_t * (R_{t+1} - g_t*M_{t+1})
w         = Gibbs(L_CMT; KL(w || uniform) <= epsilon)
```

`R/M/V` là cumulative/occupancy diagnostics; `H=R-g_tM_t` và `L_CMT` là production signals. Tail masses
`1-p(U)`/`1-q(U)` được log riêng, không đưa vào action simplex vì loss không tối ưu tail. Default
`gamma=1`, `lambda=1`, `epsilon=0.5`. Bounded-kernel unbiasedness giả định token được sample từ
scored student distribution; `top_p=1` là cấu hình exact khuyến nghị nhưng không còn là hard startup
requirement. Với `top_p<1`, factor vẫn bounded nhưng không còn unbiased cho kernel này; launcher chỉ
cảnh báo và không áp dụng importance correction không hợp lệ.
Đây là local conditional accessibility surrogate, không phải task value hay causal credit. Xem
[`CMT_REFINEMENT_DECISION.md`](CMT_REFINEMENT_DECISION.md) cho derivation và falsification matrix.

## Hot path

- Student rollout được tạo đúng một lần mỗi global rollout batch bằng persistent co-located vLLM server
  TP=1 trên từng rank. HF fallback chỉ dành cho single-process/DDP debug; production FSDP yêu cầu
  vLLM để mọi rank có lịch collective xác định.
- Common core score student Top-K và teacher-on-student IDs một lần. Với TA/RAC/CMT, student và teacher
  được forward chung theo từng bounded micro-batch. Shared scorer có thể vẫn materialize teacher Top-K
  để dựng union score của TA/CMT, nhưng các ID teacher-only không tham gia OPD loss. Mọi scoring chỉ
  giữ tensor `[B,T,K]` qua toàn rollout; hai full-vocabulary logit
  view BF16 chỉ cùng tồn tại bên trong một scoring micro-batch rồi được giải phóng.
  Default `n=1`: 64 prompt toàn cục tạo đúng 64 trajectory độc lập, tức 32 trajectory/GPU trên hai
  GPU; RAC, PGT và CMT không yêu cầu counterfactual generation.
- TA top-K/KL, actual-token log-prob ratio, global quantiles, RAC weights và loss weighting đều là
  tensor operations trên GPU, với BF16 model forward và FP32 cho logsumexp/score/reduction nhạy số.
- Bellman mặc định dùng associative affine suffix scan O(log T), vector hóa theo batch. Backend
  `reference` chứa recurrence rõ ràng để debug/cross-check.
- Student và frozen teacher đều dùng FSDP `FULL_SHARD`, Qwen3 decoder-layer auto wrap,
  `use_orig_params=True`, BF16 và FSDP-aware global gradient clipping. Trước mỗi rollout, mọi rank
  cùng materialize full student trên GPU và gửi current HF-named weights tới vLLM local qua CUDA IPC.

Không có speedup nào được khẳng định trước khi đo trên B200. Metrics tách riêng rollout, teacher
score, TA local score, Bellman scan, forward/backward, optimizer, evaluation, throughput và peak
allocated/reserved VRAM.

## Config công bằng

Các config baseline và config nghiên cứu:

```text
configs/qwen3_b200_base.yaml
configs/qwen3_b200_opd.yaml
configs/qwen3_b200_ta.yaml
configs/qwen3_b200_rac.yaml
configs/qwen3_b200_pgt.yaml
configs/qwen3_b200_cmt.yaml
```

Các method config chỉ override `experiment.method` và `experiment.output_dir`; toàn bộ model, data,
rollout, seed, batch, optimizer, schedule và evaluation settings được kế thừa từ cùng base config.
Các launcher đều expose:

```text
LR EPOCHS MAX_STEPS BATCH_SIZE NUM_RESPONSES
MAX_PROMPT_LENGTH OVERLONG_PROMPT_POLICY MAX_RESPONSE_LENGTH TOP_K TA_RHO PGT_RHO
PPO_MINI_BATCH_SIZE MICRO_BATCH_SIZE_PER_GPU
RAC_GAMMA RAC_W_MIN RAC_BETA RAC_SCAN_BACKEND
CMT_ALLOCATION_KL CMT_GAMMA CMT_SUCCESSOR_LAMBDA CMT_FULL_VOCAB_DIAGNOSTICS
EVAL_INTERVAL SAVE_INTERVAL LOG_INTERVAL SEED
```

Production defaults của OPD/TA/CMT là hai rank FSDP, BF16 full-parameter student, `LR=5e-6`, một epoch, global
prompt batch 64, `n=1`, PPO mini-batch toàn cục 16 trajectory, rồi micro-batch 8/GPU.
Prompt/response là `1024/4096`, eval mỗi 100 optimizer step và save mỗi 50 step. Micro-batch không tự
giảm khi OOM và LR không tự scale; thử `8 → 16`, rồi lùi về `4` nếu thiếu VRAM, giữ global batch 64.

Evaluation mặc định dùng vLLM `n=8`, `temperature=0.7`, `top_p=0.95`, `max_new_tokens=7168`; metric
chính là `avg@8`, tức mean của `number_correct/8` theo problem. Để eval lại toàn bộ checkpoint đã
lưu và thay thế lịch sử/file eval cũ, dùng `scripts/reeval_all_checkpoints_b200.sh`; để chỉ chạy một
method, dùng `scripts/reeval_method_checkpoints_b200.sh METHOD [RUN_NAME]`. Lệnh dry-run/chạy thật
nằm trong `RUN_B200.md`.
Các benchmark Competition-MATH test, MATH-500, AIME24, AIME25, GPQA-Diamond và AMC23
được đánh giá theo đúng `num_responses` đã cấu hình.
Step-0 base thường được generate một lần và cache có fingerprint; mọi checkpoint đã train vẫn eval riêng.
Khi các run được đánh giá độc lập, biểu đồ kiểm tra Step-0 của Competition-MATH và MATH-500
với dung sai tối đa hai điểm phần trăm; AIME không là điều kiện kiểm tra. Đường Base dùng
giá trị Step-0 đã căn chỉnh khi so sánh OPD với CMT-OPD. Evaluator bật vLLM
`performance_mode=throughput`, chunked prefill và async scheduling mặc định.

<!-- B200_AUTOTUNE_RESULT_START -->
**Measured target result:** chưa chạy trên máy B200; `scripts/smoke_test_b200.sh` sẽ cập nhật block này.
<!-- B200_AUTOTUNE_RESULT_END -->

## Output và resume

Fresh launch tự tạo tên:

```text
opd_qwen3_8b_to_1p7b_base_YYYYMMDD_HHMMSS
ta_qwen3_8b_to_1p7b_base_YYYYMMDD_HHMMSS
rac_bellman_qwen3_8b_to_1p7b_base_YYYYMMDD_HHMMSS
```

Mỗi output chứa `resolved_config.yaml`, metadata, `metrics.jsonl`, `train_metrics.csv`, TensorBoard,
`eval_metrics.csv`, periodic full eval, compact `token_score_stats`, checkpoints và `latest.json`.
Checkpoint được ghi qua temporary directory rồi atomic rename. Mọi rank tham gia full state-dict;
rank 0 ghi checkpoint HF load trực tiếp được và FSDP full optimizer state portable cùng step/RNG.
Dùng checkpoint cụ thể hoặc `RESUME=auto` cùng tên run cũ.

Eval thủ công một checkpoint bất kỳ dùng
`scripts/eval_checkpoint_b200.sh METHOD CHECKPOINT [OUTPUT_DIR]`, trong đó `METHOD` là `opd`,
`ta-opd`, `rac`, `pgt` hoặc `cmt`. Với checkpoint nằm dưới `outputs/<run>/<method>/` và bỏ qua
`OUTPUT_DIR`, artifact chi tiết được ghi dưới `<method-output>/checkpoint_eval/` và kết quả được
upsert vào `<method-output>/eval_history.jsonl` (cập nhật cùng row `(step, method)` khi chạy lại),
đồng thời cập nhật `eval_metrics.csv`. Ngoài summary và prediction theo từng dataset, evaluator
HF/vLLM đều ghi `model_outputs_detailed.jsonl.gz` gồm prompt đã render, reference answer, mọi
model response và kết quả chấm từng response. Ví dụ lệnh đầy đủ nằm trong `RUN_B200.md`.

Detailed TA selected-token JSONL được tắt mặc định để tránh tăng disk không giới hạn; compact global
histogram/quantile và bounded scalar sample vẫn luôn đủ cho plots. Có thể chủ động bật bằng
`--set logging.selected_tokens_enabled=true` cho một run audit ngắn.

Plot launch tạo một folder timestamp mới `results/.../plots/plot_YYYYMMDD_HHMMSS/`, sinh PNG và PDF
cho avg@8, loss, TA score distribution, Bellman-RAC `g/V/w`, CMT support/excess diagnostics,
và mean alignment/V/weight. `plot_training_progress.sh` hỗ trợ
`PLOT_METHODS='opd ta rac pgt cmt'` với một hoặc nhiều method.
Một method tạo sáu đường benchmark (thêm GPQA-Diamond và AMC23); từ hai method trở lên tạo sáu subplot benchmark,
mỗi subplot có một đường cho từng method. `PLOT_METHOD=both` vẫn tương thích và có nghĩa TA+RAC.

## Validation

Các test nhẹ bao phủ Top-K OPD khớp upstream trên synthetic logits, `n=1`, global
weighted-token mean, FSDP batching/accumulation, uniform pure-OPD allocation, literal TA formula,
Bellman recurrence tính tay,
padding/trajectory reset, bounds, detachment, gradient path, optimized/reference equivalence,
global distributed normalization/budget, tail padding, resume, eval schedule, TensorBoard diagnostics,
loaders, vLLM wrappers và plotting. Test tích hợp hai CUDA rank kiểm tra FULL_SHARD student/teacher,
HF parameter names/shapes, FSDP-aware update, full checkpoint và optimizer scatter-resume.

```bash
python -m unittest discover -s tests -v
ruff check b200_experiment tests
bash -n scripts/*.sh
```

`scripts/smoke_test_b200.sh` là diagnostic tùy chọn: nó chạy CPU-only unit tests rồi preflight
model/data/GPU. Launcher train không đọc marker/report và bắt đầu trực tiếp. Smoke không tự bắt đầu
full training trừ khi chủ động bật batch autotune. Mỗi tổ hợp teacher/student/data và số GPU có report riêng tại
`results/preflight/asset-<fingerprint>-<N>gpu.json`; autotune cũng tách theo số GPU tại
`configs/autotuned/asset-<fingerprint>-<N>gpu.yaml`. Vì vậy các workload dùng chung checkout không
đọc hoặc ghi đè validation của cấu hình khác. Khi autotune được bật, step đầu của OPD/TA/RAC còn
phải có cùng rollout hash và đúng allocation policy trước khi batch candidate được chấp nhận.
`scripts/smoke_test_fsdp_multigpu.sh` là smoke thật hai step trên mọi GPU đang visible (tối thiểu
hai GPU), bao gồm lần sync weight thứ hai, vLLM/HF log-prob check, checkpoint reload và kiểm tra
chính xác TensorBoard tags. Tên cũ `smoke_test_fsdp_2gpu.sh` vẫn tương thích.
