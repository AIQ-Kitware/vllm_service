# Deploy Qwen3.5-9B on Compose, pinned to GPU 1 of a 2x RTX 3090 workstation

This recipe walks through using the `compose` backend in `vllm_service` to run:

- model: `Qwen/Qwen3.5-9B`
- target: **one RTX 3090**
- GPU selection: **GPU 1 only**
- goal: **full native context**
- constraint: **leave GPU 0 free for other work**

The instructions are written as executable shell steps. Read the notes before running each section.

---

## 0) Preconditions

Assumptions:

- you already have a working checkout of `AIQ-Kitware/vllm_service`
- Docker and the NVIDIA container runtime are installed
- `nvidia-smi` shows **two** RTX 3090 GPUs
- you want the service to run **only on GPU 1**

Move into the repo first:

```bash
cd /path/to/vllm_service
pwd
git status --short
```

Verify your GPUs and confirm the numbering:

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv
```

You want output that shows something like:

- GPU 0, NVIDIA GeForce RTX 3090, 24576 MiB
- GPU 1, NVIDIA GeForce RTX 3090, 24576 MiB

---

## 1) Create or edit `models.yaml`

The built-in catalog is conservative for `qwen3.5-9b`. For this use case, create a local override that:

- sets the model context window to the full native value
- adds a dedicated Compose profile pinned to GPU 1
- keeps concurrency conservative so full-context operation has the best chance to fit on a 24 GB card

Create `models.yaml` in the repo root if it does not already exist:

```bash
cat > models.yaml <<'YAML'
models:
  qwen3.5-9b:
    hf_model_id: Qwen/Qwen3.5-9B
    served_model_name: qwen3.5-9b
    family: qwen3.5
    modalities:
      - text
    memory_class_gib: 20
    min_vram_gib_per_replica: 20
    preferred_gpu_count: 1
    context_window: 262144
    defaults:
      max_model_len: 262144
      gpu_memory_utilization: 0.90
      enable_prefix_caching: true
      max_num_batched_tokens: 4096
      max_num_seqs: 1
      enable_chunked_prefill: true

profiles:
  qwen3.5-9b-3090-fullctx-gpu1:
    description: "Qwen3.5 9B full-context profile pinned to GPU 1 on a 2x3090 workstation."
    vllm:
      enable_responses_api_store: false
      logging_level: INFO
    services:
      - service_name: qwen-main
        model: qwen3.5-9b
        served_model_name: qwen3.5-9b
        placement:
          strategy: single_gpu
          gpu_indices: [1]
          prefer_homogeneous_group: true
        topology:
          tp: 1
          dp: 1
        runtime:
          max_model_len: 262144
          gpu_memory_utilization: 0.90
          max_num_seqs: 1
          max_num_batched_tokens: 4096
          enable_prefix_caching: true
          enable_chunked_prefill: true
    router:
      aliases:
        qwen3.5-9b: qwen-main
    policy:
      require_fit_validation: true
      minimum_vram_headroom_gib: 2
      allow_unsupported_render: false
YAML
```

Sanity-check the file:

```bash
sed -n '1,220p' models.yaml
```

---

## 2) Select the Compose backend and the new profile

Configure the repo to use the Compose backend and your new profile:

```bash
python manage.py setup --backend compose --profile qwen3.5-9b-3090-fullctx-gpu1
```

Inspect the resolved profile:

```bash
python manage.py describe-profile qwen3.5-9b-3090-fullctx-gpu1 --format yaml
```

Look for these points in the output:

- `model: qwen3.5-9b`
- `max_model_len: 262144`
- `strategy: single_gpu`
- `gpu_indices: [1]`
- `tp: 1`
- `dp: 1`

---

## 3) Validate before launching

Run validation before rendering or launching:

```bash
python manage.py validate --profile qwen3.5-9b-3090-fullctx-gpu1
```

If validation passes, continue.

If validation fails because the profile is too aggressive for a 24 GB card, skip ahead to the troubleshooting section and reduce memory pressure.

---

## 4) Render the Compose deployment

Render the deployment artifacts:

```bash
python manage.py render --profile qwen3.5-9b-3090-fullctx-gpu1
```

At this point, keep the terminal output. If you want me to validate the generated Compose file, copy it or upload it after this step.

---

## 5) Launch the service

Bring the Compose deployment up in detached mode:

```bash
python manage.py up --profile qwen3.5-9b-3090-fullctx-gpu1 -d
```

Give the container a bit of time to start and load weights.

Check that only **GPU 1** is being consumed:

```bash
watch -n 1 nvidia-smi
```

You should see the model server consuming memory on:

- **GPU 1**: high memory usage
- **GPU 0**: unchanged / still free for your other tasks

Stop `watch` with `Ctrl+C` once you have confirmed this.

---

## 6) Smoke-test the service

First, ask the repo to smoke-test the model:

```bash
python manage.py smoke-test --model qwen3.5-9b
```

Then run a direct API check if you want an extra confirmation. Use the endpoint printed by your deployment, or try the common local OpenAI-compatible endpoint:

```bash
curl -s http://127.0.0.1:8000/v1/models | python -m json.tool
```

If that succeeds, try a tiny chat completion:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.5-9b",
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
    "max_tokens": 32
  }' | python -m json.tool
```

---

## 7) Verify the full-context configuration is actually active

A deployment can launch with a lower runtime context than you intended, so verify it from the resolved profile first:

```bash
python manage.py describe-profile qwen3.5-9b-3090-fullctx-gpu1 --format yaml | grep -n "max_model_len"
```

You want to see:

```text
max_model_len: 262144
```

If you later send me the generated Compose file, I can help verify the vLLM container arguments directly.

---

## 8) Optional quick benchmark

Run the repo's benchmark helper:

```bash
python manage.py benchmark --model qwen3.5-9b
```

This is not a long-context stress test by itself, but it is a quick health check that the service is usable.

---

## 9) Stop the deployment when you are done

```bash
python manage.py down --profile qwen3.5-9b-3090-fullctx-gpu1
```

---

## 10) Troubleshooting

### A) Validation fails or launch OOMs on a 24 GB 3090

The full-context target is aggressive. Start by lowering pressure in this order.

First, reduce `gpu_memory_utilization`:

```bash
python - <<'PY'
from pathlib import Path
p = Path("models.yaml")
text = p.read_text()
text = text.replace("gpu_memory_utilization: 0.90", "gpu_memory_utilization: 0.88")
p.write_text(text)
print("Updated gpu_memory_utilization to 0.88")
PY
```

Then rerun:

```bash
python manage.py validate --profile qwen3.5-9b-3090-fullctx-gpu1
python manage.py up --profile qwen3.5-9b-3090-fullctx-gpu1 -d
```

If that still fails, reduce `max_num_batched_tokens`:

```bash
python - <<'PY'
from pathlib import Path
p = Path("models.yaml")
text = p.read_text()
text = text.replace("max_num_batched_tokens: 4096", "max_num_batched_tokens: 2048")
p.write_text(text)
print("Updated max_num_batched_tokens to 2048")
PY
```

If that still fails, lower `max_model_len` as the last resort:

```bash
python - <<'PY'
from pathlib import Path
p = Path("models.yaml")
text = p.read_text()
text = text.replace("max_model_len: 262144", "max_model_len: 131072")
p.write_text(text)
print("Updated max_model_len to 131072")
PY
```

Then re-run setup/validate/up:

```bash
python manage.py setup --backend compose --profile qwen3.5-9b-3090-fullctx-gpu1
python manage.py validate --profile qwen3.5-9b-3090-fullctx-gpu1
python manage.py up --profile qwen3.5-9b-3090-fullctx-gpu1 -d
```

### B) The service starts on GPU 0 instead of GPU 1

Re-check the resolved profile:

```bash
python manage.py describe-profile qwen3.5-9b-3090-fullctx-gpu1 --format yaml
```

The placement section must still be:

```yaml
placement:
  strategy: single_gpu
  gpu_indices: [1]
```

If it is not, your local `models.yaml` was not picked up or was overridden.

### C) You want to inspect the generated Compose file

After `python manage.py render ...`, upload or paste the generated Compose YAML and I will validate that it is consistent with:

- Compose backend
- `qwen3.5-9b-3090-fullctx-gpu1`
- GPU 1 only
- TP=1 / DP=1
- full native context

---

## 11) Minimal rerun loop during tuning

When you are iterating on `models.yaml`, this is the shortest useful loop:

```bash
python manage.py setup --backend compose --profile qwen3.5-9b-3090-fullctx-gpu1
python manage.py validate --profile qwen3.5-9b-3090-fullctx-gpu1
python manage.py render --profile qwen3.5-9b-3090-fullctx-gpu1
python manage.py up --profile qwen3.5-9b-3090-fullctx-gpu1 -d
python manage.py smoke-test --model qwen3.5-9b
```

If you send me the generated Compose file after the `render` step, I can help validate the concrete container args and the GPU targeting.
