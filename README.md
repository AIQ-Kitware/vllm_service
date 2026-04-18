# vLLM Service

`vllm_service` manages **named serving profiles** for local and Kubernetes-backed inference.

A serving profile is a complete serving recipe for a model, including:

* which model to load
* what public name to serve it under
* how requests should reach it
* how it should use GPUs
* runtime settings such as tensor parallelism, context length, and batching

This repo can render and run those profiles through two backends:

* **Compose** for local single-host serving
* **KubeAI** for Kubernetes-backed serving

## Main idea

You do not work directly with raw model IDs most of the time.

You work with a **profile name**, for example:

* `gpt-oss-20b-completions`
* `gpt-oss-20b-chat`
* `qwen2-72b-instruct-tp2-balanced`

Each profile resolves to a concrete serving plan.

## Main commands

```bash id="jv5tgf"
python manage.py init
python manage.py list-profiles
python manage.py describe-profile <profile>
python manage.py render --profile <profile>
python manage.py up -d
python manage.py deploy
python manage.py switch <profile> --apply
python manage.py status
python manage.py smoke-test
```

## Inspect a profile before running it

This is the best way to understand what a profile will do:

```bash id="ecl6n7"
python manage.py describe-profile qwen2-72b-instruct-tp2-balanced --format yaml
```

That command prints the resolved serving contract for the profile, including model identity, access shape, placement, and runtime settings.  

---

## Backend 1: Compose

Use the Compose backend when you want the simplest local path on one machine.

### What it gives you

* easy local startup
* inspectable generated files
* simple iteration on one host
* a stable local API front door through LiteLLM

### What it does not give you

* Kubernetes-style model objects
* Kubernetes routing and deployment behavior
* cluster-managed serving

### Getting started

Initialize the repo, list profiles, inspect one, render it, and start it:

```bash id="6v6r5x"
python manage.py init
python manage.py list-profiles
python manage.py describe-profile gpt-oss-20b-completions --format yaml
python manage.py render --profile gpt-oss-20b-completions
python manage.py up -d
```

The Compose backend renders files such as:

* `generated/plan.yaml`
* `generated/docker-compose.yml`
* `generated/.env`
* runtime files under `state/runtime` 

### Test that it is responding

The built-in smoke test checks `/models` and then sends a chat request unless told not to. 

```bash id="e0if9v"
python manage.py smoke-test
```

You can also test it directly. The default Compose front door is:

```text
http://127.0.0.1:14000/v1
```

unless you changed the LiteLLM port in config. 

List models:

```bash id="t1q5yd"
curl http://127.0.0.1:14000/v1/models
```

Send a request:

```bash id="bby2mi"
curl http://127.0.0.1:14000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "openai/gpt-oss-20b",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 64
  }'
```

### Stop it

```bash id="gh9915"
python manage.py down
```

---

## Backend 2: KubeAI

Use the KubeAI backend when you want the same serving-profile model, but deployed through Kubernetes.

### What it adds beyond Compose

* profiles rendered as KubeAI/Kubernetes artifacts
* an OpenAI-compatible front door through KubeAI
* deployment through `kubectl` and Helm-backed KubeAI install flow
* profile switching through the Kubernetes path

### What changes compared to Compose

* you use `deploy` instead of `up`
* the default front door is `/openai/v1`
* generated artifacts go under `generated/kubeai/`
* you need a working Kubernetes cluster and KubeAI installed or installable

The current repo also includes a K3s bootstrap path, but the backend itself is `kubeai` on Kubernetes, not K3s-specific in concept. 

### Getting started

Start from the same repo, but set the backend to `kubeai` in `config.yaml`, then render and deploy.

A common flow is:

```bash id="31n0f7"
python manage.py init
python manage.py list-profiles
python manage.py describe-profile qwen2-72b-instruct-tp2-balanced --format yaml
python manage.py render --profile qwen2-72b-instruct-tp2-balanced
python manage.py deploy
python manage.py status
```

The KubeAI backend renders files such as:

* `generated/plan.yaml`
* `generated/kubeai/namespace.yaml`
* `generated/kubeai/kubeai-values.yaml`
* `generated/kubeai/models.yaml` 

### Test that it is responding

If you are not exposing ingress yet, port-forward the service:

```bash id="9c0hbv"
kubectl -n kubeai port-forward svc/kubeai 8000:80
```

Then run the smoke test against the KubeAI front door:

```bash id="l3rh5a"
python manage.py smoke-test --base-url http://127.0.0.1:8000/openai/v1
```

That `/openai/v1` path is the expected KubeAI OpenAI-compatible access shape in this repo.  

You can also test it directly.

List models:

```bash id="ce4zcf"
curl http://127.0.0.1:8000/openai/v1/models
```

Send a request:

```bash id="ajjlwm"
curl http://127.0.0.1:8000/openai/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2-72b-instruct-tp2-balanced",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 64
  }'
```

### Switch to another model profile

You can change the active serving profile and apply it through the backend:

```bash id="gdsq2p"
python manage.py switch gpt-oss-20b-completions --apply
python manage.py status
```

Then test again:

```bash id="ryav9p"
curl http://127.0.0.1:8000/openai/v1/models
```

This is a **profile switch and re-apply**, not a claim that requests automatically lazy-load arbitrary new models without redeployment. The CLI does support switching the active profile and applying it for both backends. 

---

## Which backend should I start with?

Start with **Compose** if you want:

* the fastest path to a working local server
* easy inspection of generated files
* simple single-host iteration

Move to **KubeAI** when you want:

* the same profile model on Kubernetes
* KubeAI’s OpenAI-compatible front door
* profile deployment through Kubernetes artifacts

The normal learning path is:

1. inspect a profile with `describe-profile`
2. run it with Compose
3. move to KubeAI when you want Kubernetes-backed serving

---

## Files you will look at most often

* `config.yaml`
* `models.yaml`
* `generated/plan.yaml`
* `generated/docker-compose.yml`
* `generated/kubeai/models.yaml`

---

## Typical workflow

### Local Compose iteration

```bash id="hfc0r4"
python manage.py list-profiles
python manage.py describe-profile gpt-oss-20b-completions --format yaml
python manage.py render --profile gpt-oss-20b-completions
python manage.py up -d
python manage.py smoke-test
```

### Kubernetes-backed serving with KubeAI

```bash id="nqim4m"
python manage.py list-profiles
python manage.py describe-profile qwen2-72b-instruct-tp2-balanced --format yaml
python manage.py render --profile qwen2-72b-instruct-tp2-balanced
python manage.py deploy
python manage.py status
kubectl -n kubeai port-forward svc/kubeai 8000:80
python manage.py smoke-test --base-url http://127.0.0.1:8000/openai/v1
```

### Switch profiles

```bash id="qnhmbo"
python manage.py switch qwen2-72b-instruct-tp2-balanced --apply
python manage.py switch gpt-oss-20b-completions --apply
```

---

## Troubleshooting

If something is unclear, check in this order:

```bash id="ybx2fe"
python manage.py list-profiles
python manage.py describe-profile <profile> --format yaml
python manage.py render --profile <profile>
python manage.py status
python manage.py smoke-test
```
