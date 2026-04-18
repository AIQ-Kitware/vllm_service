from __future__ import annotations

from typing import Any


def vllm_args(service: dict[str, Any]) -> list[str]:
    args = [
        f"--served-model-name={service['served_model_name']}",
        f"--tensor-parallel-size={service['tensor_parallel_size']}",
        f"--data-parallel-size={service['data_parallel_size']}",
        f"--max-model-len={service['max_model_len']}",
        f"--gpu-memory-utilization={service['gpu_memory_utilization']}",
        f"--max-num-batched-tokens={service['max_num_batched_tokens']}",
        f"--max-num-seqs={service['max_num_seqs']}",
        "--disable-log-requests",
    ]
    if service.get("enable_prefix_caching"):
        args.append("--enable-prefix-caching")
    if service.get("enable_auto_tool_choice"):
        args.append("--enable-auto-tool-choice")
        if service.get("tool_call_parser"):
            args.append(f"--tool-call-parser={service['tool_call_parser']}")
    args.extend(service.get("extra_args", []))
    return args


def default_base_url(deployment: dict[str, Any], *, explicit: str | None = None) -> str:
    if explicit:
        return explicit.rstrip("/")
    backend = deployment.get("backend", "compose")
    if backend == "kubeai":
        ingress = deployment.get("cluster", {}).get("ingress", {}) or {}
        host = ingress.get("host", "")
        if ingress.get("enabled") and host:
            return f"http://{host}/openai/v1"
        return "http://127.0.0.1:8000/openai/v1"
    return f"http://127.0.0.1:{deployment.get('ports', {}).get('litellm', 14000)}/v1"


def suggested_client_class(protocol_mode: str, backend: str) -> str:
    protocol = protocol_mode.lower()
    if backend == "compose":
        if protocol == "completions":
            return "helm.clients.openai_client.OpenAILegacyCompletionsClient"
        return "helm.clients.openai_client.OpenAIClient"
    if protocol == "completions":
        return "helm.clients.vllm_client.VLLMClient"
    return "helm.clients.vllm_client.VLLMChatClient"


def deployment_client_args(service: dict[str, Any], deployment: dict[str, Any], *, base_url: str | None = None) -> dict[str, Any]:
    resolved_base_url = default_base_url(deployment, explicit=base_url)
    backend = deployment.get("backend", "compose")
    protocol = service.get("protocol_mode", "chat")
    if backend == "compose":
        return {
            "base_url": resolved_base_url,
            "api_key_env": "LITELLM_MASTER_KEY",
            "openai_model_name": service["served_model_name"],
            "client_class": suggested_client_class(protocol, backend),
        }
    return {
        "base_url": resolved_base_url,
        "api_key": "EMPTY",
        "vllm_model_name": service["served_model_name"],
        "client_class": suggested_client_class(protocol, backend),
    }
