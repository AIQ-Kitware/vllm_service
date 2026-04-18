from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .profile_runtime import default_base_url, deployment_client_args


def helm_bundle_dir(root: Path, profile_name: str) -> Path:
    return root / "generated" / "helm" / profile_name


def _service_endpoint_shape(service: dict[str, Any], deployment: dict[str, Any], *, base_url: str | None) -> dict[str, Any]:
    resolved_base_url = default_base_url(deployment, explicit=base_url)
    protocol_mode = service.get("protocol_mode", "chat")
    return {
        "base_url": resolved_base_url,
        "models_path": f"{resolved_base_url}/models",
        "chat_completions_path": f"{resolved_base_url}/chat/completions",
        "completions_path": f"{resolved_base_url}/completions",
        "preferred_request_path": f"{resolved_base_url}/{'completions' if protocol_mode == 'completions' else 'chat/completions'}",
    }


def _helm_model_deployment(service: dict[str, Any], deployment: dict[str, Any], *, base_url: str | None) -> dict[str, Any]:
    client = deployment_client_args(service, deployment, base_url=base_url)
    args: dict[str, Any] = {"base_url": client["base_url"]}
    if client["client_class"].endswith("OpenAIClient") or client["client_class"].endswith("OpenAILegacyCompletionsClient"):
        args["api_key"] = f"ENV[{client['api_key_env']}]"
        args["openai_model_name"] = service["served_model_name"]
    else:
        args["vllm_model_name"] = service["served_model_name"]

    return {
        "name": f"local/{service['profile_public_name']}",
        "model_name": service["logical_model_name"],
        "tokenizer_name": service["tokenizer_name"],
        "max_sequence_length": int(service["max_model_len"]),
        "client_spec": {
            "class_name": client["client_class"],
            "args": args,
        },
    }


def export_helm_bundle(root: Path, deployment: dict[str, Any], *, base_url: str | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    services = deployment.get("services", [])
    if len(services) != 1:
        raise ValueError("HELM bundle export currently expects a single-service serving profile")
    service = services[0]
    if not service:
        raise ValueError("Cannot export a HELM bundle for a profile with no resolved services")

    bundle_dir = output_dir or helm_bundle_dir(root, deployment["serving_profile"]["name"])
    bundle_dir.mkdir(parents=True, exist_ok=True)
    endpoint_shape = _service_endpoint_shape(service, deployment, base_url=base_url)
    client = deployment_client_args(service, deployment, base_url=base_url)

    model_deployments = {"model_deployments": [_helm_model_deployment(service, deployment, base_url=base_url)]}
    model_deployments_path = bundle_dir / "model_deployments.yaml"
    model_deployments_path.write_text(yaml.safe_dump(model_deployments, sort_keys=False), encoding="utf-8")

    bundle = {
        "target": "helm",
        "profile": {
            "name": deployment["serving_profile"]["name"],
            "public_name": deployment["serving_profile"]["public_name"],
            "logical_model_name": service["logical_model_name"],
            "served_model_name": service["served_model_name"],
            "served_aliases": service.get("served_aliases", []),
            "base_model": service["model_ref"],
            "hf_model_id": service["hf_model_id"],
            "tokenizer_name": service["tokenizer_name"],
            "protocol_mode": service["protocol_mode"],
            "engine": service["engine"],
            "resource_profile": service["resource_profile"],
        },
        "transport": {
            "backend": deployment["backend"],
            "router_type": deployment.get("router", {}).get("type", ""),
            "endpoint_shape": endpoint_shape,
        },
        "helm": {
            "deployment_name": f"local/{service['profile_public_name']}",
            "suggested_client_class": client["client_class"],
            "model_deployments_path": str(model_deployments_path),
        },
        "artifacts": {
            "model_deployments": str(model_deployments_path),
        },
        "notes": service.get("audit_notes", []),
    }
    bundle_path = bundle_dir / "bundle.yaml"
    bundle_path.write_text(yaml.safe_dump(bundle, sort_keys=False), encoding="utf-8")

    smoke_example = {
        "local_path": "prod_env",
        "model_deployments_fpath": str(model_deployments_path),
        "model": service["logical_model_name"],
        "endpoint": endpoint_shape["preferred_request_path"],
    }
    (bundle_dir / "smoke-manifest.example.yaml").write_text(
        yaml.safe_dump(smoke_example, sort_keys=False),
        encoding="utf-8",
    )
    return {
        "bundle_dir": bundle_dir,
        "bundle_path": bundle_path,
        "model_deployments_path": model_deployments_path,
        "bundle": bundle,
    }
