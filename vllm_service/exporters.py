from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .profile_runtime import default_base_url, export_transport_config


def helm_bundle_dir(root: Path, profile_name: str) -> Path:
    return root / "generated" / "helm" / profile_name


def _maybe_repo_relative(root: Path, target: Path) -> str:
    try:
        return str(target.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(target.resolve())


def _service_endpoint_shape(service: dict[str, Any], deployment: dict[str, Any], *, base_url: str | None) -> dict[str, Any]:
    transport = export_transport_config(service, deployment, base_url=base_url)
    resolved_base_url = transport["base_url"] or default_base_url(deployment, explicit=base_url)
    protocol_mode = service.get("protocol_mode", "chat")
    return {
        "base_url": resolved_base_url,
        "models_path": f"{resolved_base_url}/models",
        "chat_completions_path": f"{resolved_base_url}/chat/completions",
        "completions_path": f"{resolved_base_url}/completions",
        "preferred_request_path": f"{resolved_base_url}/{'completions' if protocol_mode == 'completions' else 'chat/completions'}",
        "request_model_name": transport["request_model_name"],
        "transport_kind": transport["transport_kind"],
    }


def _helm_model_deployment(service: dict[str, Any], deployment: dict[str, Any], *, base_url: str | None) -> dict[str, Any]:
    transport = export_transport_config(service, deployment, base_url=base_url)
    args: dict[str, Any] = {"base_url": transport["base_url"]}
    if transport["transport_kind"] == "vllm-direct":
        args["vllm_model_name"] = transport["request_model_name"]
    else:
        args["api_key"] = transport["api_key_value"]
        args["openai_model_name"] = transport["request_model_name"]

    return {
        "name": transport["deployment_name"],
        "model_name": service["logical_model_name"],
        "tokenizer_name": service["tokenizer_name"],
        "max_sequence_length": int(service["max_model_len"]),
        "client_spec": {
            "class_name": transport["client_class"],
            "args": args,
        },
    }


def _manifest_template(
    *,
    experiment_name: str,
    description: str,
    model_name: str,
    model_deployment_name: str,
    model_deployments_fpath: str,
    max_eval_instances: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_name": experiment_name,
        "description": description,
        "run_entries": [f"ifeval:model={model_name},model_deployment={model_deployment_name}"],
        "max_eval_instances": max_eval_instances,
        "suite": experiment_name,
        "mode": "compute_if_missing",
        "materialize": "symlink",
        "backend": "tmux",
        "devices": 0,
        "tmux_workers": 1,
        "local_path": "prod_env",
        "precomputed_root": None,
        "require_per_instance_stats": True,
        "model_deployments_fpath": model_deployments_fpath,
        "enable_huggingface_models": [],
        "enable_local_huggingface_models": [],
    }


def export_helm_bundle(root: Path, deployment: dict[str, Any], *, base_url: str | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    services = deployment.get("services", [])
    if len(services) != 1:
        raise ValueError("HELM bundle export currently expects a single-service serving profile")
    service = services[0]
    if not service:
        raise ValueError("Cannot export a HELM bundle for a profile with no resolved services")

    bundle_dir = (output_dir or helm_bundle_dir(root, deployment["serving_profile"]["name"])).resolve()
    bundle_dir.mkdir(parents=True, exist_ok=True)
    endpoint_shape = _service_endpoint_shape(service, deployment, base_url=base_url)
    transport = export_transport_config(service, deployment, base_url=base_url)

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
            "kubernetes_name": service["kubernetes_name"],
        },
        "transport": {
            "backend": deployment["backend"],
            "router_type": deployment.get("router", {}).get("type", ""),
            "transport_kind": transport["transport_kind"],
            "request_model_name": transport["request_model_name"],
            "api_key_env": transport["api_key_env"],
            "api_key_value": transport["api_key_value"],
            "endpoint_shape": endpoint_shape,
        },
        "helm": {
            "deployment_name": transport["deployment_name"],
            "suggested_client_class": transport["client_class"],
            "model_deployments_path": str(model_deployments_path),
            "model_deployments_repo_relative": _maybe_repo_relative(root, model_deployments_path),
        },
        "artifacts": {
            "model_deployments": str(model_deployments_path),
        },
        "notes": service.get("audit_notes", []),
    }
    bundle_path = bundle_dir / "bundle.yaml"
    bundle_path.write_text(yaml.safe_dump(bundle, sort_keys=False), encoding="utf-8")

    model_deployments_fpath = _maybe_repo_relative(root, model_deployments_path)
    smoke_example = _manifest_template(
        experiment_name=f"{deployment['serving_profile']['name']}-smoke",
        description=f"Machine-local smoke manifest for {deployment['serving_profile']['public_name']}.",
        model_name=service["logical_model_name"],
        model_deployment_name=transport["deployment_name"],
        model_deployments_fpath=model_deployments_fpath,
        max_eval_instances=5,
    )
    full_example = _manifest_template(
        experiment_name=f"{deployment['serving_profile']['name']}-full",
        description=f"Machine-local full manifest for {deployment['serving_profile']['public_name']}.",
        model_name=service["logical_model_name"],
        model_deployment_name=transport["deployment_name"],
        model_deployments_fpath=model_deployments_fpath,
        max_eval_instances=1000,
    )
    smoke_path = bundle_dir / "smoke_manifest.yaml"
    full_path = bundle_dir / "full_manifest.yaml"
    smoke_path.write_text(yaml.safe_dump(smoke_example, sort_keys=False), encoding="utf-8")
    full_path.write_text(yaml.safe_dump(full_example, sort_keys=False), encoding="utf-8")

    return {
        "bundle_dir": bundle_dir,
        "bundle_path": bundle_path,
        "model_deployments_path": model_deployments_path,
        "smoke_manifest_path": smoke_path,
        "full_manifest_path": full_path,
        "bundle": bundle,
    }
