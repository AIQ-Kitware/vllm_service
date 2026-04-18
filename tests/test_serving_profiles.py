from __future__ import annotations

from pathlib import Path

import yaml

from vllm_service.backends.compose_renderer import render_compose_artifacts
from vllm_service.backends.kubeai_renderer import render_kubeai_artifacts
from vllm_service.config import initial_config
from vllm_service.exporters import export_helm_bundle
from vllm_service.hardware import simulate_inventory
from vllm_service.resolver import resolve
from vllm_service.validator import validate_resolved


def _cfg(tmp_path: Path, *, backend: str = "compose") -> dict:
    cfg = initial_config()
    cfg["backend"] = backend
    cfg["state"] = {
        "hf_cache": "state/hf-cache",
        "open_webui": "state/open-webui",
        "postgres": "state/postgres",
        "runtime": "state/runtime",
    }
    cfg["ports"] = {"litellm": 14000, "open_webui": 13000, "postgres": 15432}
    return cfg


def _plan(tmp_path: Path, profile_name: str, *, backend: str = "compose", inventory: str = "4x96") -> dict:
    cfg = _cfg(tmp_path, backend=backend)
    deployment = resolve(tmp_path, cfg, inventory=simulate_inventory(inventory), profile_name=profile_name)
    validated = validate_resolved(deployment)
    assert validated["ok"], validated
    return {"deployment": deployment, "validated": validated}


def test_profile_resolution_uses_named_serving_profile(tmp_path: Path) -> None:
    deployment = _plan(tmp_path, "qwen2-72b-instruct-tp2-balanced")["deployment"]
    assert deployment["serving_profile"]["public_name"] == "qwen2-72b-instruct-tp2-balanced"
    assert deployment["serving_profile"]["logical_model_name"] == "qwen/qwen2-72b-instruct"
    assert deployment["services"][0]["tensor_parallel_size"] == 2
    assert "qwen2-72b-instruct-tp2-balanced" in deployment["router"]["aliases"]


def test_legacy_profile_alias_resolves_to_canonical_profile(tmp_path: Path) -> None:
    deployment = _plan(tmp_path, "helm-qwen2-72b-instruct")["deployment"]
    assert deployment["serving_profile"]["name"] == "qwen2-72b-instruct-tp2-balanced"


def test_kubeai_render_uses_profile_identity(tmp_path: Path) -> None:
    plan = _plan(tmp_path, "qwen2-72b-instruct-tp2-balanced", backend="kubeai")
    render_kubeai_artifacts(tmp_path, plan)
    models = list(yaml.safe_load_all((tmp_path / "generated" / "kubeai" / "models.yaml").read_text()))
    assert models[0]["metadata"]["name"] == "qwen2-72b-instruct-tp2-balanced"
    assert models[0]["metadata"]["annotations"]["vllm-service/logical-model-name"] == "qwen/qwen2-72b-instruct"
    assert "--tensor-parallel-size=2" in models[0]["spec"]["args"]


def test_compose_render_includes_profile_labels_and_aliases(tmp_path: Path) -> None:
    plan = _plan(tmp_path, "gpt-oss-20b-chat")
    render_compose_artifacts(tmp_path, plan)
    compose_text = (tmp_path / "generated" / "docker-compose.yml").read_text()
    litellm_text = (tmp_path / "state" / "runtime" / "litellm_config.yaml").read_text()
    assert 'vllm_service.public_name: "gpt-oss-20b-chat"' in compose_text
    assert "openai/gpt-oss-20b" in litellm_text
    assert "gpt-oss-20b-chat" in litellm_text


def test_export_bundle_distinguishes_gpt_oss_chat_vs_completions(tmp_path: Path) -> None:
    chat_plan = _plan(tmp_path, "gpt-oss-20b-chat")
    comp_plan = _plan(tmp_path, "gpt-oss-20b-completions")
    chat = export_helm_bundle(tmp_path, chat_plan["deployment"], output_dir=tmp_path / "chat")
    comp = export_helm_bundle(tmp_path, comp_plan["deployment"], output_dir=tmp_path / "completions")
    chat_bundle = yaml.safe_load(chat["bundle_path"].read_text())
    comp_bundle = yaml.safe_load(comp["bundle_path"].read_text())
    assert chat_bundle["profile"]["protocol_mode"] == "chat"
    assert comp_bundle["profile"]["protocol_mode"] == "completions"
    assert chat_bundle["helm"]["suggested_client_class"].endswith("OpenAIClient")
    assert comp_bundle["helm"]["suggested_client_class"].endswith("OpenAILegacyCompletionsClient")


def test_export_bundle_writes_model_deployments_for_qwen_profile(tmp_path: Path) -> None:
    plan = _plan(tmp_path, "qwen2-5-7b-instruct-turbo-default")
    result = export_helm_bundle(tmp_path, plan["deployment"])
    model_deployments = yaml.safe_load(result["model_deployments_path"].read_text())
    deployment = model_deployments["model_deployments"][0]
    assert deployment["model_name"] == "qwen/qwen2.5-7b-instruct-turbo"
    assert deployment["name"] == "local/qwen2-5-7b-instruct-turbo-default"
    assert deployment["client_spec"]["class_name"].endswith("OpenAIClient")
