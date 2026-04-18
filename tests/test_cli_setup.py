from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml


MANAGE_PY = Path(__file__).resolve().parents[1] / "manage.py"


def run_cli(tmp_path: Path, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    return subprocess.run(
        [sys.executable, str(MANAGE_PY), *args],
        cwd=tmp_path,
        env=full_env,
        text=True,
        capture_output=True,
        check=True,
    )


def test_setup_compose_then_render_without_manual_file_edits(tmp_path: Path) -> None:
    run_cli(tmp_path, "setup", "--backend", "compose", "--profile", "qwen2-5-7b-instruct-turbo-default")
    cfg = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert cfg["backend"] == "compose"
    assert cfg["active_profile"] == "qwen2-5-7b-instruct-turbo-default"
    run_cli(tmp_path, "render", "--simulate-hardware", "1x96")
    assert (tmp_path / "generated" / "docker-compose.yml").exists()


def test_setup_kubeai_then_render_without_manual_file_edits(tmp_path: Path) -> None:
    run_cli(
        tmp_path,
        "setup",
        "--backend",
        "kubeai",
        "--profile",
        "qwen2-72b-instruct-tp2-balanced",
        "--namespace",
        "demo-llm",
        "--ingress",
        "--ingress-host",
        "llm.example.test",
    )
    cfg = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert cfg["backend"] == "kubeai"
    assert cfg["cluster"]["namespace"] == "demo-llm"
    assert cfg["cluster"]["ingress"]["enabled"] is True
    run_cli(tmp_path, "render", "--simulate-hardware", "2x96")
    assert (tmp_path / "generated" / "kubeai" / "models.yaml").exists()
    assert (tmp_path / "generated" / "kubeai" / "ingress.yaml").exists()


def test_render_overrides_backend_and_profile_without_persisting_config(tmp_path: Path) -> None:
    run_cli(tmp_path, "setup", "--backend", "compose", "--profile", "gpt-oss-20b-chat")
    run_cli(
        tmp_path,
        "render",
        "--backend",
        "kubeai",
        "--profile",
        "qwen2-72b-instruct-tp2-balanced",
        "--namespace",
        "override-ns",
        "--simulate-hardware",
        "2x96",
    )
    cfg = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert cfg["backend"] == "compose"
    assert cfg["active_profile"] == "gpt-oss-20b-chat"
    models = list(yaml.safe_load_all((tmp_path / "generated" / "kubeai" / "models.yaml").read_text()))
    namespace_doc = yaml.safe_load((tmp_path / "generated" / "kubeai" / "namespace.yaml").read_text())
    assert models[0]["metadata"]["name"] == "qwen2-72b-instruct-tp2-balanced"
    assert namespace_doc["metadata"]["name"] == "override-ns"


def test_setup_supports_environment_fallbacks(tmp_path: Path) -> None:
    run_cli(
        tmp_path,
        "setup",
        env={
            "VLLM_SERVICE_BACKEND": "kubeai",
            "VLLM_SERVICE_PROFILE": "gpt-oss-20b-chat",
            "VLLM_SERVICE_NAMESPACE": "env-ns",
            "VLLM_SERVICE_INGRESS_ENABLED": "true",
            "VLLM_SERVICE_INGRESS_HOST": "env.example.test",
        },
    )
    cfg = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert cfg["backend"] == "kubeai"
    assert cfg["active_profile"] == "gpt-oss-20b-chat"
    assert cfg["cluster"]["namespace"] == "env-ns"
    assert cfg["cluster"]["ingress"]["enabled"] is True
    assert cfg["cluster"]["ingress"]["host"] == "env.example.test"
