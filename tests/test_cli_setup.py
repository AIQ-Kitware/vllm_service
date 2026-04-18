from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

from vllm_service import cli as cli_mod


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


def write_kubeai_values_file(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "resourceProfiles": {
                    "gpu-single-default": {
                        "nodeSelector": {
                            "nvidia.com/gpu.product": "NVIDIA_TEST_GPU",
                            "nvidia.com/gpu.memory": "81920",
                        }
                    },
                    "gpu-tp2-balanced": {
                        "nodeSelector": {
                            "nvidia.com/gpu.product": "NVIDIA_TEST_GPU",
                            "nvidia.com/gpu.memory": "81920",
                        }
                    },
                    "gpu-tp2-maxctx": {
                        "nodeSelector": {
                            "nvidia.com/gpu.product": "NVIDIA_TEST_GPU",
                            "nvidia.com/gpu.memory": "81920",
                        }
                    },
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
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


def test_setup_kubeai_with_resource_profiles_file_syncs_canonical_values(tmp_path: Path) -> None:
    source = tmp_path / "values-kubeai-local-gpu.yaml"
    write_kubeai_values_file(source)
    run_cli(
        tmp_path,
        "setup",
        "--backend",
        "kubeai",
        "--profile",
        "qwen2-5-7b-instruct-turbo-default",
        "--namespace",
        "kubeai",
        "--resource-profiles-file",
        str(source),
    )
    values_doc = yaml.safe_load((tmp_path / "generated" / "kubeai" / "kubeai-values.yaml").read_text())
    assert "gpu-single-default" in values_doc["resourceProfiles"]


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


def test_kubeai_sync_resource_profiles_then_validate_without_config_duplication(tmp_path: Path) -> None:
    source = tmp_path / "values-kubeai-local-gpu.yaml"
    write_kubeai_values_file(source)
    run_cli(
        tmp_path,
        "setup",
        "--backend",
        "kubeai",
        "--profile",
        "qwen2-5-7b-instruct-turbo-default",
        "--namespace",
        "kubeai",
    )
    cfg = yaml.safe_load((tmp_path / "config.yaml").read_text())
    cfg.pop("resource_profiles", None)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    run_cli(tmp_path, "kubeai-sync-resource-profiles", "--from-file", str(source))
    run_cli(tmp_path, "validate", "--simulate-hardware", "1x96")


def test_switch_apply_persists_only_active_profile_and_uses_transient_overrides(
    tmp_path: Path, monkeypatch
) -> None:
    run_cli(
        tmp_path,
        "setup",
        "--backend",
        "compose",
        "--profile",
        "gpt-oss-20b-chat",
        "--compose-cmd",
        "docker compose",
    )
    monkeypatch.chdir(tmp_path)
    observed: dict[str, object] = {}

    def fake_deploy_rendered_artifacts(root: Path, deployment: dict) -> None:
        observed["root"] = root
        observed["backend"] = deployment["backend"]
        observed["namespace"] = deployment["cluster"]["namespace"]
        observed["profile"] = deployment["serving_profile"]["public_name"]

    monkeypatch.setattr(cli_mod, "deploy_rendered_artifacts", fake_deploy_rendered_artifacts)
    args = argparse.Namespace(
        profile="qwen2-5-7b-instruct-turbo-default",
        backend="kubeai",
        compose_cmd="podman compose",
        litellm_port=None,
        open_webui_port=None,
        postgres_port=None,
        namespace="override-ns",
        ingress_host=None,
        ingress_enabled=None,
        apply=True,
        allow_unsupported=False,
        simulate_hardware="1x96",
    )
    cli_mod.cmd_switch(args)

    cfg = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert cfg["active_profile"] == "qwen2-5-7b-instruct-turbo-default"
    assert cfg["backend"] == "compose"
    assert cfg["runtime"]["compose_cmd"] == "docker compose"
    assert cfg["cluster"]["namespace"] != "override-ns"

    assert observed["backend"] == "kubeai"
    assert observed["namespace"] == "override-ns"
    assert observed["profile"] == "qwen2-5-7b-instruct-turbo-default"


def test_kubeai_status_namespace_error_is_actionable(tmp_path: Path, monkeypatch) -> None:
    run_cli(tmp_path, "setup", "--backend", "kubeai", "--profile", "qwen2-5-7b-instruct-turbo-default", "--namespace", "default")
    monkeypatch.chdir(tmp_path)

    def fake_status(namespace: str) -> None:
        raise cli_mod.CommandError(f"Command failed in namespace {namespace}")

    monkeypatch.setattr(cli_mod, "kubeai_print_status", fake_status)
    args = argparse.Namespace(
        backend=None,
        compose_cmd=None,
        litellm_port=None,
        open_webui_port=None,
        postgres_port=None,
        namespace=None,
        ingress_host=None,
        ingress_enabled=None,
    )
    try:
        cli_mod.cmd_status(args)
    except SystemExit as ex:
        text = str(ex)
    else:
        raise AssertionError("expected cmd_status to raise SystemExit")
    assert "namespace 'default'" in text
    assert "setup --backend kubeai --namespace default" in text
