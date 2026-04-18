from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..config import resource_profiles_to_kubeai_values
from ..profile_runtime import vllm_args


def _resource_profile_values(plan: dict[str, Any]) -> dict[str, Any]:
    return resource_profiles_to_kubeai_values(plan.get("deployment", {}).get("resource_profiles", {}))


def _model_doc(service: dict[str, Any]) -> dict[str, Any]:
    doc = {
        "apiVersion": "kubeai.org/v1",
        "kind": "Model",
        "metadata": {
            "name": service["kubernetes_name"],
            "annotations": {
                "vllm-service/profile-name": service["profile_name"],
                "vllm-service/public-name": service["profile_public_name"],
                "vllm-service/logical-model-name": service["logical_model_name"],
                "vllm-service/protocol-mode": service["protocol_mode"],
            },
        },
        "spec": {
            "features": service.get("features", ["TextGeneration"]),
            "url": service["model_url"],
            "engine": service.get("engine", "VLLM"),
            "resourceProfile": service["resource_profile"],
            "minReplicas": int(service.get("min_replicas", 0)),
            "maxReplicas": int(service.get("max_replicas", 1)),
            "args": vllm_args(service),
        },
    }
    if service.get("priority_class_name"):
        doc["spec"]["priorityClassName"] = service["priority_class_name"]
    return doc


def render_kubeai_artifacts(root: Path, lock_data: dict) -> None:
    deployment = lock_data.get("deployment", {})
    cluster = deployment.get("cluster", {})
    namespace = cluster.get("namespace", "kubeai")
    generated = root / "generated" / "kubeai"
    generated.mkdir(parents=True, exist_ok=True)

    namespace_doc = {
        "apiVersion": "v1",
        "kind": "Namespace",
        "metadata": {"name": namespace},
    }
    (generated / "namespace.yaml").write_text(yaml.safe_dump(namespace_doc, sort_keys=False), encoding="utf-8")

    values_doc = _resource_profile_values(lock_data)
    (generated / "kubeai-values.yaml").write_text(yaml.safe_dump(values_doc, sort_keys=False), encoding="utf-8")

    model_docs = [_model_doc(service) for service in deployment.get("services", [])]
    model_text = "---\n".join(yaml.safe_dump(doc, sort_keys=False) for doc in model_docs)
    (generated / "models.yaml").write_text(model_text, encoding="utf-8")

    ingress = cluster.get("ingress", {}) or {}
    ingress_path = generated / "ingress.yaml"
    if ingress.get("enabled"):
        path_prefix = ingress.get("path_prefix", "/") or "/"
        ingress_doc: dict[str, Any] = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {
                "name": cluster.get("service_name", "kubeai"),
                "namespace": namespace,
            },
            "spec": {
                "ingressClassName": ingress.get("class_name", "traefik"),
                "rules": [
                    {
                        "http": {
                            "paths": [
                                {
                                    "path": path_prefix,
                                    "pathType": "Prefix",
                                    "backend": {
                                        "service": {
                                            "name": cluster.get("service_name", "kubeai"),
                                            "port": {"number": 80},
                                        }
                                    },
                                }
                            ]
                        }
                    }
                ],
            },
        }
        if ingress.get("host"):
            ingress_doc["spec"]["rules"][0]["host"] = ingress["host"]
        if ingress.get("tls_secret_name") and ingress.get("host"):
            ingress_doc["spec"]["tls"] = [{"hosts": [ingress["host"]], "secretName": ingress["tls_secret_name"]}]
        ingress_path.write_text(yaml.safe_dump(ingress_doc, sort_keys=False), encoding="utf-8")
    elif ingress_path.exists():
        ingress_path.unlink()

    readme = f"""# Generated KubeAI artifacts

Namespace: `{namespace}`
Release: `{cluster.get('kubeai_release_name', 'kubeai')}`
Chart: `{cluster.get('kubeai_chart', 'kubeai/kubeai')}`

Files:
- `namespace.yaml`: namespace to apply before the chart and models
- `kubeai-values.yaml`: custom resource profiles for the KubeAI chart
- `models.yaml`: KubeAI `Model` objects derived intentionally from the selected serving profile(s)
- `ingress.yaml`: optional ingress for one stable hostname

Typical flow:

```bash
kubectl apply -f generated/kubeai/namespace.yaml
helm repo add kubeai https://www.kubeai.org --force-update
helm repo update
helm upgrade --install {cluster.get('kubeai_release_name', 'kubeai')} {cluster.get('kubeai_chart', 'kubeai/kubeai')} \
  -n {namespace} --create-namespace \
  -f generated/kubeai/kubeai-values.yaml \
  --wait
kubectl apply -f generated/kubeai/models.yaml
```
"""
    (generated / "README.md").write_text(readme, encoding="utf-8")
