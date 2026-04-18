#!/usr/bin/env bash
set -euo pipefail

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required" >&2
  exit 1
fi

if ! command -v sudo >/dev/null 2>&1; then
  echo "sudo is required" >&2
  exit 1
fi

if systemctl is-active --quiet k3s; then
  echo "k3s is already active"
else
  curl -sfL https://get.k3s.io | sh -
fi

if ! command -v helm >/dev/null 2>&1; then
  echo "Installing helm"
  curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
fi

mkdir -p "$HOME/.kube"
sudo cp /etc/rancher/k3s/k3s.yaml "$HOME/.kube/config"
sudo chown "$(id -u):$(id -g)" "$HOME/.kube/config"
chmod 600 "$HOME/.kube/config"

export KUBECONFIG="$HOME/.kube/config"
kubectl wait --for=condition=Ready node --all --timeout=180s

echo "K3s installed and ready"
echo "Kubeconfig: $HOME/.kube/config"
echo "Node token: /var/lib/rancher/k3s/server/node-token"
