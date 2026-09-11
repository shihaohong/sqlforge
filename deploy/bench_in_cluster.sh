#!/usr/bin/env bash
# Run the load test from inside the cluster, against the gateway Service.
#
#   deploy/bench_in_cluster.sh [loadtest args...]
#   deploy/bench_in_cluster.sh --concurrency 1,8,64 --duration 30
#
# The driver runs as a pod, so the measurement covers the full in-cluster path
# (Service -> gateway -> vLLM) with no laptop network in the middle. The
# gateway image already carries loadtest.py and the replay questions, so a
# benchmark run builds nothing.
#
# The driver requests 2 CPUs and refuses to share a node with the gateway
# (anti-affinity), so it never competes for cores with the tier it measures:
# co-scheduled on one small node, the driver and the gateways starve each
# other and throughput collapses for reasons that say nothing about the
# service. This is the same mistake as benchmarking through an SSH tunnel,
# one layer in.
set -euo pipefail

PROJECT=${PROJECT:-text2sql-serving-sh}
NAMESPACE=${NAMESPACE:-sqlforge}
IMAGE=${IMAGE:-us-central1-docker.pkg.dev/$PROJECT/sqlforge/gateway:latest}
POD=${POD:-sqlforge-bench}
RUN_NAME=${RUN_NAME:-loadtest-gke}
DEPLOY_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNS_DIR="$(cd "$DEPLOY_DIR/.." && pwd)/runs"

# kubectl needs credentials; the stack's kubeconfig output carries a token, so
# no gke-gcloud-auth-plugin is required here either.
KUBECONFIG_FILE=${KUBECONFIG_FILE:-/tmp/sqlforge.kubeconfig}
if [[ ! -s "$KUBECONFIG_FILE" ]]; then
  (cd "$DEPLOY_DIR" && pulumi stack output kubeconfig --show-secrets) > "$KUBECONFIG_FILE"
  chmod 600 "$KUBECONFIG_FILE"
fi
export KUBECONFIG="$KUBECONFIG_FILE"

# Render the extra arguments as a YAML list.
args_yaml=""
for arg in "$@"; do
  args_yaml+="
    - \"$arg\""
done

kubectl -n "$NAMESPACE" delete pod "$POD" --ignore-not-found >/dev/null

kubectl -n "$NAMESPACE" apply -f - >/dev/null <<YAML
apiVersion: v1
kind: Pod
metadata:
  name: $POD
  namespace: $NAMESPACE
  labels:
    app: sqlforge-bench
spec:
  restartPolicy: Never
  enableServiceLinks: false
  nodeSelector:
    workload: system
  affinity:
    podAntiAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchExpressions:
          - key: app
            operator: In
            values: ["gateway"]
        topologyKey: kubernetes.io/hostname
  containers:
  - name: bench
    image: $IMAGE
    command: ["python", "scripts/loadtest.py"]
    args:
    - "--target"
    - "gateway"
    - "--base-url"
    - "http://gateway.$NAMESPACE.svc.cluster.local:8080"
    - "--out-dir"
    - "-"$args_yaml
    resources:
      requests:
        cpu: "2"
        memory: 512Mi
      limits:
        cpu: "3"
        memory: 1Gi
YAML

echo "benchmark pod running; streaming logs (this takes a few minutes)"
kubectl -n "$NAMESPACE" wait --for=condition=Ready "pod/$POD" --timeout=5m >/dev/null
log=$(mktemp)
kubectl -n "$NAMESPACE" logs -f "$POD" | tee "$log"

# Lift the result json out of the log and keep it with every other run.
mkdir -p "$RUNS_DIR"
python3 - "$log" "$RUNS_DIR/$RUN_NAME.json" <<'PY'
import sys
from pathlib import Path

BEGIN = "--- sqlforge-loadtest-json-begin ---"
END = "--- sqlforge-loadtest-json-end ---"

text = Path(sys.argv[1]).read_text()
if BEGIN not in text:
    sys.exit("no result json in the benchmark output")
body = text.split(BEGIN, 1)[1].split(END, 1)[0].strip()
Path(sys.argv[2]).write_text(body + "\n")
print(f"saved: {sys.argv[2]}")
PY

kubectl -n "$NAMESPACE" delete pod "$POD" --ignore-not-found >/dev/null
