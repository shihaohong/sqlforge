#!/usr/bin/env bash
# Run the load test from inside the cluster, against the gateway Service.
#
#   deploy/bench_in_cluster.sh [loadtest args...]
#   deploy/bench_in_cluster.sh --concurrency 1,8,64 --duration 20
#
# The driver runs as a pod on the system pool, so the measurement covers the
# full in-cluster path (Service -> gateway -> vLLM) without a laptop's network
# in the middle. The gateway image already carries loadtest.py and the replay
# questions, so there is nothing to build for a benchmark run.
set -euo pipefail

PROJECT=${PROJECT:-text2sql-serving-sh}
NAMESPACE=${NAMESPACE:-sqlforge}
IMAGE=${IMAGE:-us-central1-docker.pkg.dev/$PROJECT/sqlforge/gateway:latest}
POD=${POD:-sqlforge-bench}
DEPLOY_DIR="$(cd "$(dirname "$0")" && pwd)"

# kubectl needs credentials; the stack's kubeconfig output carries a token, so
# no gke-gcloud-auth-plugin is required here either.
KUBECONFIG_FILE=${KUBECONFIG_FILE:-/tmp/sqlforge.kubeconfig}
if [[ ! -s "$KUBECONFIG_FILE" ]]; then
  (cd "$DEPLOY_DIR" && pulumi stack output kubeconfig --show-secrets) > "$KUBECONFIG_FILE"
  chmod 600 "$KUBECONFIG_FILE"
fi
export KUBECONFIG="$KUBECONFIG_FILE"

kubectl -n "$NAMESPACE" delete pod "$POD" --ignore-not-found >/dev/null

# --restart=Never plus --rm: one run, logs streamed to this terminal, no
# leftover pod to clean up.
kubectl -n "$NAMESPACE" run "$POD" \
  --image="$IMAGE" \
  --restart=Never \
  --rm -i \
  --overrides='{"spec":{"nodeSelector":{"workload":"system"}}}' \
  --command -- python scripts/loadtest.py \
    --target gateway \
    --base-url "http://gateway.$NAMESPACE.svc.cluster.local:8080" \
    "$@"
