#!/usr/bin/env bash
# Lifecycle for the training/serving GPU VM (single L4, spot).
#
#   scripts/gcp/vm.sh create   # provision spot g2-standard-8 (1x L4)
#   scripts/gcp/vm.sh sync     # push code + sft data to the VM
#   scripts/gcp/vm.sh setup    # install uv + training deps on the VM
#   scripts/gcp/vm.sh setup-serve  # install the vLLM + gateway deps
#   scripts/gcp/vm.sh train    # launch training inside tmux
#   scripts/gcp/vm.sh status   # tail training log
#   scripts/gcp/vm.sh fetch    # pull the trained adapter back
#   scripts/gcp/vm.sh serve    # start vLLM (quantized artifact) in tmux
#   scripts/gcp/vm.sh gateway  # start the FastAPI gateway in tmux
#   scripts/gcp/vm.sh bench    # run the load-test sweep on the VM
#   scripts/gcp/vm.sh results  # pull benchmark json back into runs/
#   scripts/gcp/vm.sh tunnel   # self-reconnecting port-forward for local use
#   scripts/gcp/vm.sh logs     # tail the serving logs
#   scripts/gcp/vm.sh start    # boot a stopped VM (keeps disk state)
#   scripts/gcp/vm.sh stop     # stop compute billing, keep the disk
#   scripts/gcp/vm.sh ssh      # interactive shell
#   scripts/gcp/vm.sh delete   # tear down (stops billing)
set -euo pipefail

# The GCP project id and the VM's ~/text2sql-serving working directory predate
# the repo's rename to sqlforge; the project id is immutable, and the remote
# path stays until no run is active on the VM.
PROJECT=text2sql-serving-sh
ZONE=us-central1-a
VM=t2s-gpu
MACHINE=g2-standard-8   # includes 1x NVIDIA L4
ARTIFACT=out/merged-gptq            # the M2 serving artifact (GPTQ W4A16)
SERVED_NAME=sqlforge-3b             # model id clients and the gateway use
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

gc() { gcloud --project "$PROJECT" "$@"; }
vssh() { gc compute ssh "$VM" --zone "$ZONE" --command "$1"; }

case "${1:?subcommand required}" in
  create)
    gc compute instances create "$VM" \
      --zone "$ZONE" \
      --machine-type "$MACHINE" \
      --provisioning-model=SPOT \
      --instance-termination-action=STOP \
      --image-family=pytorch-2-9-cu129-ubuntu-2404-nvidia-580 \
      --image-project=deeplearning-platform-release \
      --boot-disk-size=150GB \
      --boot-disk-type=pd-balanced \
      --maintenance-policy=TERMINATE \
      --metadata=install-nvidia-driver=True
    echo "VM created. Wait ~2 min for driver install, then: $0 sync && $0 setup"
    ;;
  sync)
    tar -C "$REPO_DIR" -czf /tmp/t2s-sync.tgz \
      pyproject.toml uv.lock .python-version src scripts data/sft data/serving
    gc compute scp /tmp/t2s-sync.tgz "$VM":/tmp/ --zone "$ZONE"
    vssh "mkdir -p ~/text2sql-serving && tar -xzf /tmp/t2s-sync.tgz -C ~/text2sql-serving"
    ;;
  setup)
    vssh "curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; \
          cd ~/text2sql-serving && ~/.local/bin/uv sync --group train && nvidia-smi -L"
    ;;
  setup-serve)
    vssh "cd ~/text2sql-serving && ~/.local/bin/uv sync --group vllm --group serve"
    ;;
  train)
    vssh "cd ~/text2sql-serving && tmux new-session -d -s train \
          '~/.local/bin/uv run --group train scripts/train.py 2>&1 | tee train.log'"
    echo "Training started in tmux session 'train'. Follow with: $0 status"
    ;;
  status)
    vssh "tail -20 ~/text2sql-serving/train.log 2>/dev/null || echo 'no log yet'"
    ;;
  fetch)
    mkdir -p "$REPO_DIR/models"
    gc compute scp --recurse "$VM":~/text2sql-serving/out/qlora-r16 \
      "$REPO_DIR/models/" --zone "$ZONE"
    echo "adapter in models/qlora-r16"
    ;;
  serve)
    # vLLM holds the GPU for as long as it runs: nothing else (quantization,
    # a second server) can share the card, so this is the only GPU process.
    vssh "cd ~/text2sql-serving && tmux kill-session -t serve 2>/dev/null; \
          tmux new-session -d -s serve \
          '~/.local/bin/uv run --group vllm vllm serve $ARTIFACT \
             --served-model-name $SERVED_NAME \
             --max-model-len 4096 \
             --gpu-memory-utilization 0.90 \
             --port 8000 2>&1 | tee serve.log'"
    echo "vLLM starting (~2 min to load weights). Readiness: $0 logs"
    ;;
  gateway)
    vssh "cd ~/text2sql-serving && tmux kill-session -t gateway 2>/dev/null; \
          tmux new-session -d -s gateway \
          '~/.local/bin/uv run --group serve scripts/serve_gateway.py \
             --port 8080 --model $SERVED_NAME \
             --schema-cache data/serving/dev.schemas.json 2>&1 | tee gateway.log'"
    vssh "curl -sf --retry 30 --retry-delay 2 --retry-all-errors \
            http://localhost:8080/readyz && echo"
    ;;
  bench)
    # The sweep runs on the VM against localhost: driving it from a laptop
    # would measure the SSH tunnel, not the service.
    shift
    vssh "cd ~/text2sql-serving && ~/.local/bin/uv run scripts/loadtest.py ${*:-} 2>&1 | tail -40"
    ;;
  results)
    mkdir -p "$REPO_DIR/runs"
    gc compute scp "$VM":~/text2sql-serving/runs/'loadtest-*.json' \
      "$REPO_DIR/runs/" --zone "$ZONE"
    ;;
  tunnel)
    # gcloud's ssh dies with the VM's network hiccups; reconnect instead of
    # failing a long eval run that is talking to :8000 through this tunnel.
    echo "forwarding localhost:8000 (vLLM) and localhost:8080 (gateway); ctrl-c to stop"
    while true; do
      gc compute ssh "$VM" --zone "$ZONE" -- -N \
        -L 8000:localhost:8000 -L 8080:localhost:8080 || true
      echo "tunnel dropped; reconnecting in 5s" >&2
      sleep 5
    done
    ;;
  logs)
    vssh "tail -15 ~/text2sql-serving/serve.log 2>/dev/null | tail -5; \
          echo '--- gateway'; tail -5 ~/text2sql-serving/gateway.log 2>/dev/null"
    ;;
  start)
    gc compute instances start "$VM" --zone "$ZONE"
    ;;
  stop)
    gc compute instances stop "$VM" --zone "$ZONE"
    ;;
  ssh)
    gc compute ssh "$VM" --zone "$ZONE"
    ;;
  delete)
    gc compute instances delete "$VM" --zone "$ZONE" --quiet
    ;;
  *)
    echo "unknown subcommand: $1" >&2; exit 1
    ;;
esac
