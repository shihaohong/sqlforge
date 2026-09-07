#!/usr/bin/env bash
# Lifecycle for the training/serving GPU VM (single L4, spot).
#
#   scripts/gcp/vm.sh create   # provision spot g2-standard-8 (1x L4)
#   scripts/gcp/vm.sh sync     # push code + sft data to the VM
#   scripts/gcp/vm.sh setup    # install uv + deps on the VM
#   scripts/gcp/vm.sh train    # launch training inside tmux
#   scripts/gcp/vm.sh status   # tail training log
#   scripts/gcp/vm.sh fetch    # pull the trained adapter back
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
      pyproject.toml uv.lock .python-version src scripts data/sft
    gc compute scp /tmp/t2s-sync.tgz "$VM":/tmp/ --zone "$ZONE"
    vssh "mkdir -p ~/text2sql-serving && tar -xzf /tmp/t2s-sync.tgz -C ~/text2sql-serving"
    ;;
  setup)
    vssh "curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; \
          cd ~/text2sql-serving && ~/.local/bin/uv sync --group train && nvidia-smi -L"
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
