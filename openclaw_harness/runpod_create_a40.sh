#!/usr/bin/env bash
set -euo pipefail

# Creates the recommended VisualPRM/OpenClaw evaluation pod.
# Requires RunPod auth first:
#   runpodctl doctor
#
# This command starts billing if it succeeds.

runpodctl pod create \
  --template-id runpod-torch-v280 \
  --gpu-id "NVIDIA A40" \
  --cloud-type SECURE \
  --name visualprm-openclaw-a40 \
  --gpu-count 1 \
  --container-disk-in-gb 80 \
  --volume-in-gb 100 \
  --volume-mount-path /workspace \
  --ports "22/tcp,8000/http,8764/http,11434/http" \
  --ssh
