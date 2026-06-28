export HF_HOME=/mnt/volume_d2wey28/hf-cache
export HF_DATASETS_CACHE=/mnt/volume_d2wey28/hf-cache/datasets
export TORCH_HOME=/mnt/volume_d2wey28/torch-cache
export HF_TOKEN=$(cat /mnt/volume_d2wey28/hf-cache/token 2>/dev/null)
export HF_HUB_ENABLE_HF_TRANSFER=1
source /mnt/volume_d2wey28/projects/voxcpm-ghana/.venv/bin/activate
export HF_HUB_DISABLE_XET=1
