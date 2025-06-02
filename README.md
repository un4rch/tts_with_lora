python -m venv .venv
source ./venv/bin/activate

torchrun --nproc_per_node=2 train_base.py   --dataset ./data/LJSpeech   --out ./checkpoints/base   --epochs 20   --bs 64   --lr 1e-3   --download

python finetune_lora.py \
  --base_ckpt checkpoints/base/tacotron2_0020.pth \
  --data ./data/my_voice \
  --out ./loras/lora_myvoice \
  --epochs 10

python infer.py \
  --base_ckpt ./checkpoints/base/tacotron2_0020.pth \
  --lora_dir ./loras/lora_myvoice \
  --text "Hola, esta es mi voz personalizada." \
  --out mi_voz.wav

torchrun --nproc_per_node=2 infer.py \
  --ckpt ./checkpoints/base/tacotron2_0020.pth \
  --text "Este es un ejemplo de inferencia con un modelo entrenado." \
  --out ejemplo.wav

python infer_old.py \
  --ckpt ./checkpoints/base/tacotron2_0020.pth \
  --text "This is a text to audio speech test" \
  --out ejemplo.wav



python finetune_lora.py \
    --dataset ./data/my_voice \
    --out ./loras/lora_myvoice \
    --ckpt checkpoints/base/tacotron2_0020.pth \
    --epochs 150 \
    --bs 20 \
    --lr 3e-4 \
    --lora_rank 4 \
    --lora_alpha 16 \
    --sample_text "Hello, this is my voice adapted with LoRA"

python infer.py \
    --base_ckpt checkpoints/base/tacotron2_0020.pth \
    --lora_ckpt loras/lora_myvoice/lora_epoch_150.pth \
    --lora_rank 4 \
    --lora_alpha 16 \
    --text "Hello, this is my voice adapted with LoRA." \
    --out samples/myvoice_final.wav

