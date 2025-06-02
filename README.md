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

python infer.py \
  --ckpt ./checkpoints/base/tacotron2_0020.pth \
  --text "Este es un ejemplo de inferencia con un modelo entrenado." \
  --out ejemplo.wav