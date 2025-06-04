#!/usr/bin/env bash

# Crear y activar entorno virtual (opcional)
python3 -m venv venv
source venv/bin/activate

# 1) Bájate el tarball oficial de LJSpeech
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2

# 2) Descomprímelo
tar xjf LJSpeech-1.1.tar.bz2

# 3) Mueve/renombra la carpeta resultante a data/LJSpeech
mkdir -p data/LJSpeech
mv LJSpeech-1.1/* data/LJSpeech

# 4) (Opcional) Borra el tarball si ya no lo necesitas
rm LJSpeech-1.1.tar.bz2


# Actualizar pip
pip install --upgrade pip

# Instalar dependencias
pip install torch torchaudio matplotlib numpy librosa scipy tqdm optuna transformers torchvision

git clone https://github.com/NVIDIA/DeepLearningExamples.git
# cd DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2
mv model.py DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2/tacotron2/
export PYTHONPATH="$PWD/DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2:$PYTHONPATH"

echo "Instalación completa. Puedes ejecutar 'python tts_lora.py --help' para ver las opciones."

python tts_embeddings.py train \
  --data_dir data/LJSpeech \
  --hparams_path hparams.json \
  --epochs 100 \
  --batch_size 32 \
  --learning_rate 1e-3 \
  --num_workers 4 \
  --checkpoints_dir checkpoints \
  --spk_emb_size 192

python tts_embeddings.py infer \
  --hparams_path hparams.json \
  --checkpoint_path checkpoints/best_tts_zero_shot.pth \
  --reference_wav data/my_voice/wavs/Voz1.wav \
  --text "Hello, this is a zero-shot voice cloning test." \
  --output_dir samples_cloned \
  --spk_emb_size 192 \
  --sample_rate 22050
