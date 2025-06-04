#!/usr/bin/env python3
import os
import json
import argparse
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torch.nn import functional as F
from tqdm import tqdm

from speechbrain.pretrained import EncoderClassifier  # Para ECAPA-TDNN


"""
tts_lora/
├── DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2/tacotron2/model.py  # versión modificada con spk_proj
├── tts_embeddings.py
├── hparams.json      # JSON con tus hiperparámetros para Tacotron2
├── checkpoints/      # carpeta vacía o con checkpoints previos
└── data/
    └── LJSpeech/     # dataset de LJSpeech con metadata.csv y wavs/
    └── my_voice/     # tu dataset personal (opcional, para inferencia)
        └── wavs/
            └── Voz1.wav

Uso:

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
"""

# --------------------------------------------------------------------------------
# 0) UTILIDADES: cargar hparams y extraer speaker embedding
# --------------------------------------------------------------------------------

def load_hparams(hparams_path: str) -> SimpleNamespace:
    """
    Carga hiperparámetros desde un JSON. Debe contener al menos los campos de Tacotron2:
      - mask_padding
      - n_mel_channels
      - symbols_embedding_dim
      - encoder_kernel_size
      - encoder_n_convolutions
      - encoder_embedding_dim
      - attention_rnn_dim
      - attention_dim
      - attention_location_n_filters
      - attention_location_kernel_size
      - n_frames_per_step
      - decoder_rnn_dim
      - prenet_dim
      - max_decoder_steps
      - gate_threshold
      - p_attention_dropout
      - p_decoder_dropout
      - postnet_embedding_dim
      - postnet_kernel_size
      - postnet_n_convolutions
      - decoder_no_early_stopping
      - symbols  (lista de strings)
    Adicionalmente, puede incluir (opcional):
      - sample_rate (predeterminado: 22050)
      - n_fft       (predeterminado: 1024)
      - hop_length  (predeterminado: 256)
      - win_length  (predeterminado: 1024)
    """
    with open(hparams_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return SimpleNamespace(**data)


def get_spk_embedding(wav_path: str, device: torch.device) -> torch.Tensor:
    """
    Extrae el speaker embedding ECAPA-TDNN desde un archivo WAV.
    - Re-muestrea a 16 kHz mono.
    - Retorna un Tensor [192] en CPU.
    """
    # ECAPA se mantiene en CPU para evitar mismatch de tipos
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb"
    )
    ecapa.eval()

    wav, sr = torchaudio.load(wav_path)  # [1, T] o [2, T]
    if wav.dim() == 2 and wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=16000)
    wav = wav.squeeze(0).unsqueeze(0)  # [1, T] en CPU

    with torch.no_grad():
        emb = ecapa.encode_batch(wav)  # [1, 192]
    emb = emb.squeeze(0).cpu()  # [192]
    return emb


# --------------------------------------------------------------------------------
# 1) CLASE DE DATASET PARA LJSPEECH
# --------------------------------------------------------------------------------

class LJSpeechDataset(Dataset):
    """
    Dataset para LJSpeech (u otro con metadata.csv en formato fileid|transcript|normalized_transcript).
    Archivos .wav en subcarpeta 'wavs/'. Retorna:
      - text (string)
      - mel (Tensor [n_mels, T_frames])
      - spk_emb (Tensor [192]) de ECAPA-TDNN
    """
    def __init__(self, data_dir, hparams, ecapa_model, device):
        super().__init__()
        self.data_dir = data_dir
        self.wav_dir = os.path.join(data_dir, "wavs")
        metadata_path = os.path.join(data_dir, "metadata.csv")
        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(f"No se encuentra {metadata_path}")
        lines = open(metadata_path, "r", encoding="utf-8").read().strip().split("\n")
        self.entries = []
        for line in lines:
            parts = line.strip().split("|")
            if len(parts) != 3:
                continue
            fileid, _, norm = parts
            wav_path = os.path.join(self.wav_dir, f"{fileid}.wav")
            if os.path.isfile(wav_path):
                self.entries.append({"wav_path": wav_path, "text": norm.lower()})

        # Parámetros de audio/mel: usar defaults si no están en hparams
        self.sample_rate = getattr(hparams, "sample_rate", 22050)
        self.n_mels = hparams.n_mel_channels
        self.n_fft = getattr(hparams, "n_fft", 1024)
        self.hop_length = getattr(hparams, "hop_length", 256)
        self.win_length = getattr(hparams, "win_length", 1024)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
        )
        self.ecapa = ecapa_model  # ECAPA en CPU
        self.device = device

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        wav_path = entry["wav_path"]
        text = entry["text"]

        # 1) Cargar audio y calcular mel-spectrogram (en CPU)
        wav, sr = torchaudio.load(wav_path)  # [1, T] o [2, T]
        if wav.dim() == 2 and wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=self.sample_rate)
        mel_spec = self.mel_transform(wav)  # [1, n_mels, T_frames]
        mel_db = torchaudio.functional.amplitude_to_DB(
            mel_spec, multiplier=10.0, amin=1e-10, db_multiplier=0.0
        ).squeeze(0)  # [n_mels, T_frames]

        # 2) Obtener embedding de hablante (ECAPA en CPU)
        wav_ecapa, sr2 = torchaudio.load(wav_path)
        if wav_ecapa.dim() == 2 and wav_ecapa.shape[0] > 1:
            wav_ecapa = torch.mean(wav_ecapa, dim=0, keepdim=True)
        if sr2 != 16000:
            wav_ecapa = torchaudio.functional.resample(wav_ecapa, orig_freq=sr2, new_freq=16000)
        wav_ecapa = wav_ecapa.squeeze(0).unsqueeze(0)  # [1, T] en CPU
        with torch.no_grad():
            spk_emb = self.ecapa.encode_batch(wav_ecapa)  # [1, 192]
        spk_emb = spk_emb.squeeze(0).cpu()  # [192]

        return {"text": text, "mel": mel_db, "spk_emb": spk_emb}


def collate_fn(batch):
    """
    Recibe lista de dicts {"text": str, "mel": Tensor[n_mels, T], "spk_emb": Tensor[192]}.
    Retorna:
      - texts: [str]
      - padded_mels: Tensor [B, n_mels, T_max]
      - mel_lengths: LongTensor [B]
      - spk_embs: Tensor [B, 192]
    """
    texts = [item["text"] for item in batch]
    mels = [item["mel"] for item in batch]
    spk_embs = [item["spk_emb"] for item in batch]

    mel_lengths = torch.LongTensor([mel.size(1) for mel in mels])
    max_mel_len = int(mel_lengths.max().item())
    n_mels = mels[0].size(0)
    padded_mels = torch.full((len(mels), n_mels, max_mel_len), fill_value=-100.0)
    for i, mel in enumerate(mels):
        T = mel.size(1)
        padded_mels[i, :, :T] = mel

    spk_embs = torch.stack(spk_embs, dim=0)  # [B, 192]

    return texts, padded_mels, mel_lengths, spk_embs


# --------------------------------------------------------------------------------
# 2) FUNCIONES DE ENTRENAMIENTO E INFERENCIA
# --------------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Iniciando entrenamiento en dispositivo: {device}")

    # 1) Cargar hiperparámetros
    hparams = load_hparams(args.hparams_path)

    # 2) Instanciar Tacotron2 modificado (sin pretrained)
    print("[INFO] Instanciando Tacotron2 modificado (sin pretrained)...")
    BASE_DIR = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "DeepLearningExamples",
            "PyTorch",
            "SpeechSynthesis",
            "Tacotron2",
            "tacotron2",
        )
    )
    import sys
    sys.path.append(BASE_DIR)
    from model import Tacotron2  # Versión modificada con spk_proj

    model = Tacotron2(
        mask_padding=hparams.mask_padding,
        n_mel_channels=hparams.n_mel_channels,
        n_symbols=len(hparams.symbols),
        symbols_embedding_dim=hparams.symbols_embedding_dim,
        encoder_kernel_size=hparams.encoder_kernel_size,
        encoder_n_convolutions=hparams.encoder_n_convolutions,
        encoder_embedding_dim=hparams.encoder_embedding_dim,
        attention_rnn_dim=hparams.attention_rnn_dim,
        attention_dim=hparams.attention_dim,
        attention_location_n_filters=hparams.attention_location_n_filters,
        attention_location_kernel_size=hparams.attention_location_kernel_size,
        n_frames_per_step=hparams.n_frames_per_step,
        decoder_rnn_dim=hparams.decoder_rnn_dim,
        prenet_dim=hparams.prenet_dim,
        max_decoder_steps=hparams.max_decoder_steps,
        gate_threshold=hparams.gate_threshold,
        p_attention_dropout=hparams.p_attention_dropout,
        p_decoder_dropout=hparams.p_decoder_dropout,
        postnet_embedding_dim=hparams.postnet_embedding_dim,
        postnet_kernel_size=hparams.postnet_kernel_size,
        postnet_n_convolutions=hparams.postnet_n_convolutions,
        decoder_no_early_stopping=hparams.decoder_no_early_stopping,
        spk_emb_size=args.spk_emb_size,
    )
    model = model.to(device)

    # 3) Cargar ECAPA-TDNN para extracción de embeddings en el dataset (ECAPA en CPU)
    print("[INFO] Cargando ECAPA-TDNN (SpeechBrain) para extracción de speaker embeddings...")
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb"
    )
    ecapa.eval()

    # 4) Crear dataset y dataloader
    train_dataset = LJSpeechDataset(
        data_dir=args.data_dir, hparams=hparams, ecapa_model=ecapa, device=device
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"[INFO] {len(train_dataset)} muestras de entrenamiento cargadas.")

    # 5) Configurar optimizer, scheduler y loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    criterion = nn.MSELoss()

    # 6) Cargar tts_utils para tokenizar texto
    print("[INFO] Cargando tts_utils de NVIDIA para tokenizar texto...")
    tts_utils = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub", "nvidia_tts_utils")

    # 7) Loop de entrenamiento
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for texts, mels, mel_lengths, spk_embs in loop:
            # 7.1) Preparar secuencias de texto
            sequences, lengths = tts_utils.prepare_input_sequence(texts)
            sequences = sequences.to(device)
            lengths = lengths.to(device)

            # 7.2) Mover datos a dispositivo
            mels = mels.to(device)
            mel_lengths = mel_lengths.to(device)
            spk_embs = spk_embs.to(device)

            # 7.3) Construir target de gate (opcional)
            B, _, T = mels.size()
            gate_padded = torch.zeros((B, T), dtype=torch.float32).to(device)
            for i, L in enumerate(mel_lengths):
                if L < T:
                    gate_padded[i, L:] = 1.0
            output_lengths = mel_lengths

            # 7.4) Forward step (parse_batch recibe spk_embedding)
            inputs, targets, spk_batch = model.parse_batch(
                (sequences, lengths, mels, gate_padded, output_lengths, spk_embs)
            )
            mel_outputs, mel_outputs_postnet, gate_outputs, alignments = model(inputs, spk_batch)

            # 7.5) Cálculo pérdida MSE en mel-spectrogram
            T_pred = mel_outputs_postnet.size(2)
            T_target = mels.size(2)
            T_min = min(T_pred, T_target)
            loss = criterion(
                mel_outputs_postnet[:, :, :T_min],
                mels[:, :, :T_min],
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader)
        print(f"[INFO] Epoch {epoch} completada. Loss promedio: {epoch_loss:.6f}")

        # 7.6) Scheduler step
        scheduler.step()

        # 7.7) Guardar checkpoint si mejora
        ckpt_dir = args.checkpoints_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, "best_tts_zero_shot.pth")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"[INFO] Nuevo mejor checkpoint guardado en: {ckpt_path}")

    print("[INFO] Entrenamiento completado.")


def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Usando dispositivo: {device}")

    # 1) Cargar hiperparámetros
    hparams = load_hparams(args.hparams_path)

    # 2) Instanciar Tacotron2 modificado con speaker embeddings
    BASE_DIR = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "DeepLearningExamples",
            "PyTorch",
            "SpeechSynthesis",
            "Tacotron2",
            "tacotron2",
        )
    )
    import sys
    sys.path.append(BASE_DIR)
    from model import Tacotron2  # Versión modificada con spk_proj

    model = Tacotron2(
        mask_padding=hparams.mask_padding,
        n_mel_channels=hparams.n_mel_channels,
        n_symbols=len(hparams.symbols),
        symbols_embedding_dim=hparams.symbols_embedding_dim,
        encoder_kernel_size=hparams.encoder_kernel_size,
        encoder_n_convolutions=hparams.encoder_n_convolutions,
        encoder_embedding_dim=hparams.encoder_embedding_dim,
        attention_rnn_dim=hparams.attention_rnn_dim,
        attention_dim=hparams.attention_dim,
        attention_location_n_filters=hparams.attention_location_n_filters,
        attention_location_kernel_size=hparams.attention_location_kernel_size,
        n_frames_per_step=hparams.n_frames_per_step,
        decoder_rnn_dim=hparams.decoder_rnn_dim,
        prenet_dim=hparams.prenet_dim,
        max_decoder_steps=hparams.max_decoder_steps,
        gate_threshold=hparams.gate_threshold,
        p_attention_dropout=hparams.p_attention_dropout,
        p_decoder_dropout=hparams.p_decoder_dropout,
        postnet_embedding_dim=hparams.postnet_embedding_dim,
        postnet_kernel_size=hparams.postnet_kernel_size,
        postnet_n_convolutions=hparams.postnet_n_convolutions,
        decoder_no_early_stopping=hparams.decoder_no_early_stopping,
        spk_emb_size=args.spk_emb_size,
    )
    model = model.to(device)

    # 3) Cargar checkpoint entrenado
    print(f"[INFO] Cargando checkpoint: {args.checkpoint_path}")
    state = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print("[INFO] Checkpoint cargado correctamente.")

    # 4) Extraer speaker embedding de referencia
    print(f"[INFO] Extrayendo speaker embedding de: {args.reference_wav}")
    spk_emb = get_spk_embedding(args.reference_wav, device)  # [192]
    spk_emb = spk_emb.unsqueeze(0).to(device)  # [1, 192]

    # 5) Convertir texto a IDs y longitudes
    print(f"[INFO] Procesando texto: \"{args.text}\"")
    tts_utils = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub", "nvidia_tts_utils")
    text_input = [args.text.lower()]
    sequences, lengths = tts_utils.prepare_input_sequence(text_input)
    sequences = sequences.to(device)
    lengths = lengths.to(device)

    # 6) Generar mel-spectrogram con Tacotron2 modificado
    print("[INFO] Generando mel-spectrogram con Tacotron2 modificado...")
    with torch.no_grad():
        mel_spec, mel_lengths, alignments = model.infer(sequences, lengths, spk_emb)

    # 7) Cargar vocoder WaveGlow y generar audio
    print("[INFO] Cargando WaveGlow desde TorchHub...")
    waveglow = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub", "nvidia_waveglow")
    waveglow = waveglow.remove_weightnorm(waveglow).to(device)
    waveglow.eval()
    for p in waveglow.parameters():
        p.requires_grad = False

    print("[INFO] Convirtiendo mel a audio con WaveGlow...")
    with torch.no_grad():
        try:
            audio = waveglow.infer(mel_spec)
        except RuntimeError:
            audio = waveglow.infer(mel_spec.unsqueeze(1))
    audio = audio.squeeze().cpu()

    # 8) Guardar WAV resultante
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir, f"{int(torch.randint(0, 1e9, (1,)).item())}_clone.wav"
    )
    torchaudio.save(output_path, audio.unsqueeze(0), sample_rate=args.sample_rate)
    print(f"[INFO] Audio sintetizado guardado en: {output_path}")


# --------------------------------------------------------------------------------
# 3) PUNTO DE ENTRADA: argparse con subcomandos "train" e "infer"
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TTS con Tacotron2 + speaker embeddings (zero-shot integrado)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcomando "train"
    train_parser = subparsers.add_parser("train", help="Entrenar Tacotron2 con embeddings de speaker")
    train_parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Directorio LJSpeech con metadata.csv y carpeta wavs/"
    )
    train_parser.add_argument(
        "--hparams_path", type=str, required=True,
        help="Ruta al JSON de hiperparámetros de Tacotron2"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=100,
        help="Número de épocas de entrenamiento"
    )
    train_parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Tamaño de batch"
    )
    train_parser.add_argument(
        "--learning_rate", type=float, default=1e-3,
        help="Learning rate para Adam"
    )
    train_parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Número de workers para DataLoader"
    )
    train_parser.add_argument(
        "--checkpoints_dir", type=str, default="checkpoints",
        help="Directorio donde se guardarán checkpoints"
    )
    train_parser.add_argument(
        "--spk_emb_size", type=int, default=192,
        help="Dimensión del speaker embedding (ECAPA-TDNN = 192)"
    )

    # Subcomando "infer"
    infer_parser = subparsers.add_parser("infer", help="Inferir TTS usando modelo entrenado")
    infer_parser.add_argument(
        "--hparams_path", type=str, required=True,
        help="Ruta al JSON de hiperparámetros de Tacotron2"
    )
    infer_parser.add_argument(
        "--checkpoint_path", type=str, required=True,
        help="Ruta al checkpoint .pth generado en entrenamiento"
    )
    infer_parser.add_argument(
        "--reference_wav", type=str, required=True,
        help="Archivo .wav de referencia (ej: data/my_voice/wavs/Voz1.wav)"
    )
    infer_parser.add_argument(
        "--text", type=str, required=True,
        help="Texto a sintetizar (minúsculas con puntuación en inglés)"
    )
    infer_parser.add_argument(
        "--output_dir", type=str, default="samples_cloned",
        help="Directorio donde se guardará el audio generado"
    )
    infer_parser.add_argument(
        "--spk_emb_size", type=int, default=192,
        help="Dimensión del speaker embedding (ECAPA-TDNN = 192)"
    )
    infer_parser.add_argument(
        "--sample_rate", type=int, default=22050,
        help="Frecuencia de muestreo para el WAV de salida"
    )

    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "infer":
        infer(args)
