# model.py

# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

from math import sqrt
import torch
from torch import nn
from torch.nn import functional as F
import sys
from os.path import abspath, dirname

# Permite importar tacotron2_common desde el entrypoint global
sys.path.append(abspath(dirname(__file__) + '/../'))

from tacotron2_common.layers import ConvNorm, LinearNorm
from tacotron2_common.utils import to_gpu, get_mask_from_lengths


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(
            2, attention_n_filters,
            kernel_size=attention_kernel_size,
            padding=padding, bias=False, stride=1, dilation=1
        )
        self.location_dense = LinearNorm(
            attention_n_filters, attention_dim,
            bias=False, w_init_gain='tanh'
        )

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim,
                 attention_dim, attention_location_n_filters,
                 attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(
            attention_rnn_dim, attention_dim,
            bias=False, w_init_gain='tanh'
        )
        self.memory_layer = LinearNorm(
            embedding_dim, attention_dim,
            bias=False, w_init_gain='tanh'
        )
        self.v = LinearNorm(
            attention_dim, 1,
            bias=False
        )
        self.location_layer = LocationLayer(
            attention_location_n_filters,
            attention_location_kernel_size,
            attention_dim
        )
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """
        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory
        ))
        energies = energies.squeeze(2)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cumulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat
        )
        alignment = alignment.masked_fill(mask, self.score_mask_value)
        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(
            attention_weights.unsqueeze(1), memory
        )
        attention_context = attention_context.squeeze(1)
        return attention_context, attention_weights


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList([
            LinearNorm(in_size, out_size, bias=False)
            for (in_size, out_size) in zip(in_sizes, sizes)
        ])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class Postnet(nn.Module):
    """Postnet
        - Cinco convoluciones 1D con 512 canales y kernel size = 5
    """
    def __init__(self, n_mel_channels, postnet_embedding_dim,
                 postnet_kernel_size, postnet_n_convolutions):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        # Primera capa
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    n_mel_channels,
                    postnet_embedding_dim,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain='tanh'
                ),
                nn.BatchNorm1d(postnet_embedding_dim)
            )
        )

        # Capas intermedias
        for i in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(
                        postnet_embedding_dim,
                        postnet_embedding_dim,
                        kernel_size=postnet_kernel_size,
                        stride=1,
                        padding=int((postnet_kernel_size - 1) / 2),
                        dilation=1,
                        w_init_gain='tanh'
                    ),
                    nn.BatchNorm1d(postnet_embedding_dim)
                )
            )

        # Última capa
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    postnet_embedding_dim,
                    n_mel_channels,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain='linear'
                ),
                nn.BatchNorm1d(n_mel_channels)
            )
        )
        self.n_convs = len(self.convolutions)

    def forward(self, x):
        i = 0
        for conv in self.convolutions:
            if i < self.n_convs - 1:
                x = F.dropout(torch.tanh(conv(x)), 0.5, training=self.training)
            else:
                x = F.dropout(conv(x), 0.5, training=self.training)
            i += 1
        return x


class Encoder(nn.Module):
    """Encoder module:
        - Tres bloques de convoluciones 1D
        - LSTM bidireccional
    """
    def __init__(self, encoder_n_convolutions,
                 encoder_embedding_dim, encoder_kernel_size):
        super(Encoder, self).__init__()
        convolutions = []
        for _ in range(encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(
                    encoder_embedding_dim,
                    encoder_embedding_dim,
                    kernel_size=encoder_kernel_size,
                    stride=1,
                    padding=int((encoder_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain='relu'
                ),
                nn.BatchNorm1d(encoder_embedding_dim)
            )
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(
            encoder_embedding_dim,
            int(encoder_embedding_dim / 2),
            1,
            batch_first=True,
            bidirectional=True
        )

    @torch.jit.ignore
    def forward(self, x, input_lengths):
        """
        x: Tensor [B, encoder_embedding_dim, T_text] previo a BLSTM
        input_lengths: lista con las longitudes de cada secuencia textual
        Retorna:
          - outputs: Tensor [B, T_text, encoder_embedding_dim]
        """
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True
        )
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True
        )
        return outputs

    @torch.jit.export
    def infer(self, x, input_lengths):
        """
        Misma lógica que forward, pero en modo inferencia (sin dropout).
        x: Tensor [B, encoder_embedding_dim, T_text]
        input_lengths: Tensor [B]
        """
        device = x.device
        for conv in self.convolutions:
            x = F.relu(conv(x.to(device)))

        x = x.transpose(1, 2)
        input_lengths = input_lengths.cpu()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True
        )
        outputs, _ = self.lstm(x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True
        )
        return outputs


class Decoder(nn.Module):
    def __init__(self, n_mel_channels, n_frames_per_step,
                 encoder_embedding_dim, attention_dim,
                 attention_location_n_filters,
                 attention_location_kernel_size,
                 attention_rnn_dim, decoder_rnn_dim,
                 prenet_dim, max_decoder_steps, gate_threshold,
                 p_attention_dropout, p_decoder_dropout,
                 early_stopping):
        super(Decoder, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.n_frames_per_step = n_frames_per_step
        self.encoder_embedding_dim = encoder_embedding_dim
        self.attention_rnn_dim = attention_rnn_dim
        self.decoder_rnn_dim = decoder_rnn_dim
        self.prenet_dim = prenet_dim
        self.max_decoder_steps = max_decoder_steps
        self.gate_threshold = gate_threshold
        self.p_attention_dropout = p_attention_dropout
        self.p_decoder_dropout = p_decoder_dropout
        self.early_stopping = early_stopping

        self.prenet = Prenet(
            n_mel_channels * n_frames_per_step,
            [prenet_dim, prenet_dim]
        )

        self.attention_rnn = nn.LSTMCell(
            prenet_dim + encoder_embedding_dim,
            attention_rnn_dim
        )

        self.attention_layer = Attention(
            attention_rnn_dim, encoder_embedding_dim,
            attention_dim, attention_location_n_filters,
            attention_location_kernel_size
        )

        self.decoder_rnn = nn.LSTMCell(
            attention_rnn_dim + encoder_embedding_dim,
            decoder_rnn_dim, 1
        )

        self.linear_projection = LinearNorm(
            decoder_rnn_dim + encoder_embedding_dim,
            n_mel_channels * n_frames_per_step
        )

        self.gate_layer = LinearNorm(
            decoder_rnn_dim + encoder_embedding_dim, 1,
            bias=True, w_init_gain='sigmoid'
        )

    def get_go_frame(self, memory):
        """Obtiene un frame de ceros para iniciar el decoder"""
        B = memory.size(0)
        dtype = memory.dtype
        device = memory.device
        decoder_input = torch.zeros(
            B,
            self.n_mel_channels * self.n_frames_per_step,
            dtype=dtype, device=device
        )
        return decoder_input

    def initialize_decoder_states(self, memory):
        """Inicializa los estados ocultos del decoder y atención"""
        B = memory.size(0)
        MAX_TIME = memory.size(1)
        dtype = memory.dtype
        device = memory.device

        attention_hidden = torch.zeros(
            B, self.attention_rnn_dim, dtype=dtype, device=device
        )
        attention_cell = torch.zeros(
            B, self.attention_rnn_dim, dtype=dtype, device=device
        )

        decoder_hidden = torch.zeros(
            B, self.decoder_rnn_dim, dtype=dtype, device=device
        )
        decoder_cell = torch.zeros(
            B, self.decoder_rnn_dim, dtype=dtype, device=device
        )

        attention_weights = torch.zeros(
            B, MAX_TIME, dtype=dtype, device=device
        )
        attention_weights_cum = torch.zeros(
            B, MAX_TIME, dtype=dtype, device=device
        )
        attention_context = torch.zeros(
            B, self.encoder_embedding_dim, dtype=dtype, device=device
        )

        processed_memory = self.attention_layer.memory_layer(memory)

        return (attention_hidden, attention_cell,
                decoder_hidden, decoder_cell,
                attention_weights, attention_weights_cum,
                attention_context, processed_memory)

    def parse_decoder_inputs(self, decoder_inputs):
        """Prepara los inputs del decoder (mel-spectrograms) para training"""
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1) / self.n_frames_per_step), -1
        )
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """Reconstruye las salidas del decoder en la forma final"""
        # (T_out, B) -> (B, T_out)
        alignments = alignments.transpose(0, 1).contiguous()
        gate_outputs = gate_outputs.transpose(0, 1).contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = mel_outputs.transpose(0, 1).contiguous()
        shape = (mel_outputs.shape[0], -1, self.n_mel_channels)
        mel_outputs = mel_outputs.view(*shape)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)
        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input, attention_hidden, attention_cell,
               decoder_hidden, decoder_cell, attention_weights,
               attention_weights_cum, attention_context, memory,
               processed_memory, mask):
        """Un paso de inferencia en el decoder (training)"""
        cell_input = torch.cat((decoder_input, attention_context), -1)

        attention_hidden, attention_cell = self.attention_rnn(
            cell_input, (attention_hidden, attention_cell)
        )
        attention_hidden = F.dropout(
            attention_hidden, self.p_attention_dropout, self.training
        )

        attention_weights_cat = torch.cat(
            (attention_weights.unsqueeze(1),
             attention_weights_cum.unsqueeze(1)), dim=1
        )
        attention_context, attention_weights = self.attention_layer(
            attention_hidden,
            memory,
            processed_memory,
            attention_weights_cat,
            mask
        )

        attention_weights_cum += attention_weights
        decoder_input = torch.cat(
            (attention_hidden, attention_context), dim=1
        )

        decoder_hidden, decoder_cell = self.decoder_rnn(
            decoder_input, (decoder_hidden, decoder_cell)
        )
        decoder_hidden = F.dropout(
            decoder_hidden, self.p_decoder_dropout, self.training
        )

        decoder_hidden_attention_context = torch.cat(
            (decoder_hidden, attention_context), dim=1
        )
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context
        )
        gate_prediction = self.gate_layer(
            decoder_hidden_attention_context
        )

        return (decoder_output, gate_prediction, attention_hidden,
                attention_cell, decoder_hidden, decoder_cell,
                attention_weights, attention_weights_cum, attention_context)

    @torch.jit.ignore
    def forward(self, memory, decoder_inputs, memory_lengths):
        """
        Path del decoder en modo training (teacher forcing, sin condición de speaker).
        Aquí NO se aplica spk_embedding porque ya se incorporó antes, en Tacotron2.forward.
        """
        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        mask = get_mask_from_lengths(memory_lengths)
        (attention_hidden,
         attention_cell,
         decoder_hidden,
         decoder_cell,
         attention_weights,
         attention_weights_cum,
         attention_context,
         processed_memory) = self.initialize_decoder_states(memory)

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            (mel_output,
             gate_output,
             attention_hidden,
             attention_cell,
             decoder_hidden,
             decoder_cell,
             attention_weights,
             attention_weights_cum,
             attention_context) = self.decode(
                decoder_input,
                attention_hidden,
                attention_cell,
                decoder_hidden,
                decoder_cell,
                attention_weights,
                attention_weights_cum,
                attention_context,
                memory,
                processed_memory,
                mask
            )
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze()]
            alignments += [attention_weights]

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            torch.stack(mel_outputs),
            torch.stack(gate_outputs),
            torch.stack(alignments)
        )

        return mel_outputs, gate_outputs, alignments

    @torch.jit.export
    def infer(self, memory, memory_lengths):
        """
        Inference del decoder (sin teacher forcing, sin condición de speaker acá).
        La condición se aplica en Tacotron2.infer antes de llamar a este método.
        """
        decoder_input = self.get_go_frame(memory)
        mask = get_mask_from_lengths(memory_lengths)
        (attention_hidden,
         attention_cell,
         decoder_hidden,
         decoder_cell,
         attention_weights,
         attention_weights_cum,
         attention_context,
         processed_memory) = self.initialize_decoder_states(memory)

        mel_lengths = torch.zeros(
            [memory.size(0)], dtype=torch.int32, device=memory.device
        )
        not_finished = torch.ones(
            [memory.size(0)], dtype=torch.int32, device=memory.device
        )

        mel_outputs, gate_outputs, alignments = (
            torch.zeros(1), torch.zeros(1), torch.zeros(1)
        )
        first_iter = True
        while True:
            decoder_input = self.prenet(decoder_input)
            (mel_output,
             gate_output,
             attention_hidden,
             attention_cell,
             decoder_hidden,
             decoder_cell,
             attention_weights,
             attention_weights_cum,
             attention_context) = self.decode(
                decoder_input,
                attention_hidden,
                attention_cell,
                decoder_hidden,
                decoder_cell,
                attention_weights,
                attention_weights_cum,
                attention_context,
                memory,
                processed_memory,
                mask
            )

            if first_iter:
                mel_outputs = mel_output.unsqueeze(0)
                gate_outputs = gate_output
                alignments = attention_weights
                first_iter = False
            else:
                mel_outputs = torch.cat(
                    (mel_outputs, mel_output.unsqueeze(0)), dim=0
                )
                gate_outputs = torch.cat((gate_outputs, gate_output), dim=0)
                alignments = torch.cat(
                    (alignments, attention_weights), dim=0
                )

            dec = torch.le(
                torch.sigmoid(gate_output),
                self.gate_threshold
            ).to(torch.int32).squeeze(1)

            not_finished = not_finished * dec
            mel_lengths += not_finished

            if self.early_stopping and torch.sum(not_finished) == 0:
                break
            if len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments
        )

        return mel_outputs, gate_outputs, alignments, mel_lengths


class Tacotron2(nn.Module):
    def __init__(self, mask_padding, n_mel_channels,
                 n_symbols, symbols_embedding_dim, encoder_kernel_size,
                 encoder_n_convolutions, encoder_embedding_dim,
                 attention_rnn_dim, attention_dim, attention_location_n_filters,
                 attention_location_kernel_size, n_frames_per_step,
                 decoder_rnn_dim, prenet_dim, max_decoder_steps, gate_threshold,
                 p_attention_dropout, p_decoder_dropout,
                 postnet_embedding_dim, postnet_kernel_size,
                 postnet_n_convolutions, decoder_no_early_stopping,
                 spk_emb_size=192):
        """
        Parámetros originales + spk_emb_size (dimensión del embedding de hablante).
        """
        super(Tacotron2, self).__init__()
        self.mask_padding = mask_padding
        self.n_mel_channels = n_mel_channels
        self.n_frames_per_step = n_frames_per_step

        # Embedding de texto
        self.embedding = nn.Embedding(n_symbols, symbols_embedding_dim)
        std = sqrt(2.0 / (n_symbols + symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)

        # Módulos de encoder, decoder y postnet
        self.encoder = Encoder(
            encoder_n_convolutions,
            encoder_embedding_dim,
            encoder_kernel_size
        )
        self.decoder = Decoder(
            n_mel_channels, n_frames_per_step,
            encoder_embedding_dim, attention_dim,
            attention_location_n_filters,
            attention_location_kernel_size,
            attention_rnn_dim, decoder_rnn_dim,
            prenet_dim, max_decoder_steps,
            gate_threshold, p_attention_dropout,
            p_decoder_dropout,
            not decoder_no_early_stopping
        )
        self.postnet = Postnet(
            n_mel_channels, postnet_embedding_dim,
            postnet_kernel_size, postnet_n_convolutions
        )

        # --------------------------------------------
        #  Modificaciones para integrar spk_embedding
        # --------------------------------------------
        # Tamaño del embedding de hablante (p. ej., ECAPA-TDNN = 192).
        self.spk_emb_size = spk_emb_size
        # Capa para proyectar spk_embedding a encoder_embedding_dim
        self.spk_proj = nn.Linear(self.spk_emb_size, encoder_embedding_dim)
        nn.init.xavier_uniform_(self.spk_proj.weight, gain=0.1)
        nn.init.zeros_(self.spk_proj.bias)

    def parse_batch(self, batch):
        """
        Ajuste de parse_batch para recibir spk_embedding en el batch de entrenamiento.
        batch debe ser una tupla:
          (text_padded, input_lengths, mel_padded, gate_padded, output_lengths, spk_embedding)
        donde:
          - text_padded: Tensor [B, T_text]
          - input_lengths: Tensor [B]
          - mel_padded: Tensor [B, n_mel_channels, T_mel]
          - gate_padded: Tensor [B, T_mel]
          - output_lengths: Tensor [B]
          - spk_embedding: Tensor [B, spk_emb_size]
        """
        text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths, spk_embedding = batch

        text_padded   = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        max_len       = torch.max(input_lengths.data).item()
        mel_padded    = to_gpu(mel_padded).float()
        gate_padded   = to_gpu(gate_padded).float()
        output_lengths= to_gpu(output_lengths).long()
        spk_embedding = to_gpu(spk_embedding).float()  # nuevo

        return (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths),
            (mel_padded, gate_padded),
            spk_embedding  # retornamos el embedding
        )

    def parse_output(self, outputs, output_lengths):
        if self.mask_padding and output_lengths is not None:
            mask = get_mask_from_lengths(output_lengths)
            mask = mask.expand(
                self.n_mel_channels,
                mask.size(0),
                mask.size(1)
            ).permute(1, 0, 2)

            # Evitando operaciones inplace explícitamente
            outputs[0] = outputs[0].masked_fill(mask, 0.0)
            outputs[1] = outputs[1].masked_fill(mask, 0.0)
            outputs[2] = outputs[2].masked_fill(mask[:, 0, :], 1e3)

        return outputs


    def forward(self, inputs, spk_embedding):
        """
        inputs: tupla generada por parse_batch:
          - inputs: (text_padded, input_lengths, mel_padded, max_len, output_lengths)
          - targets: (mel_padded, gate_padded)
          - spk_embedding: Tensor [B, spk_emb_size]
        """
        (text_padded, input_lengths, mel_padded, max_len, output_lengths) = inputs
        _, _ = (mel_padded, output_lengths)  # se usan para parse_output
        input_lengths, output_lengths = input_lengths.data, output_lengths.data

        # 1) Embedding de texto y encoder
        embedded_inputs = self.embedding(text_padded).transpose(1, 2)
        encoder_outputs = self.encoder(embedded_inputs, input_lengths)
        # encoder_outputs: [B, T_text, encoder_embedding_dim]

        # 2) PROYECTAR y sumar el embedding del hablante
        spk_emb_norm = (spk_embedding - spk_embedding.mean(dim=1, keepdim=True)) / (spk_embedding.std(dim=1, keepdim=True) + 1e-6)
        spk_emb_proj = self.spk_proj(spk_emb_norm)
        spk_emb_proj = spk_emb_proj.squeeze() if spk_emb_proj.dim() > 2 else spk_emb_proj  # Garantiza 2 dimensiones
        print("spk_emb_proj mean:", spk_emb_proj.mean().item(), "std:", spk_emb_proj.std().item())
        B, T_text, encoder_dim = encoder_outputs.size()
        spk_emb_expanded = spk_emb_proj.unsqueeze(1).repeat(1, T_text, 1)  # [B, T_text, encoder_embedding_dim]
        encoder_outputs = encoder_outputs.clone() + spk_emb_expanded

        # 3) Decoder (teacher-forcing)
        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mel_padded, memory_lengths=input_lengths
        )

        # 4) Postnet
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            output_lengths
        )

    @torch.jit.export
    def infer(self, inputs, input_lengths, spk_embedding):
        """
        inputs: Tensor [B, T_text] con índices de texto
        input_lengths: Tensor [B]
        spk_embedding: Tensor [B, spk_emb_size]
        """
        # 1) Embedding de texto y encoder (inferencia)
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.infer(embedded_inputs, input_lengths)
        # encoder_outputs: [B, T_text, encoder_embedding_dim]

        # 2) PROYECTAR y sumar el embedding del hablante
        if spk_embedding.dim() > 2:
            spk_embedding = spk_embedding.squeeze()
            if spk_embedding.dim() == 1:
                spk_embedding = spk_embedding.unsqueeze(0)

        spk_emb_norm = (spk_embedding - spk_embedding.mean(dim=1, keepdim=True)) / (spk_embedding.std(dim=1, keepdim=True) + 1e-6)
        spk_emb_proj = self.spk_proj(spk_emb_norm)
        spk_emb_expanded = spk_emb_proj.unsqueeze(1).expand(
            -1, encoder_outputs.size(1), -1
        )
        encoder_outputs  = encoder_outputs + spk_emb_expanded

        # 3) Decoder (inferencia)
        mel_outputs, gate_outputs, alignments, mel_lengths = self.decoder.infer(
            encoder_outputs, input_lengths
        )

        # 4) Postnet
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        # 5) Ajuste final de alignments para match batch size
        BS = mel_outputs_postnet.size(0)
        alignments = alignments.unfold(1, BS, BS).transpose(0, 2)

        return mel_outputs_postnet, mel_lengths, alignments
