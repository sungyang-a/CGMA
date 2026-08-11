# -*- coding: utf-8 -*-
"""Proxy-free CGMA inference model.

This module contains only the 0.80 M-parameter deployment graph reported in
the paper. Training-only proxy modules are intentionally absent.
"""
import torch
import torch.nn as nn


def masked_mean(seq, mask):
    m = mask.unsqueeze(-1).float()
    return (seq * m).sum(1) / m.sum(1).clamp(min=1)


class CGMAInference(nn.Module):
    def __init__(self, hid=128, nclass=2):
        super().__init__()
        self.v_lstm = nn.LSTM(136, hid, batch_first=True, bidirectional=True)
        self.a_lstm = nn.LSTM(128, hid, batch_first=True, bidirectional=True)
        dim = hid * 2
        self.comp_v = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.ReLU(),
            nn.Linear(dim // 2, 1), nn.Sigmoid(),
        )
        self.comp_a = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.ReLU(),
            nn.Linear(dim // 2, 1), nn.Sigmoid(),
        )
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.fc = nn.Sequential(
            nn.Linear(dim * 2, hid), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hid, nclass),
        )

    def forward(self, video, audio, video_mask, audio_mask, return_gates=False):
        video_vec = masked_mean(self.v_lstm(video)[0], video_mask)
        audio_vec = masked_mean(self.a_lstm(audio)[0], audio_mask)
        w_video = self.comp_v(video_vec)
        w_audio = self.comp_a(audio_vec)
        h_video = w_video * video_vec
        h_audio = w_audio * audio_vec
        fusion_gate = self.gate(torch.cat([h_video, h_audio], dim=-1))
        logits = self.fc(torch.cat([h_video, fusion_gate * h_audio], dim=-1))
        if return_gates:
            return logits, w_video, w_audio
        return logits


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())
