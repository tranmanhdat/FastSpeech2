import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths
from typing import List

# TODO: Hint consider var: torch.Tensor = torch.tensor([]) -> var: torch.Tensor ## as type init only


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(
            preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

# TODO: ignore speaker_emb for immediate reference
        # self.speaker_emb = torch.Tensor()
        # if model_config["multi_speaker"]:
        #     with open(
        #         os.path.join(
        #             preprocess_config["path"]["preprocessed_path"], "speakers.json"
        #         ),
        #         "r",
        #     ) as f:
        #         n_speaker = len(json.load(f))
        # self.speaker_emb = nn.Embedding(
        #     n_speaker,
        #     model_config["transformer"]["encoder_hidden"],
        # )

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len: int = 0,
        mels=torch.tensor([]),
        mel_lens=torch.tensor([]),
        max_mel_len: int = -1,
        p_targets=torch.tensor([]),
        e_targets=torch.tensor([]),
        d_targets=torch.tensor([]),
        p_control: float = 1.0,
        e_control: float = 1.0,
        d_control: float = 1.0,
    ):
        # max_src_len = max_src_len.squeeze()
        # max_mel_len = max_mel_len.squeeze()

        # TODO: make sure empty tensors from emp_float are not refer same(e.g in grad)
        emp_float: List[float] = []

        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens.numel()
            else torch.tensor(emp_float)
        )
        # try:
        output = self.encoder(texts, src_masks)
        # except Exception as e:
        # print(f"Error when encoding {texts}, {src_masks}")
        # raise e

        # try:
        # if self.speaker_emb is not None:
        #     output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
        #         -1, max_src_len, -1
        #     )

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )
        # except Exception as e:
        #     print(f"Error with VarianceAdapter {texts} {src_masks}")
        #     raise e

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )
