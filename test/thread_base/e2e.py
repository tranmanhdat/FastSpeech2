from typing import Tuple
from torch import nn
import torch
import os
import uuid
from scipy.io import wavfile


class E2E(nn.Module):
    def __init__(self, acoustic_path: str, vocoder_path: str, model_config, preprocess_config,
                 p_control: float = 1.0, e_control: float = 1.0, d_control: float = 1.0):
        super().__init__()
        # self.model_config = model_config
        # self.preprocess_config = preprocess_config
        self.hop_length = preprocess_config["preprocessing"]["stft"]["hop_length"]
        self.max_wav_value = preprocess_config["preprocessing"]["audio"]["max_wav_value"]
        self.sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        self.acoustic_model = torch.jit.load(acoustic_path)
        self.vocoder_model = torch.jit.load(vocoder_path)
        self.p_control = p_control
        self.e_control = e_control
        self.d_control = d_control

    def forward(self, batch: Tuple[ torch.Tensor, torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,]):
        predictions = self.acoustic_model(
            *(batch[2:]),
            p_control=self.p_control,
            e_control=self.e_control,
            d_control=self.d_control,
        )
        mel_predictions = predictions[1].transpose(1, 2)

        lengths = predictions[9] * self.hop_length
            # self.preprocess_config["preprocessing"]["stft"]["hop_length"]
        wavs = self.vocoder_model(mel_predictions).squeeze(1)
        wavs = (
            wavs.detach().cpu().numpy()
            * self.max_wav_value
        ).astype("int16")
        # wavs = [wav for wav in wavs]

        for i in range(len(wavs)):
            if lengths.numel():
                wavs[i] = wavs[i][: lengths[i]]

    ######
        path = '/tmp/'
        sampling_rate = self.sampling_rate
        wav_files = []
        for wav in wavs:
            wav_name = uuid.uuid4().hex[:25].upper()
            wav_file = os.path.join(path, "{}.wav".format(wav_name))
            wavfile.write(wav_file, sampling_rate,wav)
            # yield wav_file
            wav_files.append(wav_file)

        return wav_files
