import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
from text import _clean_text


def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    # speaker = "mta0"
    speaker = "chieuthuong"
    # speakers = open(f"{in_dir}/speakers.txt", 'r').read().strip().split('\n')
    os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
    with open(os.path.join(in_dir, "transcript.lst"), encoding="utf-8") as f:
        data = f.readlines()
        for line in tqdm(data):
            elements = line.strip().split("\t")
            base_name = elements[0]
            if len(elements) <4:
                continue
            text = elements[3]
            text = _clean_text(text, cleaners)
            # texts[i] = {base_name: text}
            # texts[base_name] = text
        # wav_files = os.listdir(os.path.join(in_dir, speaker))
        # for wav_file in tqdm(wav_files):
            # wav_file = elements[1]
            # base_name = os.path.splitext(wav_file)[0]
            # text = texts[base_name]
            wav_path = elements[1]
            wav, _ = librosa.load(wav_path, sampling_rate)
            wav = wav / max(abs(wav)) * max_wav_value
            wavfile.write(
                    os.path.join(out_dir, speaker, "{}.wav".format(base_name)),
                    sampling_rate,
                    wav.astype(np.int16),
                )
            with open(os.path.join(out_dir, speaker, "{}.lab".format(base_name)),"w",) as f1:
                f1.write(text)
