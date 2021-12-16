import logging
# import numpy as np
from typing import List
import os
import torch
import uuid
import re
from scipy.io.wavfile import write, read
from ts.torch_handler.base_handler import BaseHandler
# from scipy.io import wavfile

# extra-files: phones.txt, script_model.pt, script_vocoder.pt, viet-tts-lexicon.txt
logger = logging.getLogger(__name__)

_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")
_whitespace_re = re.compile(r'\s+')

symbols = [l.strip() for l in open('./phones.txt')]
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
class E2ESynthesizer(BaseHandler):

    def __init__(self):
        self.hifigan_model = None
        self.fastspeech2_model = None
        self.mapping = None
        self.device = None
        self.initialized = False
        self.metrics = None

        self.hop_length = 256
        self.sampling_rate = 22050
        self.max_wav_value = 32768.0
    # From https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/Tacotron2/inference.py
    # def _unwrap_distributed(self, state_dict):
    #     """
    #     Unwraps model from DistributedDataParallel.
    #     DDP wraps model in additional "module.", it needs to be removed for single
    #     GPU inference.
    #     :param state_dict: model's state dict
    #     """
    #     new_state_dict = {}
    #     for key, value in state_dict.items():
    #         new_key = key.replace('module.', '')
    #         new_state_dict[new_key] = value
    #     return new_state_dict

    @staticmethod
    def to_device(data, device):
        if len(data) == 12:
            (
                ids,
                raw_texts,
                speakers,
                texts,
                src_lens,
                max_src_len,
                mels,
                mel_lens,
                max_mel_len,
                pitches,
                energies,
                durations,
            ) = data

            speakers = torch.from_numpy(speakers).long().to(device)
            texts = torch.from_numpy(texts).long().to(device)
            src_lens = torch.from_numpy(src_lens).to(device)
            mels = torch.from_numpy(mels).float().to(device)
            mel_lens = torch.from_numpy(mel_lens).to(device)
            pitches = torch.from_numpy(pitches).float().to(device)
            energies = torch.from_numpy(energies).to(device)
            durations = torch.from_numpy(durations).long().to(device)
            # max_src_len = torch.tensor([max_src_len]).int().to(device)

            return (
                ids,
                raw_texts,
                speakers,
                texts,
                src_lens,
                max_src_len,
                mels,
                mel_lens,
                max_mel_len,
                pitches,
                energies,
                durations,
            )

        if len(data) == 6:
            (ids, raw_texts, speakers, texts, src_lens, max_src_len) = data

            speakers = speakers.long().to(device)
            texts = texts.long().to(device)
            src_lens = src_lens.to(device)
            # max_src_len = torch.tensor([max_src_len]).int().to(device)

            return (ids, raw_texts, speakers, texts, src_lens, max_src_len)
        
        raise f"Input data not in len [6, 12]"
    def initialize(self, ctx):
        """First try to load torchscript else load eager mode state_dict based model"""

        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        if not torch.cuda.is_available() or properties.get("gpu_id") is None :
            raise RuntimeError("This model is not supported on CPU machines.")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")))

        # with zipfile.ZipFile(model_dir + '/tacotron.zip', 'r') as zip_ref:
        #     zip_ref.extractall(model_dir)

        self.hifigan_model = torch.jit.load(os.path.join(model_dir, "script_vocoder.pt"))
        self.hifigan_model.to(self.device)
        self.hifigan_model.eval()

        self.fastspeech2_model = torch.jit.load(os.path.join(model_dir, "script_model.pt"))
        self.fastspeech2_model.to(self.device)
        self.fastspeech2_model.eval()

        logger.debug('FastSpeech2-FifiGAN models file loaded successfully')
        self.initialized = True

    def preprocess(self, data):
        """
         Scales, crops, and normalizes a PIL image for a MNIST model,
         returns an Numpy array
        """
        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")
        text = text.decode('utf-8')

        ids = text[:100]
        speakers = torch.tensor([0])
        raw_texts = text
        phones, phone_groups = preprocess_vie(text, './viet-tts-lexicon.txt' )
        texts = torch.tensor(phones)
        text_lens = torch.tensor([len(texts[0])])
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]

        # return [self.to_device(batch, self.device) for batch in batchs]
        return self.to_device(batchs[0], self.device), phone_groups

    def inference(self, data):
        # 256: hop_length, 22050: sampling_rate
        decode_durations = lambda x: x*256/22050
        with torch.no_grad():
            # TODO: correct this
            # _, mel, _, _ = self.fastspeech2_model.forwad(data)
            # audio = self.waveglow_model.infer(mel)
            # batch = self.preprocess(data)
            batch = data[0]
            phones_groups = data[1]
            # print(f"Yasuo here {batch}")
            predictions = self.fastspeech2_model.forward(*(batch[2:]), p_control=1.0, e_control=1.0, d_control=1.0)
            p_durations = predictions[5]
            p_durations = [decode_durations[d] for d in p_durations]

            mel_predictions = predictions[1].transpose(1, 2)

            lengths = predictions[9] * self.hop_length
                # self.preprocess_config["preprocessing"]["stft"]["hop_length"]
            wavs = self.hifigan_model(mel_predictions).squeeze(1)
            wavs = (
                wavs.detach().cpu().numpy()
                * self.max_wav_value
            ).astype("int16")
            # wavs = [wav for wav in wavs]

            for i in range(len(wavs)):
                if lengths.numel():
                    wavs[i] = wavs[i][: lengths[i]]

        ######
        return wavs[0]
            # return audio

    def postprocess(self, inference_output):
        sampling_rate = self.sampling_rate
        wav_name = uuid.uuid4().hex[:25].upper()
        path = "/tmp/{}.wav".format(wav_name)
        write(path, sampling_rate, inference_output)
        # yield wav_file

        with open(path, 'rb') as output:
            data = output.read()
        os.remove(path)
        return [data]


def preprocess_vie(text:str, lexicon_path:str ):
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(lexicon_path)

    phones: List[str] = []
    errs: List[str] = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    phone_groups = []
    for w in words:
        if w.lower() in lexicon:
            p = lexicon[w.lower()]
            phones += p
            phone_groups.append((p))
        else:
            # phones += list(filter(lambda p: p != " ", g2p(w)))
            errs.append(w.lower())
    print(f"Error words: {'~'.join(errs)}")
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = torch.tensor(
        text_to_sequence(
            phones, 
        )
    )
    return sequence.unsqueeze(0), phone_groups

def text_to_sequence(text,):
    sequence = []

    # Check for curly braces and treat their contents as ARPAbet:

    while len(text):
        m = _curly_re.match(text)

        if not m:
            sequence += _symbols_to_sequence(_clean_text(text,))
            break
        sequence += _symbols_to_sequence(_clean_text(m.group(1),))
        sequence +=_symbols_to_sequence([s for s in m.group(2).split()])
        text = m.group(3)

    return sequence
def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]

def _should_keep_symbol(s):
    return s in _symbol_to_id and s != "_" and s != "~"

def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)

def _clean_text(text: str):
    text = text.lower()
    text = collapse_whitespace(text)
    return text

def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon
