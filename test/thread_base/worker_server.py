# from multiprocessing import Manager
from posixpath import basename
from e2e import E2E
import numpy as np
import re
from string import punctuation
import torch
from torch import nn
# import multiprocessing as mp
import time
import threading
import os
from types import SimpleNamespace
import zmq.green as zmq
from .misc import mpickle
# from misc import shared_memory as sm
import torch.multiprocessing as mp
import uuid
import sys
import yaml
# from text.symbols import symbols
if __name__ == '__main__':
    symbols = [l.strip() for l in open('./phones.txt')]
    preprocess_config = yaml.load(
        open('../../config/Viet_tts/preprocess.yaml', "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open('../../config/Viet_tts/model.yaml', "r"), Loader=yaml.FullLoader)
    lex_path = '../../lexicon/viet-tts-lexicon.txt' 
else:
    cwd = os.path.dirname(__file__)
    symbols = [l.strip() for l in open(os.path.join(os.path.dirname(__file__), './phones.txt'))]
    preprocess_config = yaml.load(
        open(os.path.join(cwd, '../../config/Viet_tts/preprocess.yaml'), "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(os.path.join(cwd, '../../config/Viet_tts/model.yaml'), "r"), Loader=yaml.FullLoader)
    lex_path = os.path.join(cwd, '../../lexicon/viet-tts-lexicon.txt' )


QUEUE_SIZE = mp.Value('i', 0)
TOPIC = 'snaptravel'
prediction_functions = {}
RECEIVE_PORT = os.getenv("RECEIVE_PORT")
SEND_PORT = os.getenv("SEND_PORT")
NUM_MODEL = 5

########################

punctuation = "!\"#$%&'()*+-/:;<=>?@[\]^_`{|}~"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")
_whitespace_re = re.compile(r'\s+')


class WrapperModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.e2e = E2E(*args, **kwargs)
        
    def prepare(self, text):
        ids = text[:100]
        speakers = torch.tensor([0])
        raw_texts = text
        ## TODO: load lexicon once
        texts = torch.tensor(preprocess_vie(text, lex_path))
        text_lens = torch.tensor([len(texts[0])])
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]

        return [to_device(batch, device) for batch in batchs]

    def forward(self, text, shared_dict=None):
        batch = self.prepare(text)

        # try: 
        results = self.e2e(batch)
        if shared_dict is not None:
            shared_dict.update({"result": results})
    # except Exception as e:
        # shared_dict.update({"error": str(e)})
        return results

def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)

def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]

def _should_keep_symbol(s):
    return s in _symbol_to_id and s != "_" and s != "~"

# TODO: more cleaner
def _clean_text(text: str):
    text = text.lower()
    text = collapse_whitespace(text)
    return text

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


# TODO: g2p for sp, spn, sil
def preprocess_vie(text:str, lexicon_path:str ):
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(lexicon_path)

    phones: str = []
    errs: str = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
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
    return sequence.unsqueeze(0)


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

def load_models():
    models = SimpleNamespace()
    # This is where you load your models
    # For example, model.model1 = model1.load_model()
    # where
    # `model1.py` has
    # def load_model():
    #   archive = load_archive(SERIALIZATION_DIR)
    #   archive.model.share_memory()
    #   predictor = Predictor.from_archive(archive, 'model')
    #   return predictor
    # e2e_model = E2E('./script_model.pt', './script_vocoder.pt', model_config, preprocess_config)
    models = {f'model-{i:02d}': WrapperModel('./script_model.pt', './script_vocoder.pt', model_config, preprocess_config) \
        for i in range(NUM_MODEL)}
    return models


models = load_models()


def _parse_recv_for_json(result, topic=TOPIC):
    # print(f"Inside parse json, {result}")
    compressed_json = result[len(topic) + 1:]
    return mpickle.decompress(compressed_json)


def _decrease_queue():
    with QUEUE_SIZE.get_lock():
        QUEUE_SIZE.value -= 1


def _increase_queue():
    with QUEUE_SIZE.get_lock():
        QUEUE_SIZE.value += 1




def send_prediction(message, result_publisher, topic=TOPIC):
    _increase_queue()

    print('Worker send prediction')
    model_name = message['model']
    body = message['body']
    id = message['id']

    if not model_name:
        compressed_message = mpickle.compress(
            {'error': True, 'error_msg': 'Model doesn\'t exist', 'id': id})
        result_publisher.send(f'{topic} '.encode('utf8') + compressed_message)
        _decrease_queue()
        return

    predict = prediction_functions.get(model_name)
    # f = sm.function_wrapper(predict)
    # f = function_wrapper(predict)
    f = predict
    time.sleep(2.)
    # result = sm.run_function(f, *body)
    result = run_function(f, *body)

    if result.get('error'):
        compressed_message =mpickle.compress(
            {'error': True, 'error_msg': result['error'], 'id': id})
        result_publisher.send(f'{topic} '.encode('utf8') + compressed_message)
        _decrease_queue()
        return

    if result.get('result') is None:
        compressed_message =mpickle.compress(
            {'error': True, 'error_msg': 'No result was given: ' + str(result), 'id': id})
        result_publisher.send(f'{topic} '.encode('utf8') + compressed_message)
        _decrease_queue()
        return

    prediction = result['result']

    compressed_message =mpickle.compress({'prediction': prediction, 'id': id})
    result_publisher.send(f'{topic} '.encode('utf8') + compressed_message)
    _decrease_queue()


def queue_size():
    return QUEUE_SIZE.value


def start():
    print('Worker started')
    global prediction_functions
    global models
    # models = load_models()

    # prediction_functions = {
    #   # This is where you would add your models for inference
    #   # For example, 'model1': model.model1.predict,
    #   #              'model2': model.model2.predict,
    #   'queue': queue_size
    # }

    prediction_functions = {
        f"model-{i:02d}": models[f"model-{i:02d}"].forward for i in range(NUM_MODEL)
    }
    prediction_functions['queue'] = queue_size

    print(f'Connecting to {RECEIVE_PORT} in server')
    context = zmq.Context()
    work_subscriber = context.socket(zmq.SUB)
    work_subscriber.setsockopt(zmq.SUBSCRIBE, TOPIC.encode('utf8'))
    work_subscriber.bind(f'tcp://127.0.0.1:{RECEIVE_PORT}')

    # send work
    print(f'Connecting to {SEND_PORT} in server')
    result_publisher = context.socket(zmq.PUB)
    result_publisher.bind(f'tcp://127.0.0.1:{SEND_PORT}')

    print('Server started')
    while True:
        message = _parse_recv_for_json(work_subscriber.recv())
        threading.Thread(target=send_prediction, args=(
            message, result_publisher,), kwargs={'topic': TOPIC}).start()


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    MANAGER = mp.Manager()
    def function_wrapper(f):
        
        def predict(*args, shared_dict=None):  # pylint: disable=syntax-error
        # try:
            prediction = f(*args)
            if shared_dict is not None:
                shared_dict.update({"result": prediction})
        # except Exception as e:
            shared_dict.update({"error": str(e)})
        # predict.__name__ = predict.__qualname__ = uuid.uuid4().hex
        # setattr(sys.modules[predict.__module__], predict.__name__, predict)
        return predict

    def run_function(f, *args):
    # try:
        print(f"Yassuo, {type(MANAGER)}")
        shared_dict = MANAGER.dict()
        p = mp.Process(target=f, args=args, kwargs={'shared_dict': shared_dict}, daemon=True)
        p.start()
        p.join()
        res = dict(shared_dict)
        return res
    # except Exception as e:
        # return {"error": str(e)}
    start()
