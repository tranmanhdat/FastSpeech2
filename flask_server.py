from flask import Flask
from flask import request, Response, send_file, make_response
from flask_cors import CORS, cross_origin
import yaml
from utils.model import get_model_flask, get_vocoder
import torch
import numpy as np
from synthesize import preprocess_english, synthesize
import os
from text import file_process
import base64
import io

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
path_to_file = ''

preprocess_config = "config/Viet_tts/preprocess.yaml"
model_config = "config/Viet_tts/model.yaml"
train_config = "config/Viet_tts/train.yaml"

preprocess_config = yaml.load(open(preprocess_config, "r"), Loader=yaml.FullLoader)
model_config = yaml.load(open(model_config, "r"), Loader=yaml.FullLoader)
train_config = yaml.load(open(train_config, "r"), Loader=yaml.FullLoader)
configs = (preprocess_config, model_config, train_config)
device = torch.device("cpu")
restore_step = 30000
model = get_model_flask(restore_step, configs, device)
vocoder = get_vocoder(model_config, device)
pitch_control = 1.0
energy_control = 1.0
duration_control = 1.0
control_values = pitch_control, energy_control, duration_control
speaker_id = 0
speakers = np.array([speaker_id])


@app.route('/tts', methods=['POST', 'GET'])
@cross_origin()
def tts():
    if request.method == 'POST':
        txt = request.get_json()
        text = txt['text']
        ids = raw_texts = [text[:100]]
        texts = np.array([preprocess_english(text, preprocess_config)])

        text_lens = np.array([len(texts[0])])
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]
        synthesize(model, restore_step, configs, vocoder, batchs, control_values)
        path = "./output/result/viet-tts"
        global path_to_file
        path_to_file = os.path.join(path, "{}.wav".format("out"))
        image_binary = open(os.path.join(path, "{}.png".format("out")), 'rb').read()
        response = make_response(base64.b64encode(image_binary))
        response.headers.set('Content-Type', 'image/png')
        response.headers.set('Content-Disposition', 'attachment', filename='result.png')
        return response

    else:
        def generate():
            global path_to_file
            f_wav = open(path_to_file, "rb")
            path_to_file = ''
            data = f_wav.read(1024)
            while data:
                yield data
                data = f_wav.read(1024)

        return Response(generate(), mimetype="audio/wav")


@app.route('/file_ts', methods=['POST', 'GET'])
@cross_origin()
def file_ts():
    if request.method == 'POST':
        f = request.files['the_file']
        if f.filename.find(".pdf") != -1:
            text = file_process.parsing_pdf(f)
            ids = raw_texts = [text[:100]]
            texts = np.array([preprocess_english(text, preprocess_config)])

            text_lens = np.array([len(texts[0])])
            batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]
            synthesize(model, restore_step, configs, vocoder, batchs, control_values)
            path = "./output/result/viet-tts"
            global path_to_file
            path_to_file = os.path.join(path, "{}.wav".format("out"))
            image_binary = open(os.path.join(path, "{}.png".format("out")), 'rb').read()
            response = make_response(base64.b64encode(image_binary))
            response.headers.set('Content-Type', 'image/png')
            response.headers.set('Content-Disposition', 'attachment', filename='result.png')
            return response
        elif f.filename.find(".docx") != -1:
            text = file_process.parsing_docx(f)
            print(text)
            return 'success'
        else:
            return 'err'
    else:
        def generate_data():
            global path_to_file
            f_wav = open(path_to_file, "rb")
            path_to_file = ''
            data = f_wav.read(1024)
            while data:
                yield data
                data = f_wav.read(1024)

        return Response(generate_data(), mimetype="audio/wav")


if __name__ == "__main__":
    app.run(host="localhost", port=8080)
