```bash
torch-model-archiver --model-name fastspeech2_hifigan --version 1.0  --serialized-file script_vocoder.pt --handler model_handler.py --extra-files script_model.pt,phones.txt,viet-tts-lexicon.txt
torchserve --start --model-store model_store --models fastspeech2_hifigan.mar --ts-config config.properties
