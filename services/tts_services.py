from scipy.io.wavfile import write
import time
import torch
import sys
sys.path.append("espnet")
from gtts import gTTS 
import os
# define device
import torch
device = torch.device("cpu")

# define E2E-TTS model
from argparse import Namespace
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import torch_load
from espnet.utils.dynamic_import import dynamic_import
from tacotron_cleaner.cleaners import custom_english_cleaners
from g2p_en import G2p
from parallel_wavegan.utils import load_model
trans_type = "phn"
dict_path = "downloads/en/tacotron2/data/lang_1phn/phn_train_no_dev_units.txt"
model_path = "downloads/en/tacotron2/exp/phn_train_no_dev_pytorch_train_pytorch_tacotron2.v3/results/model.last1.avg.best"
vocoder_path = "downloads/en/parallel_wavegan/ljspeech.parallel_wavegan.v2/checkpoint-400000steps.pkl"


def perform_tts(input_text):
    idim, odim, train_args = get_model_conf(model_path)
    model_class = dynamic_import(train_args.model_module)
    model = model_class(idim, odim, train_args)
    torch_load(model_path, model)
    model = model.eval().to(device)
    inference_args = Namespace(**{
        "threshold": 0.5,"minlenratio": 0.0, "maxlenratio": 10.0,
        # Only for Tacotron 2
        "use_attention_constraint": True, "backward_window": 1,"forward_window":3,
        # Only for fastspeech (lower than 1.0 is faster speech, higher than 1.0 is slower speech)
        "fastspeech_alpha": 1.0,
        })

    # define neural vocoder  
    fs = 22050
    vocoder = load_model(vocoder_path)
    vocoder.remove_weight_norm()
    vocoder = vocoder.eval().to(device)

    # define text frontend
    
    with open(dict_path) as f:
        lines = f.readlines()
    lines = [line.replace("\n", "").split(" ") for line in lines]
    char_to_id = {c: int(i) for c, i in lines}
    g2p = G2p()

    print('input : ',input_text)
    with torch.no_grad():
        start = time.time()
        x = frontend(input_text, g2p, char_to_id, idim)
        c, _, _ = model.inference(x, inference_args)
        y = vocoder.inference(c)
    rtf = (time.time() - start) / (len(y) / fs)
    print(f"RTF = {rtf:5f}")
    print(y)
    write("static/test.wav", fs, y.view(-1).cpu().numpy())
    # from IPython.display import display, Audio
    # display(Audio(y.view(-1).cpu().numpy(), rate=fs))

def frontend(text, g2p, char_to_id, idim):
    """Clean text and then convert to id sequence."""
    text = custom_english_cleaners(text)
    
    if trans_type == "phn":
        text = filter(lambda s: s != " ", g2p(text))
        text = " ".join(text)
        print(f"Cleaned text: {text}")
        charseq = text.split(" ")
    else:
        print(f"Cleaned text: {text}")
        charseq = list(text)
    idseq = []
    for c in charseq:
        if c.isspace():
            idseq += [char_to_id["<space>"]]
        elif c not in char_to_id.keys():
            idseq += [char_to_id["<unk>"]]
        else:
            idseq += [char_to_id[c]]
    idseq += [idim - 1]  # <eos>
    return torch.LongTensor(idseq).view(-1).to(device)

def perform_gtts(input_text):
    language = 'en'
    speech = gTTS(text = input_text, lang = language, slow = False)
    speech.save("static/test_google.mp3")