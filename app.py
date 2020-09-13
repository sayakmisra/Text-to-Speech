import os
import requests
from flask import Flask, render_template, request
app = Flask(__name__)
from services import tts_services

@app.route('/', methods=['GET', 'POST'])
def hello():
    return render_template('index.html')

@app.route('/tts', methods=['GET', 'POST'])
def tts():
    # print("Inside TTS..")
    render_template('play_audio.html', text="Loading..")
    if request.method == "POST":
        # get text that the user has entered
        text = request.form['text']
        tts_services.perform_tts(text)
        # tts_services.perform_gtts(text)
        print(text)
    return render_template('play_audio.html', text=text)


if __name__ == '__main__':
    app.run()

    