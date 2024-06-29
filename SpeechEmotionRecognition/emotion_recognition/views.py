import numpy as np
import librosa
import warnings
import pyaudio
import wave
import sounddevice as sd
import speech_recognition as sr
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from speechbrain.inference.TTS import Tacotron2
from speechbrain.inference.vocoders import HIFIGAN
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from keras.models import load_model
import torchaudio
import threading
import json
from django.http import JsonResponse


warnings.filterwarnings('ignore')

# Load models
ser_model = load_model('C:\\Users\\GUNA\\Desktop\\python\\ser_model.keras')
tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
blenderbot_model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts") 
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder")

recording_status = {"status": "idle", "emotion": "", "transcription": "", "response": ""}

def record_audio(duration, sample_rate=16000, channels=1):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=1024)
    print("Recording...")
    frames = []
    for _ in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)
    print("Recording finished")
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded audio as a WAV file
    output_file = 'output.wav'
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

    print(f"Audio saved to {output_file}")
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)

    return audio_data, output_file

def extract_mfcc(audio, sample_rate, n_mfcc=40):
    mfcc = librosa.feature.mfcc(y=audio.astype(float), sr=sample_rate, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

def transcription(audio_file):
    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio = r.record(source)
            text = r.recognize_google(audio)
            print("Transcribed text:", text)
            return text
    except sr.UnknownValueError:
        print("Speech recognition could not understand the audio")
        return "null"
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None

def generate_response(emotion, user_input):
    if user_input == "null":
        return "Your speech was not recognized properly."
    context = f"The user feels {emotion}. They said '{user_input}'"
    inputs = tokenizer([context], return_tensors='pt')
    reply_ids = blenderbot_model.generate(**inputs, max_length=100, temperature=0.7, top_k=50, top_p=0.9)
    reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    return reply

def synthesize_speech(text):
    mel_output, mel_length, alignment = tacotron2.encode_text(text)
    waveforms = hifi_gan.decode_batch(mel_output)
    torchaudio.save('output.wav', waveforms.squeeze(1), 22050)

    waveform = waveforms.squeeze().detach().cpu().numpy()
    return waveform

def play_audio(waveform, sample_rate=22050):
    sd.play(waveform, samplerate=sample_rate)
    sd.wait()

def process_recording():
    global recording_status
    recording_status["status"] = "Recording"
    audio_data, output_file = record_audio(duration=5, sample_rate=16000)
    recording_status["status"] = "Recording completed"
    mfcc_features = extract_mfcc(audio_data, sample_rate=16000)
    mfcc_features = np.expand_dims(mfcc_features, axis=0)
    emotion_prediction = ser_model.predict(mfcc_features)
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
    predicted_emotion_index = np.argmax(emotion_prediction)
    predicted_emotion = emotion_labels[predicted_emotion_index]
    print(f"Predicted Emotion: {predicted_emotion}")
    transcription_text = transcription(output_file)
    response_text = generate_response(predicted_emotion, transcription_text)
    print(f"Generated Response: {response_text}")
    synthesized_speech = synthesize_speech(response_text)
    play_audio(synthesized_speech)
    recording_status = {
        "status": "completed",
        "emotion": predicted_emotion,
        "transcription": transcription_text,
        "response": response_text
    }

@csrf_exempt
def start_recording_view(request):
    global recording_status
    recording_status = {"status": "preparing", "emotion": "", "transcription": "", "response": ""}
    threading.Thread(target=process_recording).start()
    return JsonResponse({"message": "Preparing to record..."})

@csrf_exempt
def check_status_view(request):
    global recording_status
    return JsonResponse(recording_status)

def emotion_recognition(request):
    return render(request, 'index.html')
