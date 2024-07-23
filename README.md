# Speech-Emotion-Recognition-with-Response

## Real-Time Emotion Recognition and Response System

This project is a real-time emotion recognition and response system built with Django. It records audio, recognizes emotions from speech, generates text responses based on detected emotions, synthesizes speech from the generated text, and plays back the response. The system uses various machine learning models and libraries for emotion recognition, speech-to-text transcription, text generation, and text-to-speech synthesis.

### Features
- **Audio Recording**: Records audio using a microphone.
- **Emotion Recognition**: Predicts emotions from the recorded audio using a pre-trained emotion recognition model.
- **Speech-to-Text Transcription**: Transcribes recorded audio to text using Google Speech Recognition.
- **Text Response Generation**: Generates text responses based on the recognized emotion and transcribed text using the BlenderBot model.
- **Text-to-Speech Synthesis**: Synthesizes speech from the generated text using Tacotron2 and HiFi-GAN models.
- **Real-Time Feedback**: Plays back the synthesized speech to the user.
- **Web Interface**: Provides a simple web interface to start recording and check the status.

### Technologies and Libraries Used
- **Backend**: Django
- **Audio Processing**: PyAudio, sounddevice, librosa, wave, torchaudio
- **Speech Recognition**: SpeechRecognition (Google Speech Recognition API)
- **Emotion Recognition**: Keras, TensorFlow
- **Text Generation**: Transformers (BlenderBot)
- **Text-to-Speech**: SpeechBrain (Tacotron2, HiFi-GAN)
- **Web Interface**: HTML/CSS/JavaScript (Django templates)


