import os
import sounddevice as sd
import soundfile as sf
import tempfile
import pyttsx3
from openai import OpenAI
from pydub import AudioSegment
from pydub.playback import play


class VoiceAssistant:
    def __init__(self, api_key, model="whisper-1", tts_engine=None):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.fs = 16000  # Sampling frequency
        self.tts_engine = tts_engine or pyttsx3.init()

    def record_audio(self, duration=5):
        print("🎤 Speak now...")
        recording = sd.rec(int(duration * self.fs), samplerate=self.fs, channels=1)
        sd.wait()
        # print("✅ Recording complete.")

        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(temp_wav.name, recording, self.fs)
        return temp_wav.name

    def speech_to_text(self, duration=5, language="fa"):
        audio_path = self.record_audio(duration)
        with open(audio_path, "rb") as audio_file:
            transcript = self.client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
                language=language
            )
        print("📝 Transcription:", transcript.text)
        return transcript.text

    def text_to_speech(self, text: str, voice: str = "onyx", model: str = "tts-1", filename: str = None):
        if not filename:
            temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            filename = temp_file.name

        response = self.client.audio.speech.create(
            model=model,
            voice=voice,
            input=text
        )

        # with open(filename, "wb") as f:
        #     f.write(response.content)
        #
        # playsound.playsound(filename) # sudo apt-get install python3-gi
        # os.remove(filename)

        with open(filename, "wb") as f:
            f.write(response.content)

        self.play_audio(filename)
        os.remove(filename)

    @staticmethod
    def play_audio(filename):
        audio = AudioSegment.from_file(filename, format="mp3")  # sudo apt install ffmpeg
        play(audio)
