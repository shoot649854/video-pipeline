import os
import shutil
from typing import List

import librosa
import numpy as np
from scipy.io.wavfile import write
from speechbrain.inference.VAD import VAD
from pydub import AudioSegment

class Audio_speechbrain:
    def __init__(self, sample_rate: int = 16000, frame_duration: int = 30, min_sec: int = 1, top_db: int = 30):
        """
        Initialize the Voice Activity Detector (VAD) with specified parameters.

        Parameters:
        - sample_rate (int): The sampling rate of the audio signal, in Hz. Default is 16000.
        - frame_duration (int): The duration of each audio frame in milliseconds. Default is 30 ms.
        - min_sec (int): The minimum duration of speech activity required to consider a segment valid, in seconds.
                        Default is 5 minutes (5 * 60 seconds).
        - top_db (int): The threshold for considering audio signal as silent. Default is 30 dB.

        Returns:
        - None

        Note:
        - The VAD (Voice Activity Detector) is initialized using the provided parameters.
        - Higher sample_rate may improve VAD accuracy but requires more computational resources.
        - Adjust frame_duration to balance between real-time processing and accuracy.
        - min_sec defines the minimum duration of speech activity to consider a segment valid.
        - top_db controls the threshold for considering audio signal as silent; decrease for more sensitivity.

        """
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.min_sec = min_sec
        self.top_db = top_db
        self.vad = self.initialize_vad()
              
    def initialize_vad(self) -> VAD:
        return VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="pretrained_models/vad-crdnn-libriparty")

    def load_audio(self, audio_file_path: str) -> np.ndarray:
        return librosa.load(audio_file_path, sr=self.sample_rate)[0]

    def extract_voiced_audio(self, audio: np.ndarray) -> np.ndarray:
        voiced_segments = self.vad.get_speech_segments(audio)
        voiced_audio = np.concatenate([audio[int(start * self.sample_rate):int(end * self.sample_rate)] for start, end, _ in voiced_segments])
        return voiced_audio

    def split_and_save_chunks(self, voiced_audio: np.ndarray, output_path: str) -> List[str]:
        self.prepare_output_directory(output_path)
        chunk_length = int(self.sample_rate * self.min_sec)
        num_chunks = len(voiced_audio) // chunk_length
        chunk_paths = []
        for i in range(num_chunks):
            chunk = voiced_audio[i * chunk_length: (i + 1) * chunk_length]
            chunk_path = os.path.join(output_path, f"chunk_{i}.wav")
            write(chunk_path, self.sample_rate, chunk)
            chunk_paths.append(chunk_path)
        return chunk_paths

    def prepare_output_directory(self, path: str):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
        
    def audio_conversion(self, audio_file_path: str, output_path: str) -> str:
        output_filename = os.path.splitext(os.path.basename(audio_file_path))[0] + '.wav'
        output_wav_path = os.path.join(output_path, output_filename)
        sound = AudioSegment.from_file(audio_file_path, format='m4a')
        sound.export(output_wav_path, format='wav')
        return output_wav_path

    def process_audio(self, audio_file_path: str, output_path: str) -> List[str]:
        if audio_file_path.endswith('.m4a'):
            audio_file_path = self.audio_conversion(audio_file_path, output_path)
        audio = self.load_audio(audio_file_path)
        voiced_audio = self.extract_voiced_audio(audio)
        return self.split_and_save_chunks(voiced_audio, output_path)