import os
import shutil
from typing import List
import logging
import librosa
import numpy as np
from scipy.io.wavfile import write
import webrtcvad
from pydub import AudioSegment

class Audio_webrtc:
	def __init__(self, sample_rate: int = 16000, frame_duration: int = 30, min_sec: int = 3 * 10**1, top_db: int = 30):
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

	def initialize_vad(self, aggressiveness: int = 3) -> webrtcvad.Vad:
		return webrtcvad.Vad(aggressiveness)

	def load_audio(self, audio_file_path: str) -> np.ndarray:
		print("Audio File: " + audio_file_path)
		audio, _ = librosa.load(audio_file_path, sr=self.sample_rate)
		return audio
	
	def extract_voiced_audio(self, audio: np.ndarray) -> np.ndarray:
		frame_size = int(self.sample_rate * self.frame_duration / 1000)
		voiced_frames = []

		for i in range(0, len(audio), frame_size):
			frame = audio[i:i + frame_size]

			if len(frame) == frame_size:
				if self.vad.is_speech((frame * 32767).astype(np.int16).tobytes(), self.sample_rate):
					voiced_frames.append(frame)

		return np.concatenate(voiced_frames, axis=0) if voiced_frames else np.array([])

	def split_and_save_chunks(self, voiced_audio: np.ndarray, output_path: str) -> List[str]:
		self.prepare_output_directory(output_path)
		non_silent_intervals = librosa.effects.split(voiced_audio, top_db=self.top_db)
		chunk_files = []
		current_start = None
		file_i = 0
		for interval in non_silent_intervals:
			start, end = interval
			if current_start is None:
				current_start = start

			if end - current_start >= self.min_sec * self.sample_rate:
				chunk_file_path = os.path.join(output_path, f"chunk_{file_i:08}.wav")
				chunk = voiced_audio[current_start:end]
				write(chunk_file_path, self.sample_rate, (chunk * 32767).astype(np.int16))
				print("File is written on " + chunk_file_path)
				chunk_files.append(chunk_file_path)
				current_start = None
				file_i += 1
			
			#if file_i > 2:
			#	break

		return chunk_files
	
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
		audio = self.load_audio(audio_file_path)
		voiced_audio = self.extract_voiced_audio(audio * 32767)
		return self.split_and_save_chunks(voiced_audio / 32767, output_path)