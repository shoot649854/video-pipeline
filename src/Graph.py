import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from typing import List

class graph:
    def __init__(self) -> None:
        pass

    def visualize_audio_amplitude(self, file_path: str, sr: int = 44100, n_fft: int = 2048, hop_length: int = None):
        y, _ = librosa.load(file_path, sr=sr)
        if hop_length is None:
            hop_length = n_fft // 2
        S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        D = librosa.amplitude_to_db(np.abs(S), ref=np.max)

        D_AVG = np.mean(D, axis=1)

        plt.figure(figsize=(10, 5))
        plt.bar(np.arange(D_AVG.shape[0]), D_AVG)
        x_ticks_positions = [n for n in range(0, n_fft // 2, n_fft // 16)]
        x_ticks_labels = [str(sr / n_fft * n) + 'Hz' for n in x_ticks_positions]
        plt.xticks(x_ticks_positions, x_ticks_labels)
        plt.xlabel('Frequency')
        plt.ylabel('dB')
        plt.title('Audio Amplitude Visualization')
        plt.show()

    def audio_analysis(self, file_path: str): 
        data, sampling_rate = librosa.load(file_path)

        # Create a time axis for plotting
        time_axis = np.arange(0, len(data)) / sampling_rate

        # Plot the waveform
        plt.figure(figsize=(12, 4))
        plt.plot(time_axis, data)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Audio Waveform')
        plt.show()
    
    def audio_analysis(self, file_paths: List[str]):
        for file_path in file_paths:
            data, sampling_rate = librosa.load(file_path)

            # Create a time axis for plotting
            time_axis = np.arange(0, len(data)) / sampling_rate

            # Plot the waveform
            plt.figure(figsize=(12, 4))
            plt.plot(time_axis, data)
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.title(f'Audio Waveform: {file_path}')
            plt.show()
    
    def chromagram(self, file_path: str):
        data, sampling_rate = librosa.load(file_path)
        y, sr = librosa.load(data)
        chromagram = librosa.feature.chroma_stft(y=y, sr=sr)

        plt.figure(figsize=(12, 4))
        librosa.display.specshow(chromagram, y_axis='chroma', x_axis='time')
        plt.colorbar()
        plt.title('Chromagram')
        plt.tight_layout()
        plt.show()