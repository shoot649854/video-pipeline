from pydub import AudioSegment
from typing import List
import os

class Combine:
    def __init__(self):
        pass

    def save_audio(self, audio: AudioSegment, output_dir: str, filename: str):
        output_path = os.path.join(output_dir, filename)
        audio.export(output_path, format='wav')
        return output_path

    def combine_audiolist(self, audio_list: List[str], output_path: str) -> str:
        combined_audio = AudioSegment.empty()
        for audio_file in audio_list:
            audio_segment = AudioSegment.from_file(audio_file)
            combined_audio += audio_segment
        combined_audio.export(output_path, format='wav')
        return output_path