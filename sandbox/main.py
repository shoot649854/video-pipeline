import os
import sys
from dotenv import load_dotenv
import glob
from pydub import AudioSegment

PARENT_DIR = os.getcwd()
SRC_DIR = os.path.join(PARENT_DIR, 'src')
sys.path.append(SRC_DIR)

from Processing import processing
from Audio_webrtc import Audio_webrtc
from Audio_speechbrain import Audio_speechbrain
from NoiseEliminator import NoiseEliminator
from Graph import graph

input_path = glob.glob(os.path.join(PARENT_DIR, 'input', '*'))

if __name__ == "__main__":
  splitter = Audio_webrtc(
    top_db=20,
  )
  eliminator = NoiseEliminator()
  audio_file_path = input_path[0]
  output_path = "../output"

  chunks = splitter.process_audio(audio_file_path, output_path)

  for chunk in chunks:
    eliminator.noise_suppression(chunk, chunk)


  combined_audio_path = os.path.join(output_path, "combined_audio.wav")
  combined_audio = AudioSegment.empty()
  for chunk in chunks:
      combined_audio += AudioSegment.from_file(chunk)

  combined_audio.export(combined_audio_path, format='wav')