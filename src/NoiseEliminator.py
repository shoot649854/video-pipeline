from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

class NoiseEliminator:
	def __init__(self):
		self.suppression_pipeline = pipeline(
			Tasks.acoustic_noise_suppression,
			model='damo/speech_frcrn_ans_cirm_16k'
		)

	def noise_suppression(self, audio_file_path: str, output_path: str):
		self.suppression_pipeline(audio_file_path, output_path=output_path)