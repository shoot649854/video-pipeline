from pydub import AudioSegment


# len 
class processing:
    def __init__(self) -> None:
        pass

    def volume(self, audio_file: str, audio_control: int):
        audio = AudioSegment.from_file(audio_file)
        return audio + audio_control 