import librosa
import numpy as np

class AudioAligner:
    def __init__(self):
        pass

    def align_two_audio(self, audio_file1, audio_file2):
        """
        Align two audio files containing similar voice content.
        
        Args:
            audio_file1 (str): Path to the first audio file.
            audio_file2 (str): Path to the second audio file.
            
        Returns:
            tuple: Aligned audio data for the two files.
        """
        # Load the audio files
        y1, sr1 = librosa.load(audio_file1)
        y2, sr2 = librosa.load(audio_file2)
        
        # Extract features (e.g., MFCCs)
        mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1)
        mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2)
        
        # Compute similarity matrix
        similarity_matrix = librosa.segmented.cross_similarity_segmented(mfcc1, mfcc2)
        
        # Find the optimal path
        path, score = librosa.sequence.dtw(similarity_matrix, global_constraints=True)
        
        # Align the audio files
        aligned_audio1 = librosa.sequence.from_path(y1, path[:, 0], sr1)
        aligned_audio2 = librosa.sequence.from_path(y2, path[:, 1], sr2)
        
        return aligned_audio1, aligned_audio2

    def align_multiple_audio(self, audio_files):
        """
        Align multiple audio files containing similar voice content.
        
        Args:
            audio_files (list): List of paths to audio files.
            
        Returns:
            list: Aligned audio data for all the files.
        """
        # Load the first audio file
        y1, sr1 = librosa.load(audio_files[0])
        aligned_audios = [y1]
        
        # Align each subsequent audio file to the first one
        for audio_file in audio_files[1:]:
            y, sr = librosa.load(audio_file)
            aligned_audio, _ = self.align_two_audio(audio_files[0], audio_file)
            aligned_audios.append(aligned_audio)
        
        return aligned_audios