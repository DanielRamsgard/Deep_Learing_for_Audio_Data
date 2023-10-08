import librosa
import librosa.display
music_file_1 = "MP3s/music_1.mp3" # covert this to real time
music, sr = librosa.load(music_file_1) # we need to get this from a combination of ADC hardware and custom software for storage
FRAME_SIZE = 2048
HOP_SIZE = 1024
s_music = librosa.stft(music, n_fft=FRAME_SIZE, hop_length=HOP_SIZE) # returns spectrogram