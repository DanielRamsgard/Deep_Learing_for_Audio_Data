import librosa
import numpy as np

file = "MP3s/music_1.mp3" # covert this to real time
FRAME_SIZE = 2048
HOP_SIZE = 1024

# waveform
signal, sr = librosa.load(file, sr=22050)

# fft -> spectrum
fft = np.fft.fft(signal)

magnitude = np.abs(fft) # contirbution of each frequency to overall sound
frequency = np.linspace(0, sr, len(magnitude)) # returns evenly spaced numbers in an interval

    # nyquist frequency theorem
left_magnitude = magnitude[:int(len(magnitude)/2)]
left_frequency = frequency[:int(len(frequency)/2)]

# stft -> spectrogram
stft = librosa.core.stft(signal, n_fft=FRAME_SIZE, hop_length=HOP_SIZE, window='hann')

spectrogram = np.abs(stft) # returns spectrogram

mean = np.mean(spectrogram)
std_dev = np.std(spectrogram)
standardized_spectrogram = (spectrogram - mean) / std_dev # returns standardized spectrogram

log_standardized_spectrogram = librosa.amplitude_to_db(standardized_spectrogram) # returns log-standardzied-spectrogram

# MFCCs

MFCCs = librosa.feature.mfcc(signal, n_fft=FRAME_SIZE, hop_length=HOP_SIZE, n_mfcc=13)