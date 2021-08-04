import sounddevice as sd
from scipy.io.wavfile import write
import sys
from db import callSelectAll
from fourier import *

def rec_voice():
    fs = 16000

    record_voice = sd.rec(int(3.5*fs), samplerate=fs, channels=2)
    sd.wait()
    write(sys.path[0] + r"\Data\output.wav", fs, record_voice)

def main():
    key_speaker, _ = ls.load(sys.path[0] + r'\Data\output.wav')
    key_coefs = mfcc_alg(key_speaker)

    all_data = callSelectAll()

    distances = []
    for data in all_data:
        distances.append(difference(eval(data['coefs']), key_coefs))

    print(distances)
    
if __name__ == "__main__":
    main()
