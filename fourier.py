import numpy as np
from scipy.fftpack import dct
import librosa as ls
import sys

def take_signal(signal, seconds, sample_rate):
    return signal[: int(sample_rate*seconds)]

def pre_emphasis(signal, *, emph_coef=0.95):
    return np.append(signal[0], signal[1:] - emph_coef*signal[:-1])

def get_frames(signal, *, frame_size=400, frame_padding=160):
    """
    Inputs: 
    signal -- set of amplitudes
    frame_size -- length of the frames to be parted
    frame_padding -- length of the padding 

    Outputs: 
    parted set of frames with applied Hamming window to them
    """
    # Calculating the max index for iteration.  
    frames_limit = (len(signal) - frame_padding)//frame_size + 1

    result_partition = np.array(signal[: frame_size])

    # Parting till it can be parted into vector of frame_size elements.  
    for index in range(1, frames_limit):
        down_border = index*frame_size - frame_padding
        result_partition = np.vstack((result_partition, signal[down_border : down_border+frame_size]))
    
    # Filling the end of signal with zeros till the size of result array is frame_size.  
    signal_padding = signal[frames_limit*frame_size-frame_padding :]
    result_partition = np.vstack((result_partition, np.append(signal_padding, np.zeros(frame_size-len(signal_padding)))))

    # Hamming window.  
    result_partition *= np.hamming(frame_size)

    return result_partition

def get_power_spectrum(frames, NFFT):
    return np.absolute(np.fft.rfft(frames, NFFT))**2 / NFFT

def filter_banks(nfilt, NFFT, sample_rate):
    low_freq_limit = 0
    high_freq_limit = (2595 * np.log10(1 + (sample_rate/2) / 700))
    mels = np.linspace(low_freq_limit, high_freq_limit, nfilt + 2)
    hertz = (700 * (10**(mels / 2595) - 1))

    f = np.floor((NFFT + 1) * hertz / sample_rate)
    # Frequencies to dict.  
    hertz = {i: hertz[i] for i in range(len(hertz))}

    H = np.zeros((nfilt, int(np.floor(NFFT/2 + 1))))

    for m in range(1, nfilt + 1):
        f_prev, f_curr, f_next = f[m-1], f[m], f[m+1]

        for k, hertz_value in hertz.items():
            if hertz_value < f_prev:
                H[m-1, k] = 0
            elif hertz_value >= f_prev and hertz_value < f_curr:
                H[m-1, k] = (hertz_value - f_prev)/(f_curr - f_prev)
            elif hertz_value >= f_curr and hertz_value <= f_next:
                H[m-1, k] = (f_next - hertz_value)/(f_next - f_curr)
            else:
                H[m-1, k] = 0
    return H    

def get_mfcc(power, filter, mfcc_amount):
    filt = np.dot(power, filter.T)
    # Replace zeros with 1e-10 to take ln of the filt.  
    filt = np.where(filt == 0, 1e-10, filt)
    filt = np.log(filt)

    return dct(filt, type=2, axis=1, norm='ortho')[:, 2 : mfcc_amount+2]

def mfcc_alg(signal, *, sample_rate=16000, nfilt=40, NFFT=512, mfcc=16):
    signal = take_signal(signal, 10, sample_rate)
    emphasized_signal = pre_emphasis(signal, emph_coef=0.97)
    frames = get_frames(emphasized_signal)
    power = get_power_spectrum(frames, NFFT)

    H = filter_banks(nfilt, NFFT, sample_rate)
    mfcc_coefs = get_mfcc(power, H, mfcc)

    normalized_mfcc = np.mean(mfcc_coefs.T, axis=1)

    return list(normalized_mfcc)

def difference(coefs1, coefs2):
    mfcc1 = np.array(coefs1)
    mfcc2 = np.array(coefs2)

    return np.sqrt(np.sum((mfcc1 - mfcc2)**2))

def main():
    #sample_rate = 16000
    #speaker1, _ = ls.load(sys.path[0] + r'\Data\test1.wav', sr=sample_rate)
    #speaker3, _ = ls.load(sys.path[0] + r'\Data\test5.wav', sr=sample_rate)
    
    speaker1, _ = ls.load(sys.path[0] + r'\Data\test2.wav', sr=16000)
    speaker2, _ = ls.load(sys.path[0] + r'\Data\test3.wav', sr=16000)
    
    #print(np.array(mfcc_alg(speaker1)))
    #print(np.array(mfcc_alg(speaker2)))
    
    print(difference(mfcc_alg(speaker1), mfcc_alg(speaker2)))

if __name__ == "__main__":
    main()
