import os
import pydub
import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
import math
import pickle

""" short time fourier transform of audio signal """
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.int(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(np.int(frameSize/2.0)), sig)
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)

""" scale frequency axis logarithmically """
def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,scale[i]:], axis=1)
        else:
            newspec[:,i] = np.sum(spec[:,scale[i]:scale[i+1]], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[scale[i]:])]
        else:
            freqs += [np.mean(allfreqs[scale[i]:scale[i+1]])]

    return newspec, freqs

""" plot spectrogram"""
def plotstft(audiopath, binsize=2**10, plotpath=None, colormap="jet"):
    samplerate, samples = wav.read(audiopath)
    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)
    ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel (non-complex sshow)
    return ims


def get_all_audio_data(directory):
    all_audio_data = []
    all_maximums = []
    audio_filenames = os.listdir(directory)
    max_files_loaded = 100
    for filename in audio_filenames[0:(len(audio_filenames), max_files_loaded)[len(audio_filenames) >= max_files_loaded]]:#REMOVE THIS ARRAY INDEX
        new_wav_file = directory[0:len(directory) - 1] + "-wav/" + filename[0:len(filename) - 4] + ".wav"

        if not os.path.isfile(new_wav_file):
            test_audio_file_edm = directory + filename
            mp3 = pydub.AudioSegment.from_mp3(test_audio_file_edm)
            mp3.export(new_wav_file, format="wav")

        print "ADDED " + new_wav_file
        audio_data = plotstft(new_wav_file)

        audio_data[audio_data == float('inf')] = 0
        audio_data[audio_data == float('-inf')] = 0

        all_audio_data.append(audio_data)

        all_maximums.append(audio_data.max(axis=1))
        distilled_maximums = []
        for maximum in all_maximums:
            distilled_maximums.append(maximum.max(axis=0))
        full_maximum = np.array(distilled_maximums).max(axis=0)
    return np.asarray(all_audio_data), full_maximum

test_ads_path = "/home/ryan/Downloads/ad-muter/test-commercials/"
test_edm_path = "/home/ryan/Downloads/ad-muter/test-edm/"
ad_audio, ad_maximum = get_all_audio_data(test_ads_path)
edm_audio, edm_maximum = get_all_audio_data(test_edm_path)

extra_data = {"adMaximum": ad_maximum, "edmMaximum": edm_maximum}
with open('adAudio.pickle', 'wb') as handle:
    pickle.dump(ad_audio, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('edmAudio.pickle', 'wb') as handle:
    pickle.dump(edm_audio, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('extraData.pickle', 'wb') as handle:
    pickle.dump(extra_data, handle, protocol=pickle.HIGHEST_PROTOCOL)