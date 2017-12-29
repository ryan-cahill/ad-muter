import os
import pydub
import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
import math
import h5py
import gc

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

    #timebins, freqbins = np.shape(ims)

    #plt.figure(figsize=(15, 7.5))
    #plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    #plt.colorbar()

    #plt.xlabel("time (s)")
    #plt.ylabel("frequency (hz)")
    #plt.xlim([0, timebins-1])
    #plt.ylim([0, freqbins])

    #xlocs = np.float32(np.linspace(0, timebins-1, 5))
    #plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate])
    #ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
    #plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])

    #if plotpath:
    #    plt.savefig(plotpath, bbox_inches="tight")
    #else:
    #    plt.show()
    #plt.clf()
    return ims


def getAllAudioData(directory):
    all_audio_data = []
    allMaximums = []
    audio_filenames = os.listdir(directory)
    for filename in audio_filenames:#[0:2]:#REMOVE THIS ARRAY INDEX
        new_wav_file = directory[0:len(directory) - 1] + "-wav/" + filename[0:len(filename) - 4] + ".wav"

        if not os.path.isfile(new_wav_file):
            test_audio_file_edm = directory + filename
            mp3 = pydub.AudioSegment.from_mp3(test_audio_file_edm)
            mp3.export(new_wav_file, format="wav")#should check to see if file exists first for efficiency

        print "ADDED " + new_wav_file
        audioData = plotstft(new_wav_file)
        all_audio_data.append(audioData)

        allMaximums.append(audioData.max(axis=1))
        distilledMaximums = []
        for maximum in allMaximums:
            distilledMaximums.append(maximum.max(axis=0))
        fullMaximum = np.array(distilledMaximums).max(axis=0)
    return all_audio_data, fullMaximum

def sliceAudio(songArray):
    groupSlices = np.array([])
    sliceWidth = 200
    for song in songArray:
        lengthToUse = len(song) - (len(song) % sliceWidth)
        song = song[0:lengthToUse]
        if len(groupSlices) == 0:
            groupSlices = np.split(song, len(song) / sliceWidth)
        else:
            groupSlices = np.concatenate((groupSlices, np.split(song, len(song) / sliceWidth)), axis=0)
    return groupSlices

def cleanseAudioSlices(rawSlices):
    rawSlices[rawSlices == float('inf')] = 0
    rawSlices[rawSlices == float('-inf')] = 0

    return rawSlices

def getFlattenedSlices(cleansedSlices, oneHotLabel):
    flattenedSlices = []
    labels = []
    for slicesList in cleansedSlices:
        flattenedSlices.append(slicesList.reshape(slicesList.shape[0], slicesList.shape[1], 1))
        labels.append(oneHotLabel)
    return flattenedSlices, labels

TOTAL_CLASSES = 2
test_ads_path = "/home/ryan/Downloads/ad-muter/test-commercials/"
test_edm_path = "/home/ryan/Downloads/ad-muter/test-edm/"
ad_audio, adMaximum = getAllAudioData(test_ads_path)
edm_audio, edmMaximum = getAllAudioData(test_edm_path)
ad_slices = sliceAudio(np.asarray(ad_audio))
edm_slices = sliceAudio(np.asarray(edm_audio))

cleansedAdSlices = cleanseAudioSlices(np.asarray(ad_slices))
cleansedEdmSlices = cleanseAudioSlices(np.asarray(edm_slices))

oneHotLabelAd = np.zeros(2)
oneHotLabelAd[0] = 1
flattenedAdSlices, adLabels = getFlattenedSlices(np.array(cleansedAdSlices), oneHotLabelAd)
oneHotLabelEdm = np.zeros(2)
oneHotLabelEdm[1] = 1
flattenedEdmSlices, edmLabels = getFlattenedSlices(np.asarray(cleansedEdmSlices), oneHotLabelEdm)

normalizedEdmSlices = np.asarray(flattenedEdmSlices) / edmMaximum
normalizedAdSlices = np.asarray(flattenedAdSlices) / adMaximum

allSlices = np.concatenate((normalizedAdSlices, normalizedEdmSlices), axis=0)
allLabels = np.concatenate((adLabels, edmLabels), axis=0)

musicDataSet = h5py.File("musicData.hdf5", "w", libver="latest")
musicDataSet.create_dataset("allSlices", allSlices.shape, dtype='f', data=allSlices)
musicDataSet.create_dataset("allLabels", allLabels.shape, dtype='i', data=allLabels)
musicDataSet.close()