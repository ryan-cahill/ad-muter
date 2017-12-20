import os
import pydub
import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
import math
import h5py

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
    audio_filenames = os.listdir(directory)
    for filename in audio_filenames:#[0:2]:#REMOVE THIS ARRAY INDEX
        test_audio_file_edm = directory + filename
        mp3 = pydub.AudioSegment.from_mp3(test_audio_file_edm)

        new_wav_file = directory[0:len(directory) - 1] + "-wav/" + filename[0:len(filename) - 4] + ".wav"
        mp3.export(new_wav_file, format="wav")#should check to see if file exists first for efficiency

        print "ADDED " + new_wav_file
        all_audio_data.append(plotstft(new_wav_file))
    return all_audio_data

def loadDataArrays(adPath, edmPath):
    ad_audio = getAllAudioData(adPath)
    edm_audio = getAllAudioData(edmPath)
    return ad_audio, edm_audio

def sliceAudio(songArray):
    groupSlices = []
    sliceWidth = 20
    for song in songArray:
        i = 0
        while i < len(song) - sliceWidth:
            groupSlices.append(song[i:i + sliceWidth])
            i += sliceWidth
    return groupSlices

def cleanseAudioSlices(rawSlices):
    cleansedSlices = []
    for slice in rawSlices:
        foundInvalidData = False
        for slicePart in slice:
            if float('Inf') in slicePart or -(float('Inf')) in slicePart:
                foundInvalidData = True
        if not foundInvalidData:
            cleansedSlices.append(slice)
    return cleansedSlices

def getFlattenedSlices(cleansedSlices, labelNumber):
    flattenedSlices = []
    labels = []
    for slicesList in cleansedSlices:
        flattenedSlices.append(slicesList)
        labels.append(labelNumber)
    return flattenedSlices, labels

test_ads_path = "/home/ryan/Downloads/ad-muter/test-commercials/"
test_edm_path = "/home/ryan/Downloads/ad-muter/test-edm/"
ad_audio, edm_audio = loadDataArrays(test_ads_path, test_edm_path)
ad_slices = sliceAudio(ad_audio)
edm_slices = sliceAudio(edm_audio)
del ad_audio
del edm_audio
cleansedAdSlices = cleanseAudioSlices(ad_slices)
cleansedEdmSlices = cleanseAudioSlices(edm_slices)
del ad_slices
del edm_slices
flattenedAdSlices, adLabels = getFlattenedSlices(np.array(cleansedAdSlices), 0)
flattenedEdmSlices, edmLabels = getFlattenedSlices(np.array(cleansedEdmSlices), 1)
del cleansedAdSlices
del cleansedEdmSlices

allSlices = np.concatenate((flattenedAdSlices, flattenedEdmSlices), axis=0)
allLabels = np.concatenate((adLabels, edmLabels), axis=0)

musicDataSet = h5py.File("musicData.hdf5", "w")
musicDataSet.create_dataset("allSlices", allSlices.shape, dtype='f', data=allSlices)
musicDataSet.create_dataset("allLabels", allLabels.shape, dtype='i', data=allLabels)
musicDataSet.close()