# Nicholas Joslyn
# Breakthrough Listen UC Berkeley SETI Intern 2018

# Functions to estimate BL RAW data spectra with integration times

import matplotlib.pyplot as plt
import numpy as np

def RAW_PSD(RAW_CHANNEL, CHANNEL, centerFrequency, CHAN_BW, TBIN, FFT_size=512):

    # Divide channel into polarizations
    xrTime = RAW_CHANNEL[:,0]
    xiTime = RAW_CHANNEL[:,1]
    yrTime = RAW_CHANNEL[:,2]
    yiTime = RAW_CHANNEL[:,3]

    # Variables for plotting
    samplingFrequency = 1/TBIN
    lowerBound = centerFrequency + CHAN_BW/2
    upperBound = centerFrequency - CHAN_BW/2
    numberTicks = 7

    pxx, freqs_x = plt.psd(xrTime + 1j*xiTime, FFT_size, samplingFrequency)
    plt.title("Channel " + str(CHANNEL) + ": X Polarization")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("PSD: FFT Size " + str(FFT_size) + " (dB)")
    plt.xticks(np.linspace(min(freqs_x), max(freqs_x), numberTicks), np.round(np.linspace(lowerBound, upperBound, numberTicks), 1))
    plt.show()

    pxy, freqs_y = plt.psd(yrTime + 1j*yiTime, FFT_size, samplingFrequency)
    plt.title("Channel " + str(CHANNEL) + ": Y Polarization")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("PSD: FFT Size " + str(FFT_size) + " (dB)")
    plt.xticks(np.linspace(min(freqs_y), max(freqs_y), numberTicks), np.round(np.linspace(lowerBound, upperBound, numberTicks), 1))
    plt.show()


def RAW_spectrogram(RAW_CHANNEL, CHANNEL, centerFrequency, CHAN_BW, TBIN, FFT_size=512):

    # Divide channel into polarizations
    xrTime = RAW_CHANNEL[:,0]
    xiTime = RAW_CHANNEL[:,1]
    yrTime = RAW_CHANNEL[:,2]
    yiTime = RAW_CHANNEL[:,3]

    # Variables for plotting
    samplingFrequency = 1/TBIN
    lowerBound = centerFrequency + CHAN_BW/2
    upperBound = centerFrequency - CHAN_BW/2
    numberTicks = 7

    _, freqs_x, _, _ = plt.specgram(xrTime + 1j*xiTime, FFT_size, samplingFrequency)
    plt.title("Channel " + str(CHANNEL) + ": X Polarization")
    plt.ylabel("Frequency (MHz)")
    plt.yticks(np.linspace(min(freqs_x), max(freqs_x), numberTicks), np.round(np.linspace(lowerBound, upperBound, numberTicks), 1))
    plt.xlabel ("Time (s)")
    cb = plt.colorbar()
    cb.set_label("dB")
    plt.show()

    _, freqs_y, _, _ = plt.specgram(yrTime + 1j*yiTime, FFT_size, samplingFrequency)
    plt.title("Channel " + str(CHANNEL) + ": Y Polarization")
    plt.yticks(np.linspace(min(freqs_y), max(freqs_y), numberTicks), np.round(np.linspace(lowerBound, upperBound, numberTicks), 1))
    plt.ylabel("Frequency (MHz)")
    plt.xlabel("Time (s)")
    cb = plt.colorbar()
    cb.set_label("dB")
    plt.show()


def RAW_waterfall(RAW_CHANNEL, CHANNEL, centerFrequency, CHAN_BW, TBIN, frequencyResolution, integrationTime):

    # Divide channel into polarizations
    xrTime = RAW_CHANNEL[:,0]
    xiTime = RAW_CHANNEL[:,1]
    yrTime = RAW_CHANNEL[:,2]
    yiTime = RAW_CHANNEL[:,3]

    # Variables for plotting
    samplingFrequency = 1/TBIN
    lowerBound = centerFrequency + CHAN_BW/2
    upperBound = centerFrequency - CHAN_BW/2
    numberTicks = 7

    # Convert frequencyResolution and integrationTime to FFT representation
    samplesPerTransform = int((1/frequencyResolution)/TBIN)
    fftsPerIntegration = int(integrationTime * frequencyResolution)
    print(len(xrTime)/(samplesPerTransform*fftsPerIntegration))
    numberOfIntegrations = len(xrTime)//(samplesPerTransform*fftsPerIntegration)
    print(numberOfIntegrations)
    waterfallData = np.zeros((numberOfIntegrations, samplesPerTransform))
    # add 1 to number of integrations to get the remaining samples
    for integration in range(numberOfIntegrations):
        summedFFT = np.zeros(samplesPerTransform)
        for individualFFT in range(fftsPerIntegration):
            index = integration * numberOfIntegrations * samplesPerTransform

            FFTxPol = np.fft.fftshift(np.fft.fft(xrTime[index + individualFFT*samplesPerTransform: index + (individualFFT+1)*samplesPerTransform] + 1j*xiTime[index + individualFFT*samplesPerTransform: index + (individualFFT+1)*samplesPerTransform]))
            summedFFT += np.absolute(FFTxPol)**2
        waterfallData[integration, :] = summedFFT

    plt.figure()
    plt.imshow(waterfallData, cmap = 'viridis', aspect = 'auto', extent = [lowerBound, upperBound, 0, integrationTime * numberOfIntegrations])
    plt.title("Waterfall Plot")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Time")
    plt.show()
