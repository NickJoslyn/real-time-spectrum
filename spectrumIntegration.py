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

    NDIM = len(xrTime)

    # Convert frequencyResolution and integrationTime to FFT representation
    samplesPerTransform = int((1/frequencyResolution)/TBIN)
    fftsPerIntegration = int(integrationTime * frequencyResolution)
    numberOfIntegrations = NDIM//(samplesPerTransform*fftsPerIntegration)

    waterfallData_x = np.zeros((1 + numberOfIntegrations, samplesPerTransform))
    waterfallData_y = np.zeros((1 + numberOfIntegrations, samplesPerTransform))

    for integration in range(numberOfIntegrations):
        summedFFT_x = np.zeros(samplesPerTransform)
        summedFFT_y = np.zeros(samplesPerTransform)

        for individualFFT in range(fftsPerIntegration):
            index = integration * fftsPerIntegration * samplesPerTransform

            summedFFT_x += np.abs(np.fft.fftshift(np.fft.fft(xrTime[index + individualFFT*samplesPerTransform: index + (individualFFT+1)*samplesPerTransform] + 1j*xiTime[index + individualFFT*samplesPerTransform: index + (individualFFT+1)*samplesPerTransform])))**2
            summedFFT_y += np.abs(np.fft.fftshift(np.fft.fft(yrTime[index + individualFFT*samplesPerTransform: index + (individualFFT+1)*samplesPerTransform] + 1j*yiTime[index + individualFFT*samplesPerTransform: index + (individualFFT+1)*samplesPerTransform])))**2

        waterfallData_x[integration, :] = summedFFT_x
        waterfallData_y[integration, :] = summedFFT_y

    currentLocation = fftsPerIntegration * numberOfIntegrations * samplesPerTransform
    samplesRemaining = NDIM-currentLocation

    if (samplesRemaining % samplesPerTransform == 0):
        FFTs_remaining = samplesRemaining/samplesPerTransform
    else:
        FFTs_remaining = 1 + samplesRemaining//samplesPerTransform

    remainingSum_x = np.zeros(samplesPerTransform)
    remainingSum_y = np.zeros(samplesPerTransform)

    for individualFFT in range(FFTs_remaining):
        remainingSum_x += np.abs(np.fft.fftshift(np.fft.fft(xrTime[currentLocation + individualFFT*samplesPerTransform:currentLocation + (individualFFT+1)*samplesPerTransform] + 1j*xiTime[currentLocation + individualFFT*samplesPerTransform:currentLocation + (individualFFT+1)*samplesPerTransform], samplesPerTransform)))**2
        remainingSum_y += np.abs(np.fft.fftshift(np.fft.fft(yrTime[currentLocation + individualFFT*samplesPerTransform:currentLocation + (individualFFT+1)*samplesPerTransform] + 1j*yiTime[currentLocation + individualFFT*samplesPerTransform:currentLocation + (individualFFT+1)*samplesPerTransform], samplesPerTransform)))**2

    waterfallData_x[-1, :] = remainingSum_x
    waterfallData_y[-1, :] = remainingSum_y


    plt.figure()
    plt.imshow(waterfallData_x, cmap = 'viridis', aspect = 'auto', extent = [lowerBound, upperBound, 0, integrationTime * numberOfIntegrations])
    plt.title("Waterfall Plot: Channel " + str(CHANNEL) + ": X Polarization")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Time")
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.imshow(waterfallData_y, cmap = 'viridis', aspect = 'auto', extent = [lowerBound, upperBound, 0, integrationTime * numberOfIntegrations])
    plt.title("Waterfall Plot: Channel " + str(CHANNEL) + ": Y Polarization")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Time")
    plt.colorbar()
    plt.show()
