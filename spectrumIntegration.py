# Nicholas Joslyn
# Breakthrough Listen UC Berkeley SETI Intern 2018

# Functions to estimate BL RAW data spectra with integration times

import matplotlib.pyplot as plt
import numpy as np

def RAW_PSD(RAW_CHANNEL, CHANNEL, centerFrequency, CHAN_BW, TBIN, FFT_size=512):
    """
    Plot the PSD of complex, dual-polarized voltage time series.

    The Power Spectral Density shows the power of a signal present in time
    series data. Pyplot defines a function to plot the PSD. This function
    wraps it conveniently to match the other functions while allowing for
    different FFT lengths.

    Parameters:
    RAW_CHANNEL (2D Array):     The time series data for the channel
    CHANNEL (int):              The channel to be analyzed
    centerFrequency (float):    Center frequency of the channel (MHz)
    CHAN_BW (float):            Bandwidth of the channel (MHz)
    TBIN (float):               Sampling period (seconds)
    FFT_size (int):             Number of samples in each FFT (default, 512)

    Returns:
    Nothing
    """


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
    """
    Plot the spectrogram of complex, dual-polarized voltage time series.

    A spectrogram displays time (x-axis), frequency (y-axis), and intensity
    (colormap) on one plot. Pyplot defines a function to plot the spectrogram of
    time series data. This function wraps it conveniently to match the other
    functions while allowing for different FFT lengths.

    Parameters:
    RAW_CHANNEL (2D Array):     The time series data for the channel
    CHANNEL (int):              The channel to be analyzed
    centerFrequency (float):    Center frequency of the channel (MHz)
    CHAN_BW (float):            Bandwidth of the channel (MHz)
    TBIN (float):               Sampling period (seconds)
    FFT_size (int):          Number of samples in each FFT (default, 512)

    Returns:
    Nothing
    """


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
    """
    Plot the waterfall plot of complex, dual-polarized voltage time series.

    A waterfall plot is similar to a spectrogram. The waterfall plot displays
    frequency (x-axis), time (y-axis), and intensity (colormap) on one plot.
    This function allows for custom time and frequency resolution to be used
    on a complex dual-polarized time series.

    Parameters:
    RAW_CHANNEL (2D Array):         The time series data for the channel
    CHANNEL (int):                  The channel to be analyzed
    centerFrequency (float):        Center frequency of the channel (MHz)
    CHAN_BW (float):                Bandwidth of the channel (MHz)
    TBIN (float):                   Sampling period (seconds)
    frequencyResolution (float):    Desired FFT bin width (Hz)
    integrationTime (float):        Desired FFT integration time (s)

    Returns:
    Nothing
    """
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


def RAW_integratedFFT(RAW_CHANNEL, CHANNEL, centerFrequency, CHAN_BW, TBIN, frequencyResolution):
    """
    Plot the integrated FFT of complex, dual-polarized voltage time series.

    The integrated FFT is a series of summed FFTs with desired frequency
    resolution. This allows weaker, periodic signals to rise above the noise
    floor. This function allows for custom frequency resolution on time
    series data.

    Parameters:
    RAW_CHANNEL (2D Array):         The time series data for the channel
    CHANNEL (int):                  The channel to be analyzed
    centerFrequency (float):        Center frequency of the channel (MHz)
    CHAN_BW (float):                Bandwidth of the channel (MHz)
    TBIN (float):                   Sampling period (seconds)
    frequencyResolution (float):    Desired FFT bin width (Hz)

    Returns:
    Nothing
    """

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

    # Convert frequencyResolution to FFT representation
    samplesPerTransform = int((1/frequencyResolution)/TBIN)
    numberOfFFTs = NDIM//samplesPerTransform
    if (NDIM%samplesPerTransform != 0):
        numberOfFFTs+=1

    integrated_x = np.zeros(samplesPerTransform)
    integrated_y = np.zeros(samplesPerTransform)

    for individualFFT in range(numberOfFFTs):
        integrated_x += np.abs(np.fft.fftshift(np.fft.fft(xrTime[individualFFT*samplesPerTransform:(individualFFT+1)*samplesPerTransform] + 1j*xiTime[individualFFT*samplesPerTransform:(individualFFT+1)*samplesPerTransform], samplesPerTransform)))**2
        integrated_y += np.abs(np.fft.fftshift(np.fft.fft(yrTime[individualFFT*samplesPerTransform:(individualFFT+1)*samplesPerTransform] + 1j*yiTime[individualFFT*samplesPerTransform:(individualFFT+1)*samplesPerTransform], samplesPerTransform)))**2

    plt.plot(np.linspace(lowerBound, upperBound, samplesPerTransform), integrated_x)
    plt.title("Channel " + str(CHANNEL) + ": X Polarization")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Integrated FFT: Power")
    plt.show()

    plt.plot(np.linspace(lowerBound, upperBound, samplesPerTransform), integrated_y)
    plt.title("Channel " + str(CHANNEL) + ": Y Polarization")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Integrated FFT: Power")
    plt.show()
