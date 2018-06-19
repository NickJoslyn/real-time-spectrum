# Nicholas Joslyn
# Breakthrough Listen UC Berkeley SETI Intern 2018

# Functions to estimate BL RAW data spectra

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

def RAW_periodogram(RAW_CHANNEL, CHANNEL, centerFrequency, CHAN_BW, TBIN, type = 'density'):
    """
    Two options to plot the periodogram of complex, dual-polarized voltage time series.

    The periodogram plots the relative strength of frequencies in time
    series data. This helps identify periodic signals. The default type, density,
    has units V**2/Hz and the spectrum type has units V**2.

    Parameters:
    RAW_CHANNEL (2D Array):     The time series data for the channel
    CHANNEL (int):              The channel to be analyzed
    centerFrequency (float):    Center frequency of the channel (MHz)
    CHAN_BW (float):            Bandwidth of the channel (MHz)
    TBIN (float):               Sampling period (seconds)
    type (string):              'density' (default) or 'spectrum'

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

    fx, periodox = signal.periodogram(xrTime + 1j*xiTime, samplingFrequency, scaling = type, return_onesided = False)

    plt.plot(fx, periodox)
    plt.xticks(np.linspace(min(fx), max(fx), numberTicks), np.round(np.linspace(lowerBound, upperBound, numberTicks), 1))
    plt.title("Channel " + str(CHANNEL) + ': X Polarization')
    plt.xlabel("Frequency (MHz)")
    if(type == 'density'):
        plt.ylabel("Periodogram: " + str(type) + " (V**2/Hz)")
    if(type == 'spectrum'):
        plt.ylabel("Periodogram: " + str(type) + "(V**2)")
    plt.show()


    fy, periodoy = signal.periodogram(yrTime + 1j*yiTime, samplingFrequency, scaling = type, return_onesided = False)

    plt.plot(fy, periodoy)
    plt.xticks(np.linspace(min(fy), max(fy), numberTicks), np.round(np.linspace(lowerBound, upperBound, numberTicks), 1))
    plt.title("Channel " + str(CHANNEL) + ": Y Polarization")
    plt.xlabel("Frequency (MHz)")
    if(type == 'density'):
        plt.ylabel("Periodogram: " + str(type) + " (V**2/Hz)")
    if(type == 'spectrum'):
        plt.ylabel("Periodogram: " + str(type) + "(V**2)")
    plt.show()



def RAW_FFT(RAW_CHANNEL, CHANNEL, centerFrequency, CHAN_BW, type = 'magnitude'):
    """
    Multiple options to plot FFT of complex, dual-polarized voltage time series.

    The FFT is a computationally efficient Fourier Transform, allowing us to
    see the frequency components of time series data. The default type, magnitude,
    is the absolute value of the FFT. Magnitude normalized divdes the default
    type by the number of samples. Power spectrum and power normalized are the
    square of the default and the square of the default divided by the number of
    samples, respectively.

    Parameters:
    RAW_CHANNEL (2D Array):     The time series data for the channel
    CHANNEL (int):              The channel to be analyzed
    centerFrequency (float):    Center frequency of the channel
    CHAN_BW (float):            Bandwidth of the channel
    type (string):              'magnitude' (default), 'magnitude normalized', 'power spectrum', or 'power normalized'

    Returns:
    Nothing
    """

    # Divide channel into polarizations
    xrTime = RAW_CHANNEL[:,0]
    xiTime = RAW_CHANNEL[:,1]
    yrTime = RAW_CHANNEL[:,2]
    yiTime = RAW_CHANNEL[:,3]

    # Variables for plotting
    transformLength = len(xrTime)
    lowerBound = centerFrequency + CHAN_BW/2
    upperBound = centerFrequency - CHAN_BW/2

    xFFT = np.abs(np.fft.fftshift(np.fft.fft(xrTime + 1j*xiTime)))
    yFFT = np.abs(np.fft.fftshift(np.fft.fft(yrTime + 1j*yiTime)))

    if (type == 'magnitude normalized'):
        xFFT = xFFT/transformLength
        yFFT = yFFT/transformLength

    if (type == 'power spectrum'):
        xFFT = xFFT**2
        yFFT = yFFT**2

    if (type == 'power normalized'):
        xFFT = (xFFT**2)/transformLength
        yFFT = (yFFT**2)/transformLength

    plt.plot(np.linspace(lowerBound, upperBound, transformLength), xFFT)
    plt.title("Channel " + str(CHANNEL) + ": X Polarization")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("FFT: " + str(type))
    plt.show()

    plt.plot(np.linspace(lowerBound, upperBound, transformLength), yFFT)
    plt.title("Channel " + str(CHANNEL) + ": Y Polarization")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("FFT: " + str(type))
    plt.show()


def RAW_pyplot_magnitude(RAW_CHANNEL, CHANNEL, centerFrequency, CHAN_BW, TBIN):
    """
    Plot the FFT magnitude of complex, dual-polarized voltage time series.

    Pyplot defines a function to plot the magnitude of the FFT of time series
    data. This function wraps it conveniently to match the other functions.

    Parameters:
    RAW_CHANNEL (2D Array):     The time series data for the channel
    CHANNEL (int):              The channel to be analyzed
    centerFrequency (float):    Center frequency of the channel (MHz)
    CHAN_BW (float):            Bandwidth of the channel (MHz)
    TBIN (float):               Sampling period (seconds)

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

    spec, freqs, discard = plt.magnitude_spectrum(xrTime + 1j*xiTime, samplingFrequency)
    plt.title("Channel " + str(CHANNEL) + ": X Polarization")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("FFT: Magnitude (Energy; Normalized)")
    plt.xticks(np.linspace(min(freqs), max(freqs), numberTicks), np.round(np.linspace(lowerBound, upperBound, numberTicks), 1))
    plt.show()

    plt.magnitude_spectrum(yrTime + 1j*yiTime, samplingFrequency)
    plt.title("Channel " + str(CHANNEL) + ": Y Polarization")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("FFT: Magnitude (Energy; Normalized)")
    plt.xticks(np.linspace(min(freqs), max(freqs), numberTicks), np.round(np.linspace(lowerBound, upperBound, numberTicks), 1))
    plt.show()
