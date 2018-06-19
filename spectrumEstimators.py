# Goal:
## Define functions to be called as spectrum estimators on BL RAW data

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

def RAW_periodogram(RAW_CHANNEL, CHANNEL, centerFrequency, CHAN_BW, TBIN, type = 'density'):

    # Put in help definer as shown on Kaggle

    xrTime = RAW_CHANNEL[:,0]
    xiTime = RAW_CHANNEL[:,1]
    yrTime = RAW_CHANNEL[:,2]
    yiTime = RAW_CHANNEL[:,3]

    samplingFrequency = 1/TBIN
    lowerBound = centerFrequency + CHAN_BW/2
    upperBound = centerFrequency - CHAN_BW/2

    #Units V**2/Hz -- density
    #Units V**2 -- spectrum

    fx, periodox = signal.periodogram(xrTime + 1j*xiTime, samplingFrequency, scaling = type, return_onesided = False)
    plt.plot(fx, periodox)
    plt.xticks(np.linspace(min(fx), max(fx), 7), np.round(np.linspace(lowerBound, upperBound, 7), 1))
    plt.title("Periodogram " + str(type) + ": Channel " + str(CHANNEL) + ': X Polarization')
    plt.show()

    fy, periodoy = signal.periodogram(yrTime + 1j*yiTime, samplingFrequency, scaling = type, return_onesided = False)
    plt.plot(fy, periodoy)
    plt.xticks(np.linspace(min(fy), max(fy), 7), np.round(np.linspace(lowerBound, upperBound, 7), 1))
    plt.title("Periodogram " + str(type) + ": Channel " + str(CHANNEL) + ": Y Polarization")
    plt.show()



def RAW_FFT(RAW_CHANNEL, CHANNEL, centerFrequency, CHAN_BW, type = 'magnitude'):

    xrTime = RAW_CHANNEL[:,0]
    xiTime = RAW_CHANNEL[:,1]
    yrTime = RAW_CHANNEL[:,2]
    yiTime = RAW_CHANNEL[:,3]

    transformLength = len(xrTime)

    xFFT = np.abs(np.fft.fftshift(np.fft.fft(xrTime + 1j*xiTime)))
    yFFT = np.abs(np.fft.fftshift(np.fft.fft(yrTime + 1j*yiTime)))

    if (type == 'power spectrum'):
        xFFT = xFFT**2
        yFFT = yFFT**2

    if (type == 'power density'):
        xFFT = (xFFT**2)/transformLength
        yFFT = (yFFT**2)/transformLength

    plt.plot(np.linspace(centerFrequency + CHAN_BW/2, centerFrequency - CHAN_BW/2, transformLength), xFFT)
    plt.title("FFT " + str(type) + ": Channel " + str(CHANNEL) + ": X Polarization")
    plt.show()

    plt.plot(np.linspace(centerFrequency + CHAN_BW/2, centerFrequency - CHAN_BW/2, transformLength), yFFT/transformLength)
    plt.title("FFT " + str(type) + ": Channel " + str(CHANNEL) + ": Y Polarization")
    plt.show()
