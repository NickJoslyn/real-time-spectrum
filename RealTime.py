# Nicholas Joslyn
# Breakthrough Listen UC Berkeley SETI Intern 2018

# Functions for real-time spectra of BL Observations
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.collections import LineCollection

def convert_resolution(customFrequencyResolution, customTimeResolution, TBIN):
    """
    Convert custom frequency (Hz) resolution and time (s) resolution into FFT parameters.

    Return:
    samplesPerTransform (int):  Number of samples for each FFT
    fftsPerIntegration (int):   Number of FFTs to integrate over
    """

    samplesPerTransform = int((1/customFrequencyResolution)/TBIN)
    fftsPerIntegration = int(customTimeResolution * customFrequencyResolution)
    
    return samplesPerTransform, fftsPerIntegration

def remove_DCoffset(BLOCK):
    """
    Remove the DC offset from a block of BL RAW time series data.

    In signal processing, the DC offset is the result of a linear trend in the data.
    It is a large spike in the frequency domain of the data that can bias
    results and analysis. This function removes the DC offset from a set of data.

    Return:
    BLOCK - DC_offset (array):   Array of identical dimensionality without DC offset
    """

    channels, samples, polarizations = BLOCK.shape
    DC_offset = np.mean(BLOCK, axis = 1).reshape(channels, 1, polarizations)
    _, DC_offset = np.broadcast_arrays(BLOCK, DC_offset)

    return (BLOCK - DC_offset)

def calculate_spectralKurtosis(SPECTRA, fftsPerIntegration):
    """
    Calculate the spectral kurtosis from a series of Power Spectral Density Estimates.

    The spectral kurtosis of data allows quick identification of Gaussianity (i.e
    distinction between natural and artificial signals). Gaussian signals will be
    approximately unity.

    Return:
    SK_estimate (array):   Array of SK estimates for the channels present in the input array
    """

    S_1 = np.sum(SPECTRA, axis = 1)
    S_2 = np.sum(SPECTRA**2, axis = 1)
    SK_estimate = ((fftsPerIntegration + 1)/(fftsPerIntegration - 1)) * ((fftsPerIntegration * S_2)/(S_1**2) - 1)

    return SK_estimate

def calculate_spectra(No_DC_BLOCK, OBSNCHAN, fftsPerIntegration, samplesPerTransform):
    """
    Calculate a series of power spectra for BL RAW time series data.

    The power spectra is defined as FFT(time series)**2. By taking the FFT, the data
    is tranformed from the time domain to the frequency domain. The dual-polarized,
    channelized voltages maintain their integrity by being returned in x and y
    multidimensional arrays.

    Return:
    x_pol_spectra (array):  3-D Array [# Channels, # FFTs, # Samples]
    y_pol_spectra (array):  3-D Array [# Channels, # FFTs, # Samples]
    """

    x_pol_spectra = np.zeros((OBSNCHAN, fftsPerIntegration, samplesPerTransform))
    y_pol_spectra = np.zeros((OBSNCHAN, fftsPerIntegration, samplesPerTransform))

    for channel in range(OBSNCHAN):
        x_pol_spectra[channel, :, :] = np.abs(np.fft.fftshift(np.fft.fft(np.split(No_DC_BLOCK[channel,:, 0] + 1j*No_DC_BLOCK[channel,:,1], fftsPerIntegration))))**2
        y_pol_spectra[channel, :, :] = np.abs(np.fft.fftshift(np.fft.fft(np.split(No_DC_BLOCK[channel,:,2] + 1j*No_DC_BLOCK[channel,:, 3], fftsPerIntegration))))**2

    return x_pol_spectra, y_pol_spectra

def plot_real_time_visualization(integrated_spectrum_x, integrated_spectrum_y, bandPass_x, bandPass_y, SK_x, SK_y, current_axis, lowerBound, upperBound, samplesPerTransform, fftsPerIntegration, TBIN):
    """
    Produce the real-time data visualization plots.

    Notes:
    Produces a 7-plot figure.
        1)      A spectrum of the entire observation with the current
        compute node's bandwidth indicated in red (X Polarization).
        2/3)    X/Y Polarization spectra of the compute node's bandwidth
        4/5)    X/Y Polarization waterfall of the compute node's bandwidth
        6/7)    X/Y Polarization spectral kurtosis of the compute node's bandwidth
    """

    totalTime = samplesPerTransform * fftsPerIntegration * TBIN

    #SET UP Big Plot
    plt.figure("Template: " + str(samplesPerTransform) + " Bins Per Channel and " + str(fftsPerIntegration) + " Integrations")

    # Full observational range
    ax1 = plt.subplot2grid((18,5), (0,0), colspan=5, rowspan=3)
    ax1.set_title("Full Observation Spectrum (X)")
    ax1.plot(current_axis, bandPass_x, color = "black")
    ax1.plot(bandPass_x[0:len(bandPass_x)//2], color = 'red')
    ax1.set_yscale('log')
    ax1.set_xlabel("Frequency (MHz)")
    ax1.set_ylabel("Power")

    # Spectra of compute node
    ax2 = plt.subplot2grid((18,5), (5,0), colspan=2, rowspan=3)
    ax2.set_title("Node Spectrum: X")
    ax2.set_xlabel("Frequency (MHz)")
    ax2.set_ylabel("Power")
    ax2.set_yscale('log')
    ax2.margins(x=0)
    ax2.plot(current_axis, bandPass_x)
    ax3 = plt.subplot2grid((18,5), (5, 3), colspan=2, rowspan=3)
    ax3.set_title("Node Spectrum: Y")
    ax3.set_xlabel("Frequency (MHz)")
    ax3.set_ylabel("Power")
    ax3.set_yscale('log')
    ax3.margins(x=0)
    ax3.plot(current_axis, bandPass_y)

    # Waterfall of compute node
    ax4 = plt.subplot2grid((18,5), (10, 0), colspan=2, rowspan=3)
    ax4.set_title("Node Waterfall: X")
    ax4.imshow(integrated_spectrum_x, cmap = 'viridis', aspect = 'auto', norm = LogNorm(), extent = [lowerBound, upperBound, totalTime, 0])
    ax4.set_xlabel("Frequency (MHz)")
    ax4.set_ylabel("Time (s)")
    ax4.margins(x=0)
    #plt.colorbar(im, ax=ax4)

    ax5 = plt.subplot2grid((18,5), (10, 3), colspan=2, rowspan=3)
    ax5.set_title("Node Waterfall: Y")
    ax5.imshow(integrated_spectrum_y, cmap = 'viridis', aspect = 'auto', norm = LogNorm(), extent = [lowerBound, upperBound, totalTime, 0])
    ax5.set_xlabel("Frequency (MHz)")
    ax5.set_ylabel("Time (s)")
    ax5.margins(x=0)

    # Spectral Kurtosis of compute node
    ax6 = plt.subplot2grid((18,5), (15,0), colspan=2, rowspan=3)
    ax6.plot(current_axis, SK_x)
    ax6.set_title("Spectral Kurtosis: X")
    ax6.margins(x=0)
    ax6.set_xlabel("Frequency (MHz)")
    ax7 = plt.subplot2grid((18,5), (15, 3), colspan=2, rowspan=3)
    ax7.plot(current_axis, SK_y)
    ax7.set_title("Spectral Kurtosis: Y")
    ax7.margins(x=0)
    ax7.set_xlabel("Frequency (MHz)")

    plt.suptitle("Real-Time Spectra of Observation")
    plt.show()

def real_time_spectra(BLOCK, OBSNCHAN, CHAN_BW, TBIN, samplesPerTransform, fftsPerIntegration, OBSFREQ, OBSBW):
    """
    Plot spectra and stats of real-time observational BL data.

    The goal is to produce a user-friendly real-time data visualization interface
    for BL observations. The RAW datastream is very fast, so the algorithms and
    resolution are computationally inexpensive.
    """
    
    # Frequency Domain
    BLOCK = remove_DCoffset(BLOCK)
    spectralData_x, spectralData_y = calculate_spectra(BLOCK, OBSNCHAN, fftsPerIntegration, samplesPerTransform)

    # Spectral Kurtosis
    SK_x = calculate_spectralKurtosis(spectralData_x, fftsPerIntegration)
    SK_y = calculate_spectralKurtosis(spectralData_y, fftsPerIntegration)
    SK_x = np.flip(SK_x, 0).reshape(-1)
    SK_y = np.flip(SK_y, 0).reshape(-1)

    # Spectral flip
    spectralData_x = np.flip(np.sum(spectralData_x, axis = 1), 0)
    spectralData_y = np.flip(np.sum(spectralData_y, axis = 1), 0)

    # Spectrum for waterfall (array in array for plt.imshow())
    waterfall_spectrum_x = [spectralData_x.reshape(-1)]
    waterfall_spectrum_y = [spectralData_y.reshape(-1)]

    # Spectrum for plotting
    bandPass_x = np.sum(waterfall_spectrum_x, 0)
    bandPass_y = np.sum(waterfall_spectrum_y, 0)

    # Helpful plotting values
    lowerBound = OBSFREQ + OBSBW/2
    upperBound = OBSFREQ - OBSBW/2
    current_RAW_axis = np.linspace(lowerBound, upperBound, OBSNCHAN *samplesPerTransform)

    plot_real_time_visualization(waterfall_spectrum_x, waterfall_spectrum_y, bandPass_x, bandPass_y, SK_x, SK_y, current_RAW_axis, lowerBound, upperBound, samplesPerTransform, fftsPerIntegration, TBIN)
