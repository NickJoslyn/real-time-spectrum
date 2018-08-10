# Nicholas Joslyn
# Breakthrough Listen (BL) UC-Berkeley SETI Intern 2018
# Quasi-real-time BL RAW data visualization for BL observations at Parkes

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sys
import subprocess
from datetime import datetime

from scipy import special
from scipy import optimize
from scipy import signal

from argparse import ArgumentParser
from matplotlib.colors import LogNorm
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages

from slacker import Slacker

################################################################################
######################---Functions---###########################################
################################################################################

################################################################################
### BL RAW ###

def extractHeader(RAW_file, byteLocation):
    """
    Extracts important information from the ASCII text header of a BL RAW data file.

    The BL RAW data format consists of an ASCII text header followed by a binary
    data segment. The header must be parsed effectively to understand the data.
    This function identifies the necessary information from the header.

    Parameters:
    RAW_file (memmap):          The BL RAW data file location (memmap-ed)
    byteLocation (int):   The current location in file (likely 0)

    Returns:
    OBSNCHAN (int):     Number of channels in each data block
    NPOL (int):         Number of polarizations
    NBITS (int):        Number of bits per real/imaginary value
    BLOCSIZE (int):     The size of the data block in bytes
    OBSFREQ (float):    The central frequency observed (MHz)
    OBSBW (float):      The bandwidth of the observation (MHz)
    TBIN (float):       The sampling period (seconds)
    headerOffset (int): The number of bytes (padding included) in the header
    """

    loop = True
    lineCounter = 0
    cardLength = 80

    while(loop):
        cardString = ''

        #Get the ASCII value of the card and convert to char
        for index in range(cardLength):
          cardString += chr(RAW_file[byteLocation + index + lineCounter * cardLength])
        #Identify the end of the header
        #If not the end, find other useful parameters from header
        if (cardString[:3] == 'END'):   #reached end of header
          loop = False

        elif(cardString[:8] == 'OBSNCHAN'): #Number of Channels
          OBSNCHAN = int(cardString[9:].strip()) #remove white spaces and convert to int

        elif(cardString[:4] == 'NPOL'):     #Number of Polarizations * 2
          NPOL = int(cardString[9:].strip())

        elif(cardString[:5] == 'NBITS'):    #Number of Bits per Data Sample
          NBITS = int(cardString[9:].strip())

        elif(cardString[:8] == 'BLOCSIZE'): #Duration of Data Block in Bytes
          BLOCSIZE = int(cardString[9:].strip())

        elif(cardString[:8] == 'DIRECTIO'):
            try:
                DIRECTIO = int(cardString[9:].strip())
            except:
                DIRECTIO = int(cardString[9:].strip()[1])

        elif(cardString[:7] == 'OBSFREQ'):
          OBSFREQ = float(cardString[9:].strip())

        elif(cardString[:5] == 'OBSBW'):
          OBSBW = float(cardString[9:].strip())

        elif(cardString[:4] == 'TBIN'):
          TBIN = float(cardString[9:].strip())

        lineCounter += 1    #Go to Next Card in Header

    #Padding Bytes
    if (DIRECTIO != 0):
        DIRECTIO_offset = 512 - (cardLength*lineCounter)%512
    else:
        DIRECTIO_offset = 0

    # Number of bytes in the header
    headerOffset = cardLength * lineCounter + DIRECTIO_offset

    return OBSNCHAN, NPOL, NBITS, BLOCSIZE, OBSFREQ, OBSBW, TBIN, headerOffset

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

def convert_to_resolution(numBins, numInts, tbin):
    """
    Convert FFT parameters (bins, integrations) to frequency (Hz) and time (s) resolution.

    Return:
    freqRes (int):  Frequency resolution (i.e. bin width)
    timeRes (float):  Time duration rounded to microsecond
    """

    freqRes = int(1/(numBins * tbin))
    timeRes = round(numInts/freqRes, 6)

    return freqRes, timeRes

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

def calculate_spectra(No_DC_BLOCK, OBSNCHAN, fftsPerIntegration, samplesPerTransform):
    """
    Calculate a series of power spectra for BL RAW time series data.

    The power spectra is defined as FFT(time series)**2. By taking the FFT, the data
    is tranformed from the time domain to the frequency domain. The dual-polarized,
    channelized voltages maintain their integrity by being returned in x and y
    multidimensional arrays. The cross-spectrum of the dual-polarized data is calculated
    as well. Cross-spectrum is defined as the conjugate FFT of the X polarization
    multipled by the FFT of the Y polarization.

    Return:
    x_pol_spectra (array):      3-D Array [# Channels, # FFTs, # Samples]
    y_pol_spectra (array):      3-D Array [# Channels, # FFTs, # Samples]
    cross_pol_spectra (array):  3-D Array [# Channels, # FFTs, # Samples]
    """

    x_pol_spectra = np.zeros((OBSNCHAN, fftsPerIntegration, samplesPerTransform))
    y_pol_spectra = np.zeros((OBSNCHAN, fftsPerIntegration, samplesPerTransform))
    cross_pol_spectra = np.zeros((OBSNCHAN, fftsPerIntegration, samplesPerTransform))

    for channel in range(OBSNCHAN):
        x_pol_spectra[channel, :, :] = np.abs(np.fft.fftshift(np.fft.fft(np.split(No_DC_BLOCK[channel,:, 0] + 1j*No_DC_BLOCK[channel,:,1], fftsPerIntegration))))**2
        y_pol_spectra[channel, :, :] = np.abs(np.fft.fftshift(np.fft.fft(np.split(No_DC_BLOCK[channel,:,2] + 1j*No_DC_BLOCK[channel,:, 3], fftsPerIntegration))))**2
        cross_pol_spectra[channel, :, :] = np.fft.fftshift(signal.csd(np.split(No_DC_BLOCK[channel,:, 0] + 1j*No_DC_BLOCK[channel,:,1], fftsPerIntegration), np.split(No_DC_BLOCK[channel,:,2] + 1j*No_DC_BLOCK[channel,:, 3], fftsPerIntegration), nperseg=samplesPerTransform, scaling='spectrum')[1])
    return x_pol_spectra, y_pol_spectra, np.abs(cross_pol_spectra)

def spectra_Find_All(BLOCK, OBSNCHAN, samplesPerTransform, fftsPerIntegration, OBSFREQ, OBSBW):
    """
    Calculate full spectral information for BL RAW time series data.

    The interactive real-time data visualization program for BL observations requires
    spectral information for both the X and Y polarizations as well as the cross-spectrum.

    Since BL observations utilize multiple compute nodes simultaneously operating
    on different frequency regions, the upper and lower frequency ranges for the
    current compute node are also calulated.

    Return:
    spectralData_x (array): 3-D Array [# Channels, # FFTs, # Samples]
    spectralData_y (array): 3-D Array [# Channels, # FFTs, # Samples]
    cross_spectra (array):  3-D Array [# Channels, # FFTs, # Samples]
    lowerBound (float):     Lower frequency limit for given compute node
    upperBound (float):     Upper frequency limit for given compute node
    """

    BLOCK = remove_DCoffset(BLOCK)
    spectralData_x, spectralData_y, cross_spectra = calculate_spectra(BLOCK, OBSNCHAN, fftsPerIntegration, samplesPerTransform)

    lowerBound = OBSFREQ + OBSBW/2
    upperBound = OBSFREQ - OBSBW/2

    return spectralData_x, spectralData_y, cross_spectra, lowerBound, upperBound

def findBand():
    """
    Find the frequency band of the observation

    Return:
    observing band (string):    L, S, C, or X
    """

    global numberOfBanks, node_Frequency_Ranges

    if (numberOfBanks == 3):
        observingBand = 'X'
    elif (numberOfBanks == 4):
        observingBand = 'C'
    else:
        if ((node_Frequency_Ranges[0,7,0]/10**3) < 1.2):
            observingBand = 'L'
        else:
            observingBand = 'S'

    return observingBand

################################################################################
### Spectral Kurtosis ###

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

def upperRoot(x, moment_2, moment_3, p):
    """
    Calculate the spectral kurtosis upper Gaussian threshold.

    Return:
    upper (float): Upper Gaussian threshold value
    """

    upper = np.abs( (1 - special.gammainc( (4 * moment_2**3)/moment_3**2, (-(moment_3-2*moment_2**2)/moment_3 + x)/(moment_3/2/moment_2)))-p)
    return upper

def lowerRoot(x, moment_2, moment_3, p):
    """
    Calculate the spectral kurtosis lower Gaussian threshold.

    Return:
    lower (float): Lower Gaussian threshold value
    """

    lower = np.abs(special.gammainc( (4 * moment_2**3)/moment_3**2, (-(moment_3-2*moment_2**2)/moment_3 + x)/(moment_3/2/moment_2))-p)
    return lower

def spectralKurtosis_thresholds(M, p = 0.0013499, N = 1, d = 1):
    """
    Calculate the spectral kurtosis thresholds for given M, N, d, and PFA values.

    The Gaussian limits (within given probability of false alarm) for spectral kurtosis
    are calculated according to Pearson Type III. Standard PFA, N, and d values are set
    as default, but can be overridden.

    Formulation followed by that given in:
        Nita, Gelu M., et al. "EOVSA Implementation of a Spectral Kurtosis Correlator
        for Transient Detection and Classification." Journal of Astronomical
        Instrumentation 5.04 (2016): 1641009.

    Return:
    lowerThreshold (float): Lower Gaussian threshold value
    upperThreshold (float): Upper Gaussian threshold value
    """

    Nd = N * d

    #Statistical moments
    moment_1 = 1
    moment_2 = ( 2*(M**2) * Nd * (1 + Nd) ) / ( (M - 1) * (6 + 5*M*Nd + (M**2)*(Nd**2)) )
    moment_3 = ( 8*(M**3)*Nd * (1 + Nd) * (-2 + Nd * (-5 + M * (4+Nd))) ) / ( ((M-1)**2) * (2+M*Nd) *(3+M*Nd)*(4+M*Nd)*(5+M*Nd))
    moment_4 = ( 12*(M**4)*Nd*(1+Nd)*(24+Nd*(48+84*Nd+M*(-32+Nd*(-245-93*Nd+M*(125+Nd*(68+M+(3+M)*Nd)))))) ) / ( ((M-1)**3)*(2+M*Nd)*(3+M*Nd)*(4+M*Nd)*(5+M*Nd)*(6+M*Nd)*(7+M*Nd) )

    #Pearson Type III Parameters
    delta = moment_1 - ( (2*(moment_2**2))/moment_3 )
    beta = 4 * ( (moment_2**3)/(moment_3**2) )
    alpha = moment_3 / (2 * moment_2)

    error_4 = np.abs( (100 * 3 * beta * (2+beta) * (alpha**4)) / (moment_4 - 1) )
    x = [1]
    upperThreshold = optimize.newton(upperRoot, x[0], args = (moment_2, moment_3, p))
    lowerThreshold = optimize.newton(lowerRoot, x[0], args = (moment_2, moment_3, p))
    return lowerThreshold, upperThreshold

def find_SK_threshold_hits(SPECTRA_polarized, fftsPerIntegration):
    """
    Find the bins exceeding spectral kurtosis Gaussian thresholds.

    The spectral kurtosis is calculated on an input array of spectra. If the spectra
    value exceeds the Gaussian thresholds (either greater than the upper limit or less
    than the lower limit), then this index is recorded.

    Return:
    indices_to_change (array):  Indices of bins exceeding threshold values
    """

    global sk_lower_threshold, sk_upper_threshold

    SK_temp = calculate_spectralKurtosis(SPECTRA_polarized, fftsPerIntegration)
    SK_temp = np.flip(SK_temp, 0).reshape(-1)

    indices_to_change_high = np.where(SK_temp >= sk_upper_threshold)[0]
    indices_to_change_low = np.where(SK_temp <= sk_lower_threshold)[0]

    indices_to_change = np.concatenate((indices_to_change_high, indices_to_change_low))

    return indices_to_change

################################################################################
### Plotting ###

## Interactive

def press(event):
    """
    Allow for and take action on selected keystrokes
    """
    global Plotted_Bank, Plotted_Node
    global Polarization_Plot
    global node_Frequency_Ranges, node_spectra_storage
    global THRESHOLD_PERCENTAGES
    global FILE_COUNT_INDICATOR, TBIN, numberOfNodes, numberOfBanks
    global OBSNCHAN, fftsPerIntegration, samplesPerTransform, SESSION_IDENTIFIER
    global OBSERVATION_IS_RUNNING, PROGRAM_IS_RUNNING
    global ACTIVE_COMPUTE_NODES, TOTAL_RACKS
    global pCLICKED

    sys.stdout.flush()
    plt.pause(0.1)

    if event.key == 'q':
        OBSERVATION_IS_RUNNING = False
        PROGRAM_IS_RUNNING = False

    if event.key == 'p':
        OBSERVATION_IS_RUNNING = False
        pCLICKED = True

    if event.key == 'x':
        Polarization_Plot += 1
        del axis1_desired.lines[:]
        if (Polarization_Plot%2 == 0):
            plot_full_bandpass(node_spectra_storage[0, :, :, 0, :, :, :], Plotted_Bank, node_Frequency_Ranges[:, :, :])
            for j in range(numberOfNodes):
                if(j!=Plotted_Node):
                    plot_otherNodes(node_spectra_storage[0, Plotted_Bank, j, 0, :, :, :], node_spectra_storage[0, Plotted_Bank, j, 0, :, :, :], OBSNCHAN, samplesPerTransform, fftsPerIntegration, node_Frequency_Ranges[Plotted_Bank, j, 0], node_Frequency_Ranges[Plotted_Bank, j, 1])
                else:
                    plot_otherNodes(node_spectra_storage[0, Plotted_Bank, j, 0, :, :, :], node_spectra_storage[0, Plotted_Bank, j, 0, :, :, :], OBSNCHAN, samplesPerTransform, fftsPerIntegration, node_Frequency_Ranges[Plotted_Bank, j, 0], node_Frequency_Ranges[Plotted_Bank, j, 1], 'red')
            if (Plotted_Bank == 0):
                axis1_desired.set_title("High Band: Spectrum (X)")
            else:
                axis1_desired.set_title("Low Band: Spectrum (X)")

        else:
            plot_full_bandpass(node_spectra_storage[0, :, :, 1, :, :, :], Plotted_Bank, node_Frequency_Ranges[:, :, :], 'Y')
            for j in range(numberOfNodes):
                if(j!=Plotted_Node):
                    plot_otherNodes(node_spectra_storage[0, Plotted_Bank, j, 1, :, :, :], node_spectra_storage[0, Plotted_Bank, j, 1, :, :, :], OBSNCHAN, samplesPerTransform, fftsPerIntegration, node_Frequency_Ranges[Plotted_Bank, j, 0], node_Frequency_Ranges[Plotted_Bank, j, 1])
                else:
                    plot_otherNodes(node_spectra_storage[0, Plotted_Bank, j, 1, :, :, :], node_spectra_storage[0, Plotted_Bank, j, 1, :, :, :], OBSNCHAN, samplesPerTransform, fftsPerIntegration, node_Frequency_Ranges[Plotted_Bank, j, 0], node_Frequency_Ranges[Plotted_Bank, j, 1], 'red')
            if (Plotted_Bank == 0):
                axis1_desired.set_title("High Band: Spectrum (Y)")
            else:
                axis1_desired.set_title("Low Band: Spectrum (Y)")
        plt.suptitle(SESSION_IDENTIFIER + " | " + str(desiredFrequencyResolution/(10**6)) + " MHz, " + str(desiredTimeResolution*(10**3)) + " ms Resolution")

    if event.key == 'up':
        Plotted_Bank -= 1
        if (Plotted_Bank <= 0):
            Plotted_Bank = (numberOfBanks - 1)

        plot_full_bandpass(node_spectra_storage[0, :, :, 0, :, :, :], Plotted_Bank, node_Frequency_Ranges[:, :, :])

        clear_full_spectrum()
        clear_node_plots()
        for j in range(numberOfNodes):
            if (j!=Plotted_Node):
                plot_otherNodes(node_spectra_storage[0, Plotted_Bank, j, 0, :, :, :], node_spectra_storage[0, Plotted_Bank, j, 1, :, :, :], OBSNCHAN, samplesPerTransform, fftsPerIntegration, node_Frequency_Ranges[Plotted_Bank, j, 0], node_Frequency_Ranges[Plotted_Bank, j, 1])
        plot_desired_from_click(node_spectra_storage[:, Plotted_Bank, Plotted_Node, 0, :, :, :], node_spectra_storage[:, Plotted_Bank, Plotted_Node, 1, :, :, :], node_spectra_storage[:, Plotted_Bank, Plotted_Node, 2, :, :, :], OBSNCHAN, TBIN, samplesPerTransform, fftsPerIntegration, node_Frequency_Ranges[Plotted_Bank, Plotted_Node, 0], node_Frequency_Ranges[Plotted_Bank, Plotted_Node, 1], THRESHOLD_PERCENTAGES[Plotted_Bank, Plotted_Node, 0, :], THRESHOLD_PERCENTAGES[Plotted_Bank, Plotted_Node, 1, :], FILE_COUNT_INDICATOR+1)
        plt.suptitle(SESSION_IDENTIFIER + " | " + str(desiredFrequencyResolution/(10**6)) + " MHz, " + str(desiredTimeResolution*(10**3)) + " ms Resolution")


    if event.key == 'down':
        Plotted_Bank += 1
        if (Plotted_Bank >= numberOfBanks):
            Plotted_Bank = 0

        plot_full_bandpass(node_spectra_storage[0, :, :, 0, :, :, :], Plotted_Bank, node_Frequency_Ranges[:, :, :])

        clear_full_spectrum()
        clear_node_plots()
        for j in range(numberOfNodes):
            if (j!=Plotted_Node):
                plot_otherNodes(node_spectra_storage[0, Plotted_Bank, j, 0, :, :, :], node_spectra_storage[0, Plotted_Bank, j, 1, :, :, :], OBSNCHAN, samplesPerTransform, fftsPerIntegration, node_Frequency_Ranges[Plotted_Bank, j, 0], node_Frequency_Ranges[Plotted_Bank, j, 1])
        plot_desired_from_click(node_spectra_storage[:, Plotted_Bank, Plotted_Node, 0, :, :, :], node_spectra_storage[:, Plotted_Bank, Plotted_Node, 1, :, :, :], node_spectra_storage[:, Plotted_Bank, Plotted_Node, 2, :, :, :], OBSNCHAN, TBIN, samplesPerTransform, fftsPerIntegration, node_Frequency_Ranges[Plotted_Bank, Plotted_Node, 0], node_Frequency_Ranges[Plotted_Bank, Plotted_Node, 1], THRESHOLD_PERCENTAGES[Plotted_Bank, Plotted_Node, 0, :], THRESHOLD_PERCENTAGES[Plotted_Bank, Plotted_Node, 1, :], FILE_COUNT_INDICATOR+1)
        plt.suptitle(SESSION_IDENTIFIER + " | " + str(desiredFrequencyResolution/(10**6)) + " MHz, " + str(desiredTimeResolution*(10**3)) + " ms Resolution")



    #spectral flip for seemingly opposite increment on nodes
    if event.key == 'right':
        Plotted_Node -= 1
        if (Plotted_Node <= 0):
            Plotted_Node = (numberOfNodes - 1)

        plot_full_bandpass(node_spectra_storage[0, :, :, 0, :, :, :], Plotted_Bank, node_Frequency_Ranges[:, :, :])

        clear_full_spectrum()
        clear_node_plots()
        for j in range(numberOfNodes):
            if (j!=Plotted_Node):
                plot_otherNodes(node_spectra_storage[0, Plotted_Bank, j, 0, :, :, :], node_spectra_storage[0, Plotted_Bank, j, 1, :, :, :], OBSNCHAN, samplesPerTransform, fftsPerIntegration, node_Frequency_Ranges[Plotted_Bank, j, 0], node_Frequency_Ranges[Plotted_Bank, j, 1])
        plot_desired_from_click(node_spectra_storage[:, Plotted_Bank, Plotted_Node, 0, :, :, :], node_spectra_storage[:, Plotted_Bank, Plotted_Node, 1, :, :, :], node_spectra_storage[:, Plotted_Bank, Plotted_Node, 2, :, :, :], OBSNCHAN, TBIN, samplesPerTransform, fftsPerIntegration, node_Frequency_Ranges[Plotted_Bank, Plotted_Node, 0], node_Frequency_Ranges[Plotted_Bank, Plotted_Node, 1], THRESHOLD_PERCENTAGES[Plotted_Bank, Plotted_Node, 0, :], THRESHOLD_PERCENTAGES[Plotted_Bank, Plotted_Node, 1, :], FILE_COUNT_INDICATOR+1)
        plt.suptitle(SESSION_IDENTIFIER + " | " + str(desiredFrequencyResolution/(10**6)) + " MHz, " + str(desiredTimeResolution*(10**3)) + " ms Resolution")


    if event.key == 'left':
        Plotted_Node += 1
        if (Plotted_Node >= numberOfNodes):
            Plotted_Node = 0

        plot_full_bandpass(node_spectra_storage[0, :, :, 0, :, :, :], Plotted_Bank, node_Frequency_Ranges[:, :, :])

        clear_full_spectrum()
        clear_node_plots()
        for j in range(numberOfNodes):
            if (j!=Plotted_Node):
                plot_otherNodes(node_spectra_storage[0, Plotted_Bank, j, 0, :, :, :], node_spectra_storage[0, Plotted_Bank, j, 1, :, :, :], OBSNCHAN, samplesPerTransform, fftsPerIntegration, node_Frequency_Ranges[Plotted_Bank, j, 0], node_Frequency_Ranges[Plotted_Bank, j, 1])
        plot_desired_from_click(node_spectra_storage[:, Plotted_Bank, Plotted_Node, 0, :, :, :], node_spectra_storage[:, Plotted_Bank, Plotted_Node, 1, :, :, :], node_spectra_storage[:, Plotted_Bank, Plotted_Node, 2, :, :, :], OBSNCHAN, TBIN, samplesPerTransform, fftsPerIntegration, node_Frequency_Ranges[Plotted_Bank, Plotted_Node, 0], node_Frequency_Ranges[Plotted_Bank, Plotted_Node, 1], THRESHOLD_PERCENTAGES[Plotted_Bank, Plotted_Node, 0, :], THRESHOLD_PERCENTAGES[Plotted_Bank, Plotted_Node, 1, :], FILE_COUNT_INDICATOR+1)
        plt.suptitle(SESSION_IDENTIFIER + " | " + str(desiredFrequencyResolution/(10**6)) + " MHz, " + str(desiredTimeResolution*(10**3)) + " ms Resolution")

    plt.pause(0.25)

######## Non - interactive

def clear_full_spectrum():
    """
    Clear the top panel -- full spectrum
    """

    global axis1_desired
    axis1_desired.clear()

def clear_node_plots():
    """
    Clear all panels other than the full spectrum
    """

    global axis2_desired, axis3_desired, axis4_desired, axis5_desired, axis6_desired, axis7_desired, axis8_desired, axis9_desired
    global axis6_desired_twin, axis7_desired_twin
    axis6_desired_twin.clear()
    axis7_desired_twin.clear()
    axis2_desired.clear()
    axis3_desired.clear()
    axis4_desired.clear()
    axis5_desired.clear()
    axis6_desired.clear()
    axis7_desired.clear()
    axis8_desired.clear()
    axis9_desired.clear()

def plot_real_time_visualization_general(current_axis, bandPass_x, defaultColor = 'black'):
    """
    Plot the top panel -- full spectrum of all active nodes (except node of interest)
    """
    global axis1_desired
    axis1_desired.plot(current_axis, 10*np.log10(bandPass_x), color = defaultColor)

def plot_real_time_visualization_desired(integrated_spectrum_x, integrated_spectrum_y, bandPass_x, bandPass_y, bandPass_cross, SK_x, SK_y, SK_cross, current_axis, lowerBound, upperBound, samplesPerTransform, fftsPerIntegration, TBIN, thresholdHitsX, thresholdHitsY, file_index):
    """
    Produce the real-time data visualization plots.

    Notes:
    Produces a 9-plot figure.
        1)      A spectrum of the entire observation with the current
        compute node's bandwidth indicated in red (X Polarization).
        2/3)    X/Y Polarization spectra of the compute node's bandwidth
        4/5)    X/Y Polarization waterfall of the compute node's bandwidth flanking the sides vertically
        6/7)    X/Y Polarization spectral kurtosis of the compute node's bandwidth
        8/9)    Cross-Spectrum and spectral kurtosis of cross-spectrum
    """

    global axis1_desired, axis2_desired, axis3_desired, axis4_desired, axis5_desired, axis6_desired, axis7_desired, axis8_desired, axis9_desired
    global axis6_desired_twin, axis7_desired_twin, axis0_desired
    global SESSION_IDENTIFIER, desiredFrequencyResolution, desiredTimeResolution
    global Plotted_Bank, Plotted_Node, colorbar4, colorbar5, PFA_Nita, sk_lower_threshold, sk_upper_threshold
    global CURRENT_TIME_STAMP

    if (Plotted_Bank == 0):
        axis1_desired.set_title("High Band: Spectrum (X)")
    else:
        axis1_desired.set_title("Low Band: Spectrum (X)")

    axis1_desired.plot(current_axis, 10*np.log10(bandPass_x), color = 'red')
    axis1_desired.set_ylabel("Power (dB)")
    axis1_desired.set_xlabel("Frequency (MHz)")
    axis1_desired.margins(x=0)

    axis2_desired.clear()
    axis2_desired.set_title("blc" + str(ACTIVE_COMPUTE_NODES[Plotted_Bank,Plotted_Node]) + " Spectrum: X")
    axis2_desired.set_xlabel("Frequency (MHz)")
    axis2_desired.set_ylabel("Power (dB)")
    axis2_desired.margins(x=0)
    axis2_desired.plot(current_axis, 10*np.log10(bandPass_x), color = 'C0')

    axis3_desired.clear()
    axis3_desired.set_title("blc" + str(ACTIVE_COMPUTE_NODES[Plotted_Bank,Plotted_Node]) + " Spectrum: Y")
    axis3_desired.set_xlabel("Frequency (MHz)")
    axis3_desired.set_ylabel("Power (dB)")
    axis3_desired.margins(x=0)
    axis3_desired.plot(current_axis, 10*np.log10(bandPass_y), color = 'C0')

    im4 = axis4_desired.imshow(10*np.log10(integrated_spectrum_x), cmap = 'viridis', aspect = 'auto', extent = [lowerBound, upperBound, 1, 0])
    divider4 = make_axes_locatable(axis4_desired)
    cax4 = divider4.append_axes('right', size = '5%', pad = 0.05)
    axis4_desired.set_xlabel("Frequency (MHz)")
    axis4_desired.set_yticks([0,1])
    axis4_desired.set_yticklabels([CURRENT_TIME_STAMP, "~Hr Ago"])
    axis4_desired.set_title("blc" + str(ACTIVE_COMPUTE_NODES[Plotted_Bank,Plotted_Node]) + " Waterfall: X")
    if (colorbar4==0):
        colorbar4 = plt.colorbar(im4, cax=cax4, orientation = 'vertical')
        colorbar4.set_label("Power (dB)")
    else:
        colorbar4.remove()
        colorbar4 = plt.colorbar(im4, cax=cax4, orientation='vertical')
        colorbar4.set_label("Power (dB)")

    im5 = axis5_desired.imshow(10*np.log10(integrated_spectrum_y), cmap = 'viridis', aspect = 'auto', extent = [lowerBound, upperBound, 1, 0])
    divider5 = make_axes_locatable(axis5_desired)
    cax5 = divider5.append_axes('right', size = '5%', pad = 0.05)
    axis5_desired.set_xlabel("Frequency (MHz)")
    axis5_desired.set_title("blc" + str(ACTIVE_COMPUTE_NODES[Plotted_Bank,Plotted_Node]) + " Waterfall: Y")
    axis5_desired.set_yticks([0,1])
    axis5_desired.set_yticklabels([CURRENT_TIME_STAMP, "~Hr Ago"])
    if (colorbar5==0):
        colorbar5 = plt.colorbar(im5, cax=cax5, orientation='vertical')
        colorbar5.set_label("Power (dB)")
    else:
        colorbar5.remove()
        colorbar5 = plt.colorbar(im5, cax=cax5, orientation='vertical')
        colorbar5.set_label("Power (dB)")

    axis6_desired.clear()
    axis6_desired.plot(current_axis, SK_x, color = 'C0')
    axis6_desired.set_title("blc" + str(ACTIVE_COMPUTE_NODES[Plotted_Bank,Plotted_Node]) + " Spectral Kurtosis: X")
    axis6_desired.margins(x=0)
    axis6_desired.set_ylim(-0.5, 5)
    axis6_desired.axhline(y=sk_upper_threshold, color = 'y')
    axis6_desired.axhline(y=sk_lower_threshold, color = 'y')
    axis6_desired.set_xlabel("Frequency (MHz)")
    axis6_desired.lines[1].set_label('Gaussian Thresholds')
    axis6_desired.legend(loc = 1)
    axis6_desired.set_ylabel("SK Value", color='C0')
    axis6_desired.tick_params('y', colors='C0')
    axis6_desired.text(current_axis[0]+1, -0.3, "M = " + str(fftsPerIntegration) + " | N = 1 | D = 1 | PFA = " + str(PFA_Nita), fontsize = "8")

    axis6_desired_twin.clear()
    axis6_desired_twin.plot(current_axis, 100*(thresholdHitsX/file_index), 'm.', markersize=3)
    axis6_desired_twin.set_ylabel('Threshold Hits (%)', color='m')
    axis6_desired_twin.tick_params('y', colors='m')
    axis6_desired_twin.margins(x=0)
    axis6_desired_twin.set_ylim(-20, 100)


    axis7_desired.clear()
    axis7_desired.plot(current_axis, SK_y, color = 'C0')
    axis7_desired.set_title("blc" + str(ACTIVE_COMPUTE_NODES[Plotted_Bank,Plotted_Node]) + " Spectral Kurtosis: Y")
    axis7_desired.margins(x=0)
    axis7_desired.set_ylim(-0.5, 5)
    axis7_desired.axhline(y=sk_upper_threshold, color = 'y')
    axis7_desired.axhline(y=sk_lower_threshold, color = 'y')
    axis7_desired.set_xlabel("Frequency (MHz)")
    axis7_desired.lines[1].set_label('Gaussian Thresholds')
    axis7_desired.legend(loc = 1)
    axis7_desired.set_ylabel("SK Value", color='C0')
    axis7_desired.tick_params('y', colors='C0')
    axis7_desired.text(current_axis[0]+1, -0.3, "M = " + str(fftsPerIntegration) + " | N = 1 | D = 1 | PFA = " + str(PFA_Nita), fontsize = "8")

    axis7_desired_twin.clear()
    axis7_desired_twin.plot(current_axis, 100*(thresholdHitsY/file_index), 'm.', markersize=3)
    axis7_desired_twin.set_ylabel('Threshold Hits (%)', color='m')
    axis7_desired_twin.tick_params('y', colors='m')
    axis7_desired_twin.margins(x=0)
    axis7_desired_twin.set_ylim(-20, 100)

    # Cross Spectrum and SK of cross spectrum
    axis8_desired.clear()
    axis8_desired.plot(current_axis, 10*np.log10(bandPass_cross), color = 'C0')
    axis8_desired.set_title("blc" + str(ACTIVE_COMPUTE_NODES[Plotted_Bank,Plotted_Node]) + " Cross-Spectrum")
    axis8_desired.margins(x=0)
    axis8_desired.set_xlabel("Frequency (MHz)")
    axis8_desired.set_ylabel("Power (dB)")

    axis9_desired.clear()
    axis9_desired.plot(current_axis, SK_cross, color = 'C0')
    axis9_desired.set_title("blc" + str(ACTIVE_COMPUTE_NODES[Plotted_Bank,Plotted_Node]) + " Spectral Kurtosis: Cross-Spectrum")
    axis9_desired.margins(x=0)
    axis9_desired.set_ylim(0, 5)
    axis9_desired.set_xlabel("Frequency (MHz)")

    plt.suptitle(SESSION_IDENTIFIER + " | " + str(desiredFrequencyResolution/(10**6)) + " MHz, " + str(desiredTimeResolution*(10**3)) + " ms Resolution")

    plt.connect('key_press_event', press)
    plt.pause(0.25)

def plot_desired_from_click(spectralData_x, spectralData_y, spectralData_cross, OBSNCHAN, TBIN, samplesPerTransform, fftsPerIntegration, lowerBound, upperBound, thresholdHitsX, thresholdHitsY, file_counter):
    """
    After keystroke, prepare new compute node's spectra for the master plotting function
    """

    global most_possible_files_read

    # Spectral Kurtosis
    SK_x = calculate_spectralKurtosis(spectralData_x[0, :, :, :], fftsPerIntegration)
    SK_y = calculate_spectralKurtosis(spectralData_y[0, :, :, :], fftsPerIntegration)
    SK_cross = calculate_spectralKurtosis(spectralData_cross[0, :, :, :], fftsPerIntegration)
    SK_x = np.flip(SK_x, 0).reshape(-1)
    SK_y = np.flip(SK_y, 0).reshape(-1)
    SK_cross = np.flip(SK_cross, 0).reshape(-1)

    # Spectral flip
    waterfall_spectrum_x = np.flip(np.sum(spectralData_x[:, :, :, :], axis = 2), 1).reshape(most_possible_files_read, -1)
    waterfall_spectrum_y = np.flip(np.sum(spectralData_y[:, :, :, :], axis = 2), 1).reshape(most_possible_files_read, -1)

    # Spectrum for plotting
    bandPass_x = waterfall_spectrum_x[0, :]
    bandPass_y = waterfall_spectrum_y[0, :]
    bandPass_cross = np.flip(np.sum(spectralData_cross[0, :, :, :], axis=1), 0).reshape(-1)

    current_RAW_axis = np.linspace(lowerBound, upperBound, OBSNCHAN *samplesPerTransform)
    plot_real_time_visualization_desired(waterfall_spectrum_x, waterfall_spectrum_y, bandPass_x, bandPass_y, bandPass_cross, SK_x, SK_y, SK_cross, current_RAW_axis, lowerBound, upperBound, samplesPerTransform, fftsPerIntegration, TBIN, thresholdHitsX, thresholdHitsY, file_counter)


def plot_otherNodes(spectralData_x, spectralData_y, OBSNCHAN, samplesPerTransform, fftsPerIntegration, lowerBound, upperBound, plot_color = 'black'):
    """
    Calculate spectra of all active nodes (except node of interest)
    """

    bandPass_x = np.flip(np.sum(spectralData_x, 1),0).reshape(-1)
    #bandPass_y = np.sum(np.flip(np.sum(spectralData_y, 1),0), 0)

    current_RAW_axis = np.linspace(lowerBound, upperBound, OBSNCHAN *samplesPerTransform)

    plot_real_time_visualization_general(current_RAW_axis, bandPass_x, plot_color)

def plot_full_bandpass(dataForBandpass, bank_currently_highlighting, range_information, polarization_title = 'X'):
    global axis0_desired, numberOfBanks, numberOfNodes

    axis0_desired.clear()
    axis0_desired.set_title("Full Frequency Range: (" + str(polarization_title) + ")")
    axis0_desired.set_ylabel("Power (dB)")
    axis0_desired.set_xlabel("Frequency (MHz)")
    axis0_desired.margins(x=0)

    for eachBank in range(numberOfBanks):
        for eachNode in range(numberOfNodes):
            current_RAW_axis = np.linspace(range_information[eachBank, eachNode, 0], range_information[eachBank, eachNode, 1], OBSNCHAN *samplesPerTransform)
            individualNode_data = np.flip(np.sum(dataForBandpass[eachBank, eachNode, :, :, :], 1),0).reshape(-1)
            if (eachBank == bank_currently_highlighting):
                axis0_desired.plot(current_RAW_axis, 10*np.log10(individualNode_data), color = 'red')
            else:
                axis0_desired.plot(current_RAW_axis, 10*np.log10(individualNode_data), color = 'black')



################################################################################
#######################---Program---############################################
################################################################################

if __name__ == "__main__":

    parser = ArgumentParser(description="Produces real-time spectral information display. Creates summary waterfall and RFI pdfs.")
    parser.add_argument('-f', action='store',  default=60, dest='files_per_export', type=int,
                        help="Files Per Export. The number of raw files analyzed before exporting waterfall plots. Default: 60")
    parser.add_argument('-b', action='store',  default=13, dest='nodes_in_bank', type=int,
                        help="Nodes per bank. Program assumes total number of compute nodes is multiple of this value. Default: 13")
    parser.add_argument('-c', action='store',  default=44, dest='channels_per_node', type=int,
                        help="Channels per node. Default: 44 (standard for Parkes)")
    parser.add_argument('-s', action='store',  default=16, dest='samples_per_transform', type=int,
                        help="Time Samples per FFT. Default: 16")
    parser.add_argument('-i', action='store',  default=50, dest='ffts_per_integration', type=int,
                        help="Number FFTs to accumulate. Default: 50")
    parser.add_argument('-t', action='store',  default='No', dest='slack_token', type=str,
                        help="Slack token. Specifying token allows PDFs to be exported to Slack. Default: No")
    parser.add_argument('-u', action='store',  default='No', dest='slack_channel', type=str,
                        help="Slack channel username. Specify active_observations channel. Must specify if using Slack. Default: No")
    parse_args = parser.parse_args()

    # Initialize
    most_possible_files_read = parse_args.files_per_export
    numberOfNodes = parse_args.nodes_in_bank
    OBSNCHAN = parse_args.channels_per_node
    samplesPerTransform = parse_args.samples_per_transform
    fftsPerIntegration = parse_args.ffts_per_integration
    slackToken = parse_args.slack_token
    SLACK_CHANNEL = parse_args.slack_channel

    DIRECTORY_NAMES = ['ObservationRFI', 'ObservationWaterfalls']
    # Make folder for .pdf's if it doesn't exist
    for individualDirectory in DIRECTORY_NAMES:
        if (int(subprocess.check_output("find -maxdepth 1 -type d -name " + individualDirectory + " | wc -l", shell=True)) == 0):
            subprocess.Popen("mkdir " + individualDirectory, shell=True)

    if (slackToken != 'No'):
        slack = Slacker(slackToken)
        SLACK_CHANNEL = "#" + SLACK_CHANNEL

    #Temps for initial plot, will be overwritten
    colorbar4 = 0
    colorbar5 = 0
    desiredTimeResolution = 0
    desiredFrequencyResolution = 0

    # Standard 3sigma false alarm error rate
    PFA_Nita = 0.0013499

    # Adjust if the number of compute racks changes
    TOTAL_RACKS = 2
    numberOfBanks = 2

    # Gaussian limits for Spectral Kurtosis
    sk_lower_threshold, sk_upper_threshold = spectralKurtosis_thresholds(fftsPerIntegration, PFA_Nita)

    PROGRAM_IS_RUNNING = True

    HI_BAND = ['01', '03', '05', '07', '11', '13', '15', '17', '21', '23', '25', '27', '31']
    LOW_BAND = ['02', '04', '06', '10', '12', '14', '16', '20', '22', '24', '26', '00', '32']
    ACTIVE_COMPUTE_NODES = np.array([HI_BAND, LOW_BAND])


    while(PROGRAM_IS_RUNNING):
        print('Checking for new session')
        pCLICKED = False

        Polarization_Plot = 0
        Plotted_Bank = 0
        Plotted_Node = 0
        ########################
        ### Shell commands
        #Find the session

        SESSION_IDENTIFIER = str(datetime.now().strftime('%Y%b%d'))


        ########################
        node_Frequency_Ranges = np.zeros((numberOfBanks, numberOfNodes, 2))
        node_spectra_storage = np.zeros((most_possible_files_read, numberOfBanks, numberOfNodes, 3, OBSNCHAN, fftsPerIntegration, samplesPerTransform))
        THRESHOLD_PERCENTAGES = np.zeros((numberOfBanks, numberOfNodes, 2, OBSNCHAN * samplesPerTransform))

        #Initialize Plot
        #SET UP Big Plot -- Can vary how we want big plot to look by adjusting subplot2grid
        plt.figure("Real-Time Data Visualization")
        plt.suptitle(SESSION_IDENTIFIER + " | " + str(desiredFrequencyResolution/(10**6)) + " MHz, " + str(desiredTimeResolution*(10**3)) + " ms Resolution")
        plt.ion()
        plt.show()

        # Full observation range
        axis0_desired = plt.subplot2grid((23,15), (0,3), colspan=9, rowspan=3)
        axis0_desired.set_title("Full Frequency Range: (X)")
        axis0_desired.set_ylabel("Power (dB)")
        axis0_desired.set_xlabel("Frequency (MHz)")
        axis0_desired.margins(x=0)

        # Full bank range
        axis1_desired = plt.subplot2grid((23,15), (5,3), colspan=9, rowspan=3)
        axis1_desired.set_title("blc" + str(ACTIVE_COMPUTE_NODES[Plotted_Bank,Plotted_Node][0]) + "* Spectrum (X)")
        axis1_desired.set_ylabel("Power (dB)")
        axis1_desired.set_xlabel("Frequency (MHz)")
        axis1_desired.margins(x=0)

        # Spectra of compute node
        axis2_desired = plt.subplot2grid((23,15), (10,3), colspan=4, rowspan=3)
        axis2_desired.set_title("blc" + str(ACTIVE_COMPUTE_NODES[Plotted_Bank,Plotted_Node]) + " Spectrum: X")
        axis2_desired.set_xlabel("Frequency (MHz)")
        axis2_desired.set_ylabel("Power (dB)")
        axis2_desired.margins(x=0)

        axis3_desired = plt.subplot2grid((23,15), (10, 8), colspan=4, rowspan=3)
        axis3_desired.set_title("blc" + str(ACTIVE_COMPUTE_NODES[Plotted_Bank,Plotted_Node]) + " Spectrum: Y")
        axis3_desired.set_xlabel("Frequency (MHz)")
        axis3_desired.set_ylabel("Power (dB)")
        axis3_desired.margins(x=0)

        # Waterfall of compute node
        axis4_desired = plt.subplot2grid((23,15), (0, 0), colspan=2, rowspan=23)
        axis4_desired.set_title("blc" + str(ACTIVE_COMPUTE_NODES[Plotted_Bank,Plotted_Node]) + " Waterfall: X")
        axis4_desired.set_xlabel("Frequency (MHz)")
        axis4_desired.margins(x=0)

        axis5_desired = plt.subplot2grid((23,15), (0, 13), colspan=2, rowspan=23)
        axis5_desired.set_title("blc" + str(ACTIVE_COMPUTE_NODES[Plotted_Bank,Plotted_Node]) + " Waterfall: Y")
        axis5_desired.set_xlabel("Frequency (MHz)")
        axis5_desired.margins(x=0)

        # Spectral Kurtosis of compute node
        axis6_desired = plt.subplot2grid((23,15), (15,3), colspan=4, rowspan=3)
        axis6_desired.set_title("blc" + str(ACTIVE_COMPUTE_NODES[Plotted_Bank,Plotted_Node]) + " Spectral Kurtosis: X")
        axis6_desired.margins(x=0)
        axis6_desired.set_ylim(-0.5, 5)
        axis6_desired.set_xlabel("Frequency (MHz)")
        axis6_desired.set_ylabel("SK Value", color='C0')
        axis6_desired.tick_params('y',colors='C0')
        axis6_desired.text(0, -0.1, "M = " + str(fftsPerIntegration) + " | N = 1 | D = 1 | PFA = " + str(PFA_Nita), fontsize = "8")

        axis6_desired_twin = axis6_desired.twinx()
        axis6_desired_twin.set_ylim(-20, 100)
        axis6_desired_twin.set_ylabel('Threshold Hits (%)', color='m')
        axis6_desired_twin.tick_params('y', colors='m')
        axis6_desired_twin.margins(x=0)

        axis7_desired = plt.subplot2grid((23,15), (15, 8), colspan=4, rowspan=3)
        axis7_desired.set_title("blc" + str(ACTIVE_COMPUTE_NODES[Plotted_Bank,Plotted_Node]) + " Spectral Kurtosis: Y")
        axis7_desired.margins(x=0)
        axis7_desired.set_ylim(-0.5, 5)
        axis7_desired.set_xlabel("Frequency (MHz)")
        axis7_desired.set_ylabel("SK Value", color='C0')
        axis7_desired.tick_params('y', colors='C0')
        axis7_desired.text(0, -0.1, "M = " + str(fftsPerIntegration) + " | N = 1 | D = 1 | PFA = " + str(PFA_Nita), fontsize = "8")

        axis7_desired_twin = axis7_desired.twinx()
        axis7_desired_twin.set_ylim(-20, 100)
        axis7_desired_twin.set_ylabel('Threshold Hits (%)', color='m')
        axis7_desired_twin.tick_params('y', colors='m')
        axis7_desired_twin.margins(x=0)

        # Cross Spectrum and SK of cross spectrum
        axis8_desired = plt.subplot2grid((23,15), (20, 3), colspan=4, rowspan=3)
        axis8_desired.set_title("blc" + str(ACTIVE_COMPUTE_NODES[Plotted_Bank,Plotted_Node]) + " Cross-Spectrum")
        axis8_desired.margins(x=0)
        axis8_desired.set_xlabel("Frequency (MHz)")
        axis8_desired.set_ylabel("Power (dB)")

        axis9_desired = plt.subplot2grid((23,15), (20, 8), colspan=4, rowspan=3)
        axis9_desired.set_title("blc" + str(ACTIVE_COMPUTE_NODES[Plotted_Bank,Plotted_Node]) + " Spectral Kurtosis: Cross-Spectrum")
        axis9_desired.margins(x=0)
        axis9_desired.set_ylim(0, 5)
        axis9_desired.set_xlabel("Frequency (MHz)")

        plt.connect('key_press_event', press)


        CURRENT_TIME_STAMP = datetime.now().strftime('%H:%M')
        FILE_COUNT_INDICATOR = 0


        test_raw_count_string = 'find /mnt_blc' + str(ACTIVE_COMPUTE_NODES[0,0]) + '/datax/PKSMB/GUPPI/ -name "*.raw" | wc -l'
        START_NUMBER_FILES = int(subprocess.check_output(test_raw_count_string, shell = True)[:-1])
        check_for_raw = True
        raw_count_temp = 0
        while(check_for_raw):
            if (int(subprocess.check_output(test_raw_count_string, shell = True)[:-1]) > START_NUMBER_FILES):
                OBSERVATION_IS_RUNNING = True
                print('Starting session ' + str(SESSION_IDENTIFIER))
                check_for_raw = False
                startTime = datetime.now().strftime('%H:%M')
                START_NUMBER_FILES = int(subprocess.check_output(test_raw_count_string, shell = True)[:-1])

                #make session directory for outputs
                if (int(subprocess.check_output("find -maxdepth 2 -type d -name " + SESSION_IDENTIFIER + " | wc -l", shell=True)) == 0):
                    subprocess.Popen("mkdir ObservationWaterfalls/" + SESSION_IDENTIFIER, shell=True)


            else:
                raw_count_temp += 1
                if (raw_count_temp == 720):
                    OBSERVATION_IS_RUNNING = False
                    break
                plt.pause(5)


        while(OBSERVATION_IS_RUNNING):
            endOfObservationCounter = 0

            for bank in range(numberOfBanks):
                for node in range(numberOfNodes):
                    test_Number_Files_String = 'ls /mnt_blc' + str(ACTIVE_COMPUTE_NODES[bank, node]) + '/datax/PKSMB/GUPPI/*.raw | wc -l'
                    waiting_for_written_file = True

                    while(waiting_for_written_file):
                        if (int(subprocess.check_output(test_Number_Files_String, shell=True)[:-1]) > (START_NUMBER_FILES + 1)):
                            waiting_for_written_file = False
                        else:
                            START_NUMBER_FILES = int(subprocess.check_output(test_Number_Files_String, shell=True)[:-1]) - 1
                            endOfObservationCounter += 1
                            if (endOfObservationCounter == 60):
                                export_time_final = datetime.now().strftime('%H:%M')
                                print_time_final = datetime.now().strftime('%Y %B %d | %H:%M')
                            if (endOfObservationCounter == 1800):
                                OBSERVATION_IS_RUNNING = False
                            if (OBSERVATION_IS_RUNNING == False):
                                break
                            plt.pause(2)


                    if (OBSERVATION_IS_RUNNING == False):
                        break

                    test_input_file_string = 'ls -trd /mnt_blc' + str(ACTIVE_COMPUTE_NODES[bank, node]) + '/datax/PKSMB/GUPPI/*.raw | tail -2 | head -1'
                    inputFileName = subprocess.check_output(test_input_file_string, shell = True)[:-1]
                    readIn = np.memmap(inputFileName, dtype = 'int8', mode = 'r')
                    currentBytesPassed = 0

                    OBSNCHAN, NPOL, NBITS, BLOCSIZE, OBSFREQ, OBSBW, TBIN, headerOffset = extractHeader(readIn, currentBytesPassed)
                    if (FILE_COUNT_INDICATOR == 0):
                        desiredFrequencyResolution, desiredTimeResolution = convert_to_resolution(samplesPerTransform, fftsPerIntegration, TBIN)

                    NDIM = int(BLOCSIZE/(OBSNCHAN*NPOL*(NBITS/8)))
                    #samplesPerTransform, fftsPerIntegration = convert_resolution(desiredFrequencyResolution, desiredTimeResolution, TBIN)
                    dataBuffer = readIn[(currentBytesPassed + headerOffset):(currentBytesPassed + headerOffset + BLOCSIZE)].reshape(OBSNCHAN, NDIM, NPOL)
                    NDIMsmall = samplesPerTransform * fftsPerIntegration

                    temp_spec_x, temp_spec_y, temp_spec_cross, node_Frequency_Ranges[bank, node, 0], node_Frequency_Ranges[bank, node, 1] = spectra_Find_All(dataBuffer[:, 0:NDIMsmall, :], OBSNCHAN, samplesPerTransform, fftsPerIntegration, OBSFREQ, OBSBW)

                    node_spectra_storage[:, bank, node, 0, :, :, :] = np.insert(node_spectra_storage[:, bank, node, 0, :, :, :], 0, temp_spec_x, axis = 0)[:-1, :, :, :]
                    node_spectra_storage[:, bank, node, 1, :, :, :] = np.insert(node_spectra_storage[:, bank, node, 1, :, :, :], 0, temp_spec_y, axis = 0)[:-1, :, :, :]
                    node_spectra_storage[:, bank, node, 2, :, :, :] = np.insert(node_spectra_storage[:, bank, node, 2, :, :, :], 0, temp_spec_cross, axis = 0)[:-1, :, :, :]

                    x_temp_indices = find_SK_threshold_hits(node_spectra_storage[0, bank, node, 0, :, :, :], fftsPerIntegration)
                    np.add.at(THRESHOLD_PERCENTAGES[bank, node, 0, :], x_temp_indices, 1)
                    y_temp_indices = find_SK_threshold_hits(node_spectra_storage[0, bank, node, 1, :, :, :], fftsPerIntegration)
                    np.add.at(THRESHOLD_PERCENTAGES[bank, node, 1, :], y_temp_indices, 1)

                    del readIn

                    plt.pause(0.05)

                if (OBSERVATION_IS_RUNNING == False):
                    break
            if (OBSERVATION_IS_RUNNING == False):
                break

            CURRENT_TIME_STAMP = datetime.now().strftime('%H:%M')
            START_NUMBER_FILES = int(subprocess.check_output('ls /mnt_blc' + str(ACTIVE_COMPUTE_NODES[0,0]) + '/datax/PKSMB/GUPPI/*.raw | wc -l', shell=True)[:-1]) - 1

            if (FILE_COUNT_INDICATOR>0):
                clear_full_spectrum()

            FILE_COUNT_INDICATOR += 1

            plot_full_bandpass(node_spectra_storage[0, :, :, 0, :, :, :], Plotted_Bank, node_Frequency_Ranges[:, :, :])

            ## Done with spectra collection; plot
            for i in range(numberOfNodes):
                if (i!=Plotted_Node):
                    plot_otherNodes(node_spectra_storage[0, Plotted_Bank, i, 0, :, :, :], node_spectra_storage[0, Plotted_Bank, i, 1, :, :, :], OBSNCHAN, samplesPerTransform, fftsPerIntegration, node_Frequency_Ranges[Plotted_Bank, i, 0], node_Frequency_Ranges[Plotted_Bank, i, 1])

            plot_desired_from_click(node_spectra_storage[:, Plotted_Bank, Plotted_Node, 0, :, :, :], node_spectra_storage[:, Plotted_Bank, Plotted_Node, 1, :, :, :], node_spectra_storage[:, Plotted_Bank, Plotted_Node, 2, :, :, :], OBSNCHAN, TBIN, samplesPerTransform, fftsPerIntegration, node_Frequency_Ranges[Plotted_Bank, Plotted_Node, 0], node_Frequency_Ranges[Plotted_Bank, Plotted_Node, 1], THRESHOLD_PERCENTAGES[Plotted_Bank, Plotted_Node, 0, :], THRESHOLD_PERCENTAGES[Plotted_Bank, Plotted_Node, 1, :], FILE_COUNT_INDICATOR)

            plt.pause(0.2)
            print(datetime.now().strftime('%H:%M:%S'))

            if (FILE_COUNT_INDICATOR%most_possible_files_read==0):
                endTime = datetime.now().strftime('%H:%M')
                ################Export Waterfalls###################################
                BAND_IDENTIFIER = 'MB'
                exportPath = "ObservationWaterfalls/" + str(SESSION_IDENTIFIER) + "/" + str(SESSION_IDENTIFIER) + "_" + str(BAND_IDENTIFIER) + "-band_" + str(startTime.replace(":", "")) + "-" + str(endTime.replace(":", "")) + "_waterfall.pdf"
                pp = PdfPages(exportPath)
                for export_bank in range(numberOfBanks):
                    for export_node in range(numberOfNodes):

                        #### Set up data
                        export_waterfall_spectrum_x = np.flip(np.sum(node_spectra_storage[:, export_bank, export_node, 0, :, :, :], axis = 2), 1).reshape(most_possible_files_read, -1)
                        export_waterfall_spectrum_y = np.flip(np.sum(node_spectra_storage[:, export_bank, export_node, 1, :, :, :], axis = 2), 1).reshape(most_possible_files_read, -1)
                        export_waterfall_spectrum_cross = np.flip(np.sum(node_spectra_storage[:, export_bank, export_node, 2, :, :, :], axis = 2), 1).reshape(most_possible_files_read, -1)
                        ########

                        ###### Set up plot
                        export_fig = plt.figure(figsize=(12,10))
                        plt.suptitle("blc" + str(ACTIVE_COMPUTE_NODES[export_bank,export_node]) + " | " + str(desiredFrequencyResolution/(10**6)) + " MHz, " + str(desiredTimeResolution*(10**3)) + " ms Resolution")

                        export_axis1 = plt.subplot2grid((14,8), (0, 0), colspan=2, rowspan=14)
                        export_axis1.set_title("X")
                        export_axis1.set_xlabel("Frequency (MHz)")
                        #export_axis1.set_ylabel("Time (Hours)")
                        export_axis1.margins(x=0)

                        export_axis2 = plt.subplot2grid((14,8), (0, 6), colspan=2, rowspan=14)
                        export_axis2.set_title("Y")
                        export_axis2.set_xlabel("Frequency (MHz)")
                        #export_axis2.set_ylabel("Time (Hours)")
                        export_axis2.margins(x=0)

                        export_axis3 = plt.subplot2grid((14,8), (0, 3), colspan=2, rowspan=14)
                        export_axis3.set_title("Cross-Spectrum")
                        export_axis3.set_xlabel("Frequency (MHz)")
                        #export_axis3.set_ylabel("Time (Hours)")
                        export_axis3.margins(x=0)
                        ########

                        ####### Plot data
                        export_im_x = export_axis1.imshow(10*np.log10(export_waterfall_spectrum_x), cmap = 'viridis', aspect = 'auto', extent = [node_Frequency_Ranges[export_bank, export_node, 0], node_Frequency_Ranges[export_bank, export_node, 1], 1, 0])
                        export_divider_x = make_axes_locatable(export_axis1)
                        export_cax_x = export_divider_x.append_axes('right', size = '5%', pad = 0.05)
                        export_colorbar_x = plt.colorbar(export_im_x, cax=export_cax_x, orientation = 'vertical')
                        #export_colorbar_x.set_label("Power (dB)")
                        export_axis1.set_yticks([0,1])
                        export_axis1.set_yticklabels([endTime, startTime])
                        export_axis1.get_xaxis().set_ticks([round(node_Frequency_Ranges[export_bank, export_node, 0],0), round(node_Frequency_Ranges[export_bank, export_node, 1],0)])

                        export_im_y = export_axis2.imshow(10*np.log10(export_waterfall_spectrum_y), cmap = 'viridis', aspect = 'auto', extent = [node_Frequency_Ranges[export_bank, export_node, 0], node_Frequency_Ranges[export_bank, export_node, 1], 1, 0])
                        export_divider_y = make_axes_locatable(export_axis2)
                        export_cax_y = export_divider_y.append_axes('right', size = '5%', pad = 0.05)
                        export_colorbar_y = plt.colorbar(export_im_y, cax=export_cax_y, orientation = 'vertical')
                        export_colorbar_y.set_label("Power (dB)")
                        export_axis2.get_yaxis().set_ticks([])
                        export_axis2.get_xaxis().set_ticks([round(node_Frequency_Ranges[export_bank, export_node, 0],0), round(node_Frequency_Ranges[export_bank, export_node, 1],0)])

                        export_im_cross = export_axis3.imshow(10*np.log10(export_waterfall_spectrum_cross), cmap = 'viridis', aspect = 'auto', extent = [node_Frequency_Ranges[export_bank, export_node, 0], node_Frequency_Ranges[export_bank, export_node, 1], 1, 0])
                        export_divider_cross = make_axes_locatable(export_axis3)
                        export_cax_cross = export_divider_cross.append_axes('right', size = '5%', pad = 0.05)
                        export_colorbar_cross = plt.colorbar(export_im_cross, cax=export_cax_cross, orientation = 'vertical')
                        #export_colorbar_cross.set_label("Power (dB)")
                        export_axis3.get_yaxis().set_ticks([])
                        export_axis3.get_xaxis().set_ticks([round(node_Frequency_Ranges[export_bank, export_node, 0],0), round(node_Frequency_Ranges[export_bank, export_node, 1],0)])
                        ########

                        ######## Write to PDF
                        plt.close()
                        pp.savefig(export_fig)
                        ########

                pp.close()

                if (slackToken != 'No'):
                    slack.files.upload(exportPath, channels = SLACK_CHANNEL)

                startTime = datetime.now().strftime('%H:%M')


        ################END OF WHILE LOOP###################################
        ####################################################################

        if (PROGRAM_IS_RUNNING == False):
            plt.close()
            break

        endTime = datetime.now().strftime('%H:%M')
        if (endOfObservationCounter == 1800):
            endTime = export_time_final

        if (FILE_COUNT_INDICATOR != 0):
            last_non_exported_spectra = FILE_COUNT_INDICATOR%most_possible_files_read
            BAND_IDENTIFIER = 'MB'
            exportPath = "ObservationWaterfalls/" + str(SESSION_IDENTIFIER) + "/" + str(SESSION_IDENTIFIER) + "_" + str(BAND_IDENTIFIER) + "-band_" + str(startTime.replace(":", "")) + "-" + str(endTime.replace(":", "")) + "_waterfall.pdf"
            pp = PdfPages(exportPath)
            for export_bank in range(numberOfBanks):
                for export_node in range(numberOfNodes):

                    #### Set up data
                    export_waterfall_spectrum_x = np.flip(np.sum(node_spectra_storage[:last_non_exported_spectra, export_bank, export_node, 0, :, :, :], axis = 2), 1).reshape(last_non_exported_spectra, -1)
                    export_waterfall_spectrum_y = np.flip(np.sum(node_spectra_storage[:last_non_exported_spectra, export_bank, export_node, 1, :, :, :], axis = 2), 1).reshape(last_non_exported_spectra, -1)
                    export_waterfall_spectrum_cross = np.flip(np.sum(node_spectra_storage[:last_non_exported_spectra, export_bank, export_node, 2, :, :, :], axis = 2), 1).reshape(last_non_exported_spectra, -1)
                    ########

                    ###### Set up plot
                    export_fig = plt.figure(figsize=(12,10))
                    plt.suptitle("blc" + str(ACTIVE_COMPUTE_NODES[export_bank,export_node]) + " | " + str(desiredFrequencyResolution/(10**6)) + " MHz, " + str(desiredTimeResolution*(10**3)) + " ms Resolution")

                    export_axis1 = plt.subplot2grid((14,8), (0, 0), colspan=2, rowspan=14)
                    export_axis1.set_title("X")
                    export_axis1.set_xlabel("Frequency (MHz)")
                    #export_axis1.set_ylabel("Time (Hours)")
                    export_axis1.margins(x=0)

                    export_axis2 = plt.subplot2grid((14,8), (0, 6), colspan=2, rowspan=14)
                    export_axis2.set_title("Y")
                    export_axis2.set_xlabel("Frequency (MHz)")
                    #export_axis2.set_ylabel("Time (Hours)")
                    export_axis2.margins(x=0)

                    export_axis3 = plt.subplot2grid((14,8), (0, 3), colspan=2, rowspan=14)
                    export_axis3.set_title("Cross-Spectrum")
                    export_axis3.set_xlabel("Frequency (MHz)")
                    #export_axis3.set_ylabel("Time (Hours)")
                    export_axis3.margins(x=0)
                    ########

                    ####### Plot data
                    export_im_x = export_axis1.imshow(10*np.log10(export_waterfall_spectrum_x), cmap = 'viridis', aspect = 'auto', extent = [node_Frequency_Ranges[export_bank, export_node, 0], node_Frequency_Ranges[export_bank, export_node, 1], 1, 0])
                    export_divider_x = make_axes_locatable(export_axis1)
                    export_cax_x = export_divider_x.append_axes('right', size = '5%', pad = 0.05)
                    export_colorbar_x = plt.colorbar(export_im_x, cax=export_cax_x, orientation = 'vertical')
                    #export_colorbar_x.set_label("Power (dB)")
                    export_axis1.set_yticks([0,1])
                    export_axis1.set_yticklabels([endTime, startTime])
                    export_axis1.get_xaxis().set_ticks([round(node_Frequency_Ranges[export_bank, export_node, 0],0), round(node_Frequency_Ranges[export_bank, export_node, 1],0)])

                    export_im_y = export_axis2.imshow(10*np.log10(export_waterfall_spectrum_y), cmap = 'viridis', aspect = 'auto', extent = [node_Frequency_Ranges[export_bank, export_node, 0], node_Frequency_Ranges[export_bank, export_node, 1], 1, 0])
                    export_divider_y = make_axes_locatable(export_axis2)
                    export_cax_y = export_divider_y.append_axes('right', size = '5%', pad = 0.05)
                    export_colorbar_y = plt.colorbar(export_im_y, cax=export_cax_y, orientation = 'vertical')
                    export_colorbar_y.set_label("Power (dB)")
                    export_axis2.get_yaxis().set_ticks([])
                    export_axis2.get_xaxis().set_ticks([round(node_Frequency_Ranges[export_bank, export_node, 0],0), round(node_Frequency_Ranges[export_bank, export_node, 1],0)])

                    export_im_cross = export_axis3.imshow(10*np.log10(export_waterfall_spectrum_cross), cmap = 'viridis', aspect = 'auto', extent = [node_Frequency_Ranges[export_bank, export_node, 0], node_Frequency_Ranges[export_bank, export_node, 1], 1, 0])
                    export_divider_cross = make_axes_locatable(export_axis3)
                    export_cax_cross = export_divider_cross.append_axes('right', size = '5%', pad = 0.05)
                    export_colorbar_cross = plt.colorbar(export_im_cross, cax=export_cax_cross, orientation = 'vertical')
                    #export_colorbar_cross.set_label("Power (dB)")
                    export_axis3.get_yaxis().set_ticks([])
                    export_axis3.get_xaxis().set_ticks([round(node_Frequency_Ranges[export_bank, export_node, 0],0), round(node_Frequency_Ranges[export_bank, export_node, 1],0)])
                    ########

                    ######## Write to PDF
                    plt.close()
                    pp.savefig(export_fig)
                    ########

            pp.close()

            if (slackToken != 'No'):
                slack.files.upload(exportPath, channels = SLACK_CHANNEL)

            exportPath = "ObservationRFI/" + str(SESSION_IDENTIFIER) + "_" + str(BAND_IDENTIFIER) + "-band_" + str(endTime.replace(":", "")) + "_RFI.pdf"
            pp = PdfPages(exportPath)
            for export_bank in range(numberOfBanks):
                for export_node in range(numberOfNodes):

                    #### Set up data
                    export_RFI_x = THRESHOLD_PERCENTAGES[export_bank, export_node, 0, :]
                    export_RFI_y = THRESHOLD_PERCENTAGES[export_bank, export_node, 1, :]
                    export_current_axis = np.linspace(node_Frequency_Ranges[export_bank, export_node, 0], node_Frequency_Ranges[export_bank, export_node, 1], OBSNCHAN *samplesPerTransform)

                    ########

                    ###### Set up plot
                    export_fig = plt.figure(figsize=(12,10))

                    export_axis1 = plt.subplot2grid((2,1), (0, 0))
                    export_axis1.set_title("blc" + str(ACTIVE_COMPUTE_NODES[export_bank,export_node]) + " X | Percent RFI")
                    export_axis1.set_xlabel("Frequency (MHz)")
                    export_axis1.set_ylabel("%")
                    export_axis1.set_ylim(0,100)
                    export_axis1.margins(x=0)
                    export_axis1.plot(export_current_axis, 100*(export_RFI_x/(FILE_COUNT_INDICATOR + 1)))

                    export_axis2 = plt.subplot2grid((2,1), (1, 0))
                    export_axis2.set_title("blc" + str(ACTIVE_COMPUTE_NODES[export_bank,export_node]) + " Y | Percent RFI")
                    export_axis2.set_xlabel("Frequency (MHz)")
                    export_axis2.set_ylabel("%")
                    export_axis2.margins(x=0)
                    export_axis2.set_ylim(0,100)
                    export_axis2.plot(export_current_axis, 100*(export_RFI_y/(FILE_COUNT_INDICATOR + 1)))
                    ########

                    ######## Write to PDF
                    plt.tight_layout()
                    plt.close()
                    pp.savefig(export_fig)

            pp.close()

            if (slackToken != 'No'):
                slack.files.upload(exportPath, channels = SLACK_CHANNEL)

            #### Monthly RFI stuff
            monthly_RFI_file_x = "monthlyRFI_" + str(BAND_IDENTIFIER) + "_X.npy"
            monthly_RFI_file_y = "monthlyRFI_" + str(BAND_IDENTIFIER) + "_Y.npy"
            monthly_RFI_counter_file = "ObservationRFI/monthlyRFI_" + str(BAND_IDENTIFIER) + "_Counter.npy"
            if (int(subprocess.check_output("find -maxdepth 2 -name " + monthly_RFI_file_x + " | wc -l", shell=True)) == 0):
                np.save("ObservationRFI/" + monthly_RFI_file_x, 100*(THRESHOLD_PERCENTAGES[:,:,0,:]/(FILE_COUNT_INDICATOR + 1)))
                np.save("ObservationRFI/" + monthly_RFI_file_y, 100*(THRESHOLD_PERCENTAGES[:,:,1,:]/(FILE_COUNT_INDICATOR + 1)))
                np.save(monthly_RFI_counter_file, 1)
                np.save("ObservationRFI/" + str(BAND_IDENTIFIER) + "_FrequencyRange.npy", [node_Frequency_Ranges[numberOfBanks-1, numberOfNodes-1, 0], node_Frequency_Ranges[0,0,1]])

            else:
                temp_RFI_load_x = np.load("ObservationRFI/" + monthly_RFI_file_x)
                temp_RFI_load_y = np.load("ObservationRFI/" + monthly_RFI_file_y)

                np.save("ObservationRFI/" + monthly_RFI_file_x, 100*(THRESHOLD_PERCENTAGES[:,:,0,:]/(FILE_COUNT_INDICATOR + 1)) + temp_RFI_load_x)
                np.save("ObservationRFI/" + monthly_RFI_file_y, 100*(THRESHOLD_PERCENTAGES[:,:,1,:]/(FILE_COUNT_INDICATOR + 1)) + temp_RFI_load_y)

                temp_RFI_counter = np.load(monthly_RFI_counter_file)
                np.save(monthly_RFI_counter_file, temp_RFI_counter+1)

        waiting_for_new_observation = True

        print_time = datetime.now().strftime('%Y %B %d | %H:%M')
        if (endOfObservationCounter == 1800):
            print_time = print_time_final
        print("End of session " + str(SESSION_IDENTIFIER) + ": " + str(print_time))

        if (pCLICKED == True):
            if (raw_input("New Observation? (y/n): ") == 'n'):
                plt.close()
                PROGRAM_IS_RUNNING = False
            else:
                plt.close()
            pCLICKED = False
            waiting_for_new_observation = False

        START_NUMBER_FILES = int(subprocess.check_output('find /mnt_blc' + str(ACTIVE_COMPUTE_NODES[0,0]) + '/datax/PKSMB/GUPPI/ -name "*.raw" | wc -l', shell=True)[:-1])
        ### automatic restart check in comparison to CURRENT_NUMBER_OF_OBS
        while (waiting_for_new_observation):
            if (PROGRAM_IS_RUNNING == False):
                break

            if (int(subprocess.check_output('find /mnt_blc' + str(ACTIVE_COMPUTE_NODES[0,0]) + '/datax/PKSMB/GUPPI/ -name "*.raw" | wc -l', shell=True)[:-1]) > START_NUMBER_FILES):
                break
            else:
                START_NUMBER_FILES = int(subprocess.check_output('find /mnt_blc' + str(ACTIVE_COMPUTE_NODES[0,0]) + '/datax/PKSMB/GUPPI/ -name "*.raw" | wc -l', shell=True)[:-1])
            plt.pause(10)
