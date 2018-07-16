# Nicholas Joslyn
# Breakthrough Listen UC Berkeley SETI Intern 2018

# Quasi-real time BL RAW data visualization for BL observations

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import subprocess
from datetime import datetime

from scipy import special
from scipy import optimize
from scipy import signal

from matplotlib.colors import LogNorm
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages
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
    CHAN_BW (float):    The bandwidth of a channel (MHz)
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

        elif(cardString[:7] == 'CHAN_BW'):
          CHAN_BW = float(cardString[9:].strip())

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

    return OBSNCHAN, NPOL, NBITS, BLOCSIZE, OBSFREQ, CHAN_BW, OBSBW, TBIN, headerOffset

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
    cross_pol_spectra = np.zeros((OBSNCHAN, fftsPerIntegration, samplesPerTransform))

    for channel in range(OBSNCHAN):
        x_pol_spectra[channel, :, :] = np.abs(np.fft.fftshift(np.fft.fft(np.split(No_DC_BLOCK[channel,:, 0] + 1j*No_DC_BLOCK[channel,:,1], fftsPerIntegration))))**2
        y_pol_spectra[channel, :, :] = np.abs(np.fft.fftshift(np.fft.fft(np.split(No_DC_BLOCK[channel,:,2] + 1j*No_DC_BLOCK[channel,:, 3], fftsPerIntegration))))**2
        cross_pol_spectra[channel, :, :] = np.fft.fftshift(signal.csd(np.split(No_DC_BLOCK[channel,:, 0] + 1j*No_DC_BLOCK[channel,:,1], fftsPerIntegration), np.split(No_DC_BLOCK[channel,:,2] + 1j*No_DC_BLOCK[channel,:, 3], fftsPerIntegration), nperseg=samplesPerTransform, scaling='spectrum')[1])
    return x_pol_spectra, y_pol_spectra, np.abs(cross_pol_spectra)

def spectra_Find_All(BLOCK, OBSNCHAN, samplesPerTransform, fftsPerIntegration, OBSFREQ, OBSBW):

    BLOCK = remove_DCoffset(BLOCK)
    spectralData_x, spectralData_y, cross_spectra = calculate_spectra(BLOCK, OBSNCHAN, fftsPerIntegration, samplesPerTransform)
    # nodeBand_x = np.flip(np.sum(spectralData_x, 1),0).reshape(-1)
    # nodeBand_y = np.sum(np.flip(np.sum(spectralData_y, 1),0), 0)
    lowerBound = OBSFREQ + OBSBW/2
    upperBound = OBSFREQ - OBSBW/2

    return spectralData_x, spectralData_y, cross_spectra, lowerBound, upperBound

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
    upper = np.abs( (1 - special.gammainc( (4 * moment_2**3)/moment_3**2, (-(moment_3-2*moment_2**2)/moment_3 + x)/(moment_3/2/moment_2)))-p)
    return upper

def lowerRoot(x, moment_2, moment_3, p):
    lower = np.abs(special.gammainc( (4 * moment_2**3)/moment_3**2, (-(moment_3-2*moment_2**2)/moment_3 + x)/(moment_3/2/moment_2))-p)
    return lower

def spectralKurtosis_thresholds(M, p = 0.0013499, N = 1, d = 1):

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

    global PFA_Nita

    sk_lower_threshold, sk_upper_threshold = spectralKurtosis_thresholds(fftsPerIntegration, PFA_Nita)


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
    global Plotted_Bank, Plotted_Node
    global Polarization_Plot
    global node_Frequency_Ranges, node_spectra_storage, THRESHOLD_PERCENTAGES
    global FILE_COUNT_INDICATOR, TBIN, BANK_OFFSET, numberOfNodes, numberOfBanks
    global OBSNCHAN, fftsPerIntegration, samplesPerTransform, SESSION_IDENTIFIER, OBSERVATION_IS_RUNNING
    sys.stdout.flush()

    plt.pause(0.00001)

    if event.key == 'q':
        OBSERVATION_IS_RUNNING = False

    if event.key == 'x':
        Polarization_Plot += 1
        del axis1_desired.lines[:]
        if (Polarization_Plot%2 == 0):
            for j in range(numberOfNodes):
                if(j!=Plotted_Node):
                    plot_otherNodes(node_spectra_storage[FILE_COUNT_INDICATOR - 1, Plotted_Bank, j, 0, :, :, :], node_spectra_storage[FILE_COUNT_INDICATOR - 1, Plotted_Bank, j, 0, :, :, :], OBSNCHAN, samplesPerTransform, fftsPerIntegration, node_Frequency_Ranges[Plotted_Bank, j, 0], node_Frequency_Ranges[Plotted_Bank, j, 1])
                else:
                    plot_otherNodes(node_spectra_storage[FILE_COUNT_INDICATOR - 1, Plotted_Bank, j, 0, :, :, :], node_spectra_storage[FILE_COUNT_INDICATOR - 1, Plotted_Bank, j, 0, :, :, :], OBSNCHAN, samplesPerTransform, fftsPerIntegration, node_Frequency_Ranges[Plotted_Bank, j, 0], node_Frequency_Ranges[Plotted_Bank, j, 1], 'red')
            axis1_desired.set_title("blc" + str(Plotted_Bank + BANK_OFFSET) + "{0..7} Spectrum (X)")

        else:
            for j in range(numberOfNodes):
                if(j!=Plotted_Node):
                    plot_otherNodes(node_spectra_storage[FILE_COUNT_INDICATOR - 1, Plotted_Bank, j, 1, :, :, :], node_spectra_storage[FILE_COUNT_INDICATOR - 1, Plotted_Bank, j, 1, :, :, :], OBSNCHAN, samplesPerTransform, fftsPerIntegration, node_Frequency_Ranges[Plotted_Bank, j, 0], node_Frequency_Ranges[Plotted_Bank, j, 1])
                else:
                    plot_otherNodes(node_spectra_storage[FILE_COUNT_INDICATOR - 1, Plotted_Bank, j, 1, :, :, :], node_spectra_storage[FILE_COUNT_INDICATOR - 1, Plotted_Bank, j, 1, :, :, :], OBSNCHAN, samplesPerTransform, fftsPerIntegration, node_Frequency_Ranges[Plotted_Bank, j, 0], node_Frequency_Ranges[Plotted_Bank, j, 1], 'red')
            axis1_desired.set_title("blc" + str(Plotted_Bank + BANK_OFFSET) + "{0..7} Spectrum (Y)")


    if event.key == 'up':
        Plotted_Bank += 1
        if (Plotted_Bank > (numberOfBanks-1)):
            Plotted_Bank = 0
        clear_full_spectrum()
        clear_node_plots()
        for j in range(numberOfNodes):
            if (j!=Plotted_Node):
                plot_otherNodes(node_spectra_storage[FILE_COUNT_INDICATOR - 1, Plotted_Bank, j, 0, :, :, :], node_spectra_storage[FILE_COUNT_INDICATOR - 1, Plotted_Bank, j, 1, :, :, :], OBSNCHAN, samplesPerTransform, fftsPerIntegration, node_Frequency_Ranges[Plotted_Bank, j, 0], node_Frequency_Ranges[Plotted_Bank, j, 1])
        plot_desired_from_click(node_spectra_storage[:, Plotted_Bank, Plotted_Node, 0, :, :, :], node_spectra_storage[:, Plotted_Bank, Plotted_Node, 1, :, :, :], node_spectra_storage[:, Plotted_Bank, Plotted_Node, 2, :, :, :], OBSNCHAN, TBIN, samplesPerTransform, fftsPerIntegration, node_Frequency_Ranges[Plotted_Bank, Plotted_Node, 0], node_Frequency_Ranges[Plotted_Bank, Plotted_Node, 1], FILE_COUNT_INDICATOR - 1, THRESHOLD_PERCENTAGES[Plotted_Bank, Plotted_Node, 0, :], THRESHOLD_PERCENTAGES[Plotted_Bank, Plotted_Node, 1, :])

    if event.key == 'down':
        Plotted_Bank -= 1
        if (Plotted_Bank < 0):
            Plotted_Bank = (numberOfBanks -1)
        clear_full_spectrum()
        clear_node_plots()
        for j in range(numberOfNodes):
            if (j!=Plotted_Node):
                plot_otherNodes(node_spectra_storage[FILE_COUNT_INDICATOR - 1, Plotted_Bank, j, 0, :, :, :], node_spectra_storage[FILE_COUNT_INDICATOR - 1, Plotted_Bank, j, 1, :, :, :], OBSNCHAN, samplesPerTransform, fftsPerIntegration, node_Frequency_Ranges[Plotted_Bank, j, 0], node_Frequency_Ranges[Plotted_Bank, j, 1])
        plot_desired_from_click(node_spectra_storage[:, Plotted_Bank, Plotted_Node, 0, :, :, :], node_spectra_storage[:, Plotted_Bank, Plotted_Node, 1, :, :, :], node_spectra_storage[:, Plotted_Bank, Plotted_Node, 2, :, :, :], OBSNCHAN, TBIN, samplesPerTransform, fftsPerIntegration, node_Frequency_Ranges[Plotted_Bank, Plotted_Node, 0], node_Frequency_Ranges[Plotted_Bank, Plotted_Node, 1], FILE_COUNT_INDICATOR - 1, THRESHOLD_PERCENTAGES[Plotted_Bank, Plotted_Node, 0, :], THRESHOLD_PERCENTAGES[Plotted_Bank, Plotted_Node, 1, :])

    #spectral flip for seemingly opposite increment on nodes
    if event.key == 'right':
        Plotted_Node -= 1
        if (Plotted_Node < 0):
            Plotted_Node = (numberOfNodes-1)
        clear_node_plots()
        for j in range(numberOfNodes):
            if (j!=Plotted_Node):
                plot_otherNodes(node_spectra_storage[FILE_COUNT_INDICATOR - 1, Plotted_Bank, j, 0, :, :, :], node_spectra_storage[FILE_COUNT_INDICATOR - 1, Plotted_Bank, j, 1, :, :, :], OBSNCHAN, samplesPerTransform, fftsPerIntegration, node_Frequency_Ranges[Plotted_Bank, j, 0], node_Frequency_Ranges[Plotted_Bank, j, 1])
        plot_desired_from_click(node_spectra_storage[:, Plotted_Bank, Plotted_Node, 0, :, :, :], node_spectra_storage[:, Plotted_Bank, Plotted_Node, 1, :, :, :], node_spectra_storage[:, Plotted_Bank, Plotted_Node, 2, :, :, :], OBSNCHAN, TBIN, samplesPerTransform, fftsPerIntegration, node_Frequency_Ranges[Plotted_Bank, Plotted_Node, 0], node_Frequency_Ranges[Plotted_Bank, Plotted_Node, 1], FILE_COUNT_INDICATOR - 1, THRESHOLD_PERCENTAGES[Plotted_Bank, Plotted_Node, 0, :], THRESHOLD_PERCENTAGES[Plotted_Bank, Plotted_Node, 1, :])

    if event.key == 'left':
        Plotted_Node += 1
        if (Plotted_Node > (numberOfNodes-1)):
            Plotted_Node = 0
        clear_node_plots()
        for j in range(numberOfNodes):
            if (j!=Plotted_Node):
                plot_otherNodes(node_spectra_storage[FILE_COUNT_INDICATOR - 1, Plotted_Bank, j, 0, :, :, :], node_spectra_storage[FILE_COUNT_INDICATOR - 1, Plotted_Bank, j, 1, :, :, :], OBSNCHAN, samplesPerTransform, fftsPerIntegration, node_Frequency_Ranges[Plotted_Bank, j, 0], node_Frequency_Ranges[Plotted_Bank, j, 1])
        plot_desired_from_click(node_spectra_storage[:, Plotted_Bank, Plotted_Node, 0, :, :, :], node_spectra_storage[:, Plotted_Bank, Plotted_Node, 1, :, :, :], node_spectra_storage[:, Plotted_Bank, Plotted_Node, 2, :, :, :], OBSNCHAN, TBIN, samplesPerTransform, fftsPerIntegration, node_Frequency_Ranges[Plotted_Bank, Plotted_Node, 0], node_Frequency_Ranges[Plotted_Bank, Plotted_Node, 1], FILE_COUNT_INDICATOR - 1, THRESHOLD_PERCENTAGES[Plotted_Bank, Plotted_Node, 0, :], THRESHOLD_PERCENTAGES[Plotted_Bank, Plotted_Node, 1, :])

    plt.suptitle(SESSION_IDENTIFIER + " | " + str(desiredFrequencyResolution/(10**6)) + " MHz, " + str(desiredTimeResolution*(10**3)) + " ms Resolution")
    plt.pause(0.000001)

######## Non - interactive

def clear_full_spectrum():

    global axis1_desired
    #del axis1_desired.lines[:]
    axis1_desired.clear()

def clear_node_plots():
    global axis2_desired, axis3_desired, axis4_desired, axis5_desired, axis6_desired, axis7_desired, axis8_desired, axis9_desired
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

def plot_real_time_visualization_desired(integrated_spectrum_x, integrated_spectrum_y, bandPass_x, bandPass_y, bandPass_cross, SK_x, SK_y, SK_cross, current_axis, lowerBound, upperBound, samplesPerTransform, fftsPerIntegration, TBIN, thresholdHitsX, thresholdHitsY):
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

    #totalTime = samplesPerTransform * fftsPerIntegration * TBIN * 10
    #GBT: 6 Hours, Parkes: 11 Hours
    totalTime = 0.5
    global axis1_desired, axis2_desired, axis3_desired, axis4_desired, axis5_desired, axis6_desired, axis7_desired, axis8_desired, axis9_desired, axis6_desired_twin, axis7_desired_twin
    global Plotted_Bank, Plotted_Node, colorbar4, colorbar5, PFA_Nita, FILE_COUNT_INDICATOR
    sk_lower_threshold, sk_upper_threshold = spectralKurtosis_thresholds(fftsPerIntegration, PFA_Nita)

    axis1_desired.set_title("blc" + str(Plotted_Bank + BANK_OFFSET) + "{0..7} Spectrum (X)")
    axis1_desired.plot(current_axis, 10*np.log10(bandPass_x), color = 'red')
    axis1_desired.set_ylabel("Power (dB)")
    axis1_desired.set_xlabel("Frequency (MHz)")
    axis1_desired.margins(x=0)

    axis2_desired.clear()
    axis2_desired.set_title("blc" + str(Plotted_Bank + BANK_OFFSET) + str(Plotted_Node) + " Spectrum: X")
    axis2_desired.set_xlabel("Frequency (MHz)")
    axis2_desired.set_ylabel("Power (dB)")
    axis2_desired.margins(x=0)
    axis2_desired.plot(current_axis, 10*np.log10(bandPass_x), color = 'C0')

    axis3_desired.clear()
    axis3_desired.set_title("blc" + str(Plotted_Bank + BANK_OFFSET) + str(Plotted_Node) + " Spectrum: Y")
    axis3_desired.set_xlabel("Frequency (MHz)")
    axis3_desired.set_ylabel("Power (dB)")
    axis3_desired.margins(x=0)
    axis3_desired.plot(current_axis, 10*np.log10(bandPass_y), color = 'C0')

    im4 = axis4_desired.imshow(10*np.log10(integrated_spectrum_x), cmap = 'viridis', aspect = 'auto', extent = [lowerBound, upperBound, totalTime, 0])
    divider4 = make_axes_locatable(axis4_desired)
    cax4 = divider4.append_axes('right', size = '5%', pad = 0.05)
    axis4_desired.set_ylabel("Time (Hours)")
    axis4_desired.set_xlabel("Frequency (MHz)")
    axis4_desired.set_title("blc" + str(Plotted_Bank + BANK_OFFSET) + str(Plotted_Node) + " Waterfall: X")
    if (colorbar4==0):
        colorbar4 = plt.colorbar(im4, cax=cax4, orientation = 'vertical')
        colorbar4.set_label("Power (dB)")
    else:
        colorbar4.remove()
        colorbar4 = plt.colorbar(im4, cax=cax4, orientation='vertical')
        colorbar4.set_label("Power (dB)")

    im5 = axis5_desired.imshow(10*np.log10(integrated_spectrum_y), cmap = 'viridis', aspect = 'auto', extent = [lowerBound, upperBound, totalTime, 0])
    divider5 = make_axes_locatable(axis5_desired)
    cax5 = divider5.append_axes('right', size = '5%', pad = 0.05)
    axis5_desired.set_ylabel("Time (Hours)")
    axis5_desired.set_xlabel("Frequency (MHz)")
    axis5_desired.set_title("blc" + str(Plotted_Bank + BANK_OFFSET) + str(Plotted_Node) + " Waterfall: Y")
    if (colorbar5==0):
        colorbar5 = plt.colorbar(im5, cax=cax5, orientation='vertical')
        colorbar5.set_label("Power (dB)")
    else:
        colorbar5.remove()
        colorbar5 = plt.colorbar(im5, cax=cax5, orientation='vertical')
        colorbar5.set_label("Power (dB)")

    axis6_desired.clear()
    axis6_desired.plot(current_axis, SK_x, color = 'C0')
    axis6_desired.set_title("blc" + str(Plotted_Bank + BANK_OFFSET) + str(Plotted_Node) + " Spectral Kurtosis: X")
    axis6_desired.margins(x=0)
    axis6_desired.set_ylim(-0.5, 5)
    axis6_desired.axhline(y=sk_upper_threshold, color = 'y')
    axis6_desired.axhline(y=sk_lower_threshold, color = 'y')
    axis6_desired.set_xlabel("Frequency (MHz)")
    axis6_desired.lines[1].set_label('Gaussian Thresholds')
    axis6_desired.legend(loc = 1)
    axis6_desired.set_ylabel("SK Value", color='C0')
    axis6_desired.text(0, -0.08, "M = " + str(fftsPerIntegration) + " | N = 1 | D = 1 | PFA = " + str(PFA_Nita), fontsize = "8")

    axis6_desired_twin.clear()
    axis6_desired_twin.plot(thresholdHitsX/FILE_COUNT_INDICATOR, color='lightcoral')
    axis6_desired_twin.set_ylabel('Threshold Hits (%)', color='r')
    axis6_desired_twin.tick_params('y', colors='lightcoral')

    axis7_desired.clear()
    axis7_desired.plot(current_axis, SK_y, color = 'C0')
    axis7_desired.set_title("blc" + str(Plotted_Bank + BANK_OFFSET) + str(Plotted_Node) + " Spectral Kurtosis: Y")
    axis7_desired.margins(x=0)
    axis7_desired.set_ylim(-0.5, 5)
    axis7_desired.axhline(y=sk_upper_threshold, color = 'y')
    axis7_desired.axhline(y=sk_lower_threshold, color = 'y')
    axis7_desired.set_xlabel("Frequency (MHz)")
    axis7_desired.lines[1].set_label('Gaussian Thresholds')
    axis7_desired.legend(loc = 1)
    axis7_desired.set_ylabel("SK Value", color='C0')
    axis7_desired.text(0, -0.08, "M = " + str(fftsPerIntegration) + " | N = 1 | D = 1 | PFA = " + str(PFA_Nita), fontsize = "8")

    axis7_desired_twin.clear()
    axis7_desired_twin.plot(thresholdHitsY/FILE_COUNT_INDICATOR, color='lightcoral')
    axis7_desired_twin.set_ylabel('Threshold Hits (%)', color='r')
    axis7_desired_twin.tick_params('y', colors='lightcoral')

    # Cross Spectrum and SK of cross spectrum
    axis8_desired.clear()
    axis8_desired.plot(current_axis, bandPass_cross, color = 'C0')
    axis8_desired.set_title("blc" + str(Plotted_Bank + BANK_OFFSET) + str(Plotted_Node) + " Cross-Spectrum")
    axis8_desired.margins(x=0)
    axis8_desired.set_xlabel("Frequency (MHz)")
    axis8_desired.set_ylabel("Power (Linear)")

    axis9_desired.clear()
    axis9_desired.plot(current_axis, 10*np.log10(SK_cross), color = 'C0')
    axis9_desired.set_title("blc" + str(Plotted_Bank + BANK_OFFSET) + str(Plotted_Node) + " Spectral Kurtosis: Cross-Spectrum")
    axis9_desired.margins(x=0)
    #axis9_desired.set_ylim(0, 5)
    axis9_desired.set_xlabel("Frequency (MHz)")

    plt.connect('key_press_event', press)
    plt.pause(0.000001)

def plot_desired_from_click(spectralData_x, spectralData_y, spectralData_cross, OBSNCHAN, TBIN, samplesPerTransform, fftsPerIntegration, lowerBound, upperBound, file_index, thresholdHitsX, thresholdHitsY):

    global most_possible_files_read

    # Spectral Kurtosis
    SK_x = calculate_spectralKurtosis(spectralData_x[file_index, :, :, :], fftsPerIntegration)
    SK_y = calculate_spectralKurtosis(spectralData_y[file_index, :, :, :], fftsPerIntegration)
    SK_cross = calculate_spectralKurtosis(spectralData_cross[file_index, :, :, :], fftsPerIntegration)
    SK_x = np.flip(SK_x, 0).reshape(-1)
    SK_y = np.flip(SK_y, 0).reshape(-1)
    SK_cross = np.flip(SK_cross, 0).reshape(-1)

    # Spectral flip
    tempx = np.flip(np.sum(spectralData_x[:, :, :, :], axis = 2), 1)
    tempy = np.flip(np.sum(spectralData_y[:, :, :, :], axis = 2), 1)

    waterfall_spectrum_x = np.zeros((most_possible_files_read, OBSNCHAN * samplesPerTransform))
    waterfall_spectrum_y = np.zeros((most_possible_files_read, OBSNCHAN * samplesPerTransform))

    for id in range(file_index + 1):
        waterfall_spectrum_x[id,:] = tempx[id, :, :].reshape(-1)
        waterfall_spectrum_y[id,:] = tempy[id, :, :].reshape(-1)

    # Spectrum for plotting
    bandPass_x = waterfall_spectrum_x[file_index, :]
    bandPass_y = waterfall_spectrum_y[file_index, :]
    bandPass_cross = np.flip(np.sum(spectralData_cross[file_index, :, :, :], axis=1), 0).reshape(-1)

    current_RAW_axis = np.linspace(lowerBound, upperBound, OBSNCHAN *samplesPerTransform)
    plot_real_time_visualization_desired(waterfall_spectrum_x, waterfall_spectrum_y, bandPass_x, bandPass_y, bandPass_cross, SK_x, SK_y, SK_cross, current_RAW_axis, lowerBound, upperBound, samplesPerTransform, fftsPerIntegration, TBIN, thresholdHitsX, thresholdHitsY)

def plot_desired(spectralData_x, spectralData_y, spectralData_cross, OBSNCHAN, TBIN, samplesPerTransform, fftsPerIntegration, lowerBound, upperBound, file_index, thresholdHitsX, thresholdHitsY):

    global most_possible_files_read

    # Spectral Kurtosis
    SK_x = calculate_spectralKurtosis(spectralData_x, fftsPerIntegration)
    SK_y = calculate_spectralKurtosis(spectralData_y, fftsPerIntegration)
    SK_cross = calculate_spectralKurtosis(spectralData_cross, fftsPerIntegration)
    SK_x = np.flip(SK_x, 0).reshape(-1)
    SK_y = np.flip(SK_y, 0).reshape(-1)
    SK_cross = np.flip(SK_cross, 0).reshape(-1)

    # Spectral flip
    spectralData_x = np.flip(np.sum(spectralData_x, axis = 1), 0)
    spectralData_y = np.flip(np.sum(spectralData_y, axis = 1), 0)

    # Spectrum for waterfall (array in array for plt.imshow())
    waterfall_spectrum_x = np.zeros((most_possible_files_read, OBSNCHAN * samplesPerTransform))
    waterfall_spectrum_y = np.zeros((most_possible_files_read, OBSNCHAN * samplesPerTransform))
    waterfall_spectrum_x[file_index, :] = spectralData_x.reshape(-1)
    waterfall_spectrum_y[file_index, :] = spectralData_y.reshape(-1)

    # Spectrum for plotting
    bandPass_x = np.sum(waterfall_spectrum_x, 0)
    bandPass_y = np.sum(waterfall_spectrum_y, 0)
    bandPass_cross = np.flip(np.sum(spectralData_cross, axis = 1), 0).reshape(-1)

    current_RAW_axis = np.linspace(lowerBound, upperBound, OBSNCHAN *samplesPerTransform)
    plot_real_time_visualization_desired(waterfall_spectrum_x, waterfall_spectrum_y, bandPass_x, bandPass_y, bandPass_cross, SK_x, SK_y, SK_cross, current_RAW_axis, lowerBound, upperBound, samplesPerTransform, fftsPerIntegration, TBIN, thresholdHitsX, thresholdHitsY)

def plot_otherNodes(spectralData_x, spectralData_y, OBSNCHAN, samplesPerTransform, fftsPerIntegration, lowerBound, upperBound, plot_color = 'black'):
    """
    Calculate spectra of all active nodes (except node of interest)
    """

    bandPass_x = np.flip(np.sum(spectralData_x, 1),0).reshape(-1)
    #bandPass_y = np.sum(np.flip(np.sum(spectralData_y, 1),0), 0)

    current_RAW_axis = np.linspace(lowerBound, upperBound, OBSNCHAN *samplesPerTransform)

    plot_real_time_visualization_general(current_RAW_axis, bandPass_x, plot_color)

################################################################################
#######################---Program---############################################
################################################################################

if __name__ == "__main__":
    global Plotted_Bank, Plotted_Node
    global node_Frequency_Ranges, node_spectra_storage
    global FILE_COUNT_INDICATOR, TBIN, Polarization_Plot, colorbar4, colorbar5
    global most_possible_files_read, BANK_OFFSET, numberOfNodes, numberOfBanks
    global OBSNCHAN, fftsPerIntegration, samplesPerTransform, SESSION_IDENTIFIER, PFA_Nita
    global OBSERVATION_IS_RUNNING, desiredFrequencyResolution, desiredTimeResolution
    #GBT - 6 hours; 20s files
    most_possible_files_read = 80
    OBSERVATION_IS_RUNNING = True
    colorbar4 = 0
    colorbar5 = 0
    Polarization_Plot = 0
    Plotted_Bank = 0
    Plotted_Node = 0

    PFA_Nita = 0.0013499

    #User inputted resolutions
    desiredFrequencyResolution = 183105 #16 Bins
    desiredTimeResolution = 0.0003 #54 Integrations
    samplesPerTransform = 16
    fftsPerIntegration = 54
    OBSNCHAN = 64
    dualPolarization = 2

    startTime = datetime.now().strftime('%H:%M')

    ########################
    ### Shell commands

    #Find the nodes
    p = subprocess.check_output(['cat', '/home/obs/triggers/hosts_running'])
    p = p.split()
    BANK_OFFSET = int(p[0][-2])
    numberOfBanks = (int(p[len(p)-1][-2]) - BANK_OFFSET) + 1
    numberOfNodes = int(len(p)/numberOfBanks)

    #Find the session
    string_for_session = 'ls -trd /mnt_blc' + p[0][-2:] + '/datax/dibas/* | tail -1'
    SESSION_IDENTIFIER = subprocess.check_output(string_for_session, shell = True)[23:-1]

    ########################

    node_Frequency_Ranges = np.zeros((numberOfBanks, numberOfNodes, 2))
    node_spectra_storage = np.zeros((most_possible_files_read, numberOfBanks, numberOfNodes, 3, OBSNCHAN, fftsPerIntegration, samplesPerTransform))
    THRESHOLD_PERCENTAGES = np.zeros((numberOfBanks, numberOfNodes, 2, OBSNCHAN * samplesPerTransform))


    #Initialize Plot
    #SET UP Big Plot -- Can vary how we want big plot to look by adjusting subplot2grid
    plt.figure("Test")
    plt.suptitle(SESSION_IDENTIFIER + " | " + str(desiredFrequencyResolution/(10**6)) + " MHz, " + str(desiredTimeResolution*(10**3)) + " ms Resolution")
    plt.ion()
    plt.show()

    # Full observational range
    axis1_desired = plt.subplot2grid((19,15), (0,3), colspan=9, rowspan=3)
    axis1_desired.set_title("blc" + str(Plotted_Bank + BANK_OFFSET) + "{0..7} Spectrum (X)")
    axis1_desired.set_ylabel("Power (dB)")
    axis1_desired.set_xlabel("Frequency (MHz)")
    axis1_desired.margins(x=0)

    # Spectra of compute node
    axis2_desired = plt.subplot2grid((19,15), (5,3), colspan=4, rowspan=3)
    axis2_desired.set_title("blc" + str(Plotted_Bank + BANK_OFFSET) + str(Plotted_Node) + " Spectrum: X")
    axis2_desired.set_xlabel("Frequency (MHz)")
    axis2_desired.set_ylabel("Power (dB)")
    axis2_desired.margins(x=0)

    axis3_desired = plt.subplot2grid((19,15), (5, 8), colspan=4, rowspan=3)
    axis3_desired.set_title("blc" + str(Plotted_Bank + BANK_OFFSET) + str(Plotted_Node) + " Spectrum: Y")
    axis3_desired.set_xlabel("Frequency (MHz)")
    axis3_desired.set_ylabel("Power (dB)")
    axis3_desired.margins(x=0)

    # Waterfall of compute node
    axis4_desired = plt.subplot2grid((19,15), (0, 0), colspan=2, rowspan=19)
    axis4_desired.set_title("blc" + str(Plotted_Bank + BANK_OFFSET) + str(Plotted_Node) + " Waterfall: X")
    axis4_desired.set_xlabel("Frequency (MHz)")
    axis4_desired.set_ylabel("Time (Hours)")
    axis4_desired.margins(x=0)

    axis5_desired = plt.subplot2grid((19,15), (0, 13), colspan=2, rowspan=19)
    axis5_desired.set_title("blc" + str(Plotted_Bank + BANK_OFFSET) + str(Plotted_Node) + " Waterfall: Y")
    axis5_desired.set_xlabel("Frequency (MHz)")
    axis5_desired.set_ylabel("Time (Hours)")
    axis5_desired.margins(x=0)

    # Spectral Kurtosis of compute node
    axis6_desired = plt.subplot2grid((19,15), (10,3), colspan=4, rowspan=3)
    axis6_desired.set_title("blc" + str(Plotted_Bank + BANK_OFFSET) + str(Plotted_Node) + " Spectral Kurtosis: X")
    axis6_desired.margins(x=0)
    axis6_desired.set_ylim(-0.5, 5)
    axis6_desired.set_xlabel("Frequency (MHz)")
    axis6_desired.set_ylabel("SK Value", color='C0')
    axis6_desired.text(0, -0.08, "M = " + str(fftsPerIntegration) + " | N = 1 | D = 1 | PFA = " + str(PFA_Nita), fontsize = "8")

    axis6_desired_twin = axis6_desired.twinx()
    ##plottt
    axis6_desired_twin.set_ylabel('Threshold Hits (%)', color='r')
    axis6_desired_twin.tick_params('y', colors='lightcoral')


    axis7_desired = plt.subplot2grid((19,15), (10, 8), colspan=4, rowspan=3)
    axis7_desired.set_title("blc" + str(Plotted_Bank + BANK_OFFSET) + str(Plotted_Node) + " Spectral Kurtosis: Y")
    axis7_desired.margins(x=0)
    axis7_desired.set_ylim(-0.5, 5)
    axis7_desired.set_xlabel("Frequency (MHz)")
    axis7_desired.set_ylabel("SK Value", color='C0')
    axis7_desired.text(0, -0.08, "M = " + str(fftsPerIntegration) + " | N = 1 | D = 1 | PFA = " + str(PFA_Nita), fontsize = "8")

    axis7_desired_twin = axis7_desired.twinx()
    ##plottt
    axis7_desired_twin.set_ylabel('Threshold Hits (%)', color='r')
    axis7_desired_twin.tick_params('y', colors='lightcoral')

    # Cross Spectrum and SK of cross spectrum
    axis8_desired = plt.subplot2grid((19,15), (15, 3), colspan=4, rowspan=3)
    axis8_desired.set_title("blc" + str(Plotted_Bank + BANK_OFFSET) + str(Plotted_Node) + " Cross-Spectrum")
    axis8_desired.margins(x=0)
    axis8_desired.set_xlabel("Frequency (MHz)")
    axis8_desired.set_ylabel("Power (dB)")

    axis9_desired = plt.subplot2grid((19,15), (15, 8), colspan=4, rowspan=3)
    axis9_desired.set_title("blc" + str(Plotted_Bank + BANK_OFFSET) + str(Plotted_Node) + " Spectral Kurtosis: Cross-Spectrum")
    axis9_desired.margins(x=0)
    axis9_desired.set_ylim(0, 5)
    axis9_desired.set_xlabel("Frequency (MHz)")

    plt.connect('key_press_event', press)


    FILE_COUNT_INDICATOR = 0

    while(OBSERVATION_IS_RUNNING):

        for bank in range(numberOfBanks):
            bank = bank + BANK_OFFSET
            for node in range(numberOfNodes):
                test_Number_Files_String = 'ls /mnt_blc' + str(bank) + str(node) + '/datax/dibas/' + str(SESSION_IDENTIFIER) + '/GUPPI/BLP' + str(bank - BANK_OFFSET) + str(node) + '/*.raw | wc -l'
                waiting_for_written_file = True

                while(waiting_for_written_file):
                    if (int(subprocess.check_output(test_Number_Files_String, shell=True)[:-1]) > (FILE_COUNT_INDICATOR + 1)):
                        waiting_for_written_file = False
                    else:
                        plt.pause(2)
                        if (OBSERVATION_IS_RUNNING == False):
                            break

                if (OBSERVATION_IS_RUNNING == False):
                    break

                test_input_file_string = 'ls -trd /mnt_blc' + str(bank) + str(node) + '/datax/dibas/' + str(SESSION_IDENTIFIER) + '/GUPPI/BLP' + str(bank - BANK_OFFSET) + str(node) + '/*.raw | tail -2 | head -1'
                inputFileName = subprocess.check_output(test_input_file_string, shell = True)[:-1]
                readIn = np.memmap(inputFileName, dtype = 'int8', mode = 'r')
                fileBytes = os.path.getsize(inputFileName)
                currentBytesPassed = 0

                OBSNCHAN, NPOL, NBITS, BLOCSIZE, OBSFREQ, CHAN_BW, OBSBW, TBIN, headerOffset = extractHeader(readIn, currentBytesPassed)
                NDIM = int(BLOCSIZE/(OBSNCHAN*NPOL*(NBITS/8)))

                samplesPerTransform, fftsPerIntegration = convert_resolution(desiredFrequencyResolution, desiredTimeResolution, TBIN)
                dataBuffer = readIn[(currentBytesPassed + headerOffset):(currentBytesPassed + headerOffset + BLOCSIZE)].reshape(OBSNCHAN, NDIM, NPOL)
                NDIMsmall = samplesPerTransform * fftsPerIntegration

                node_spectra_storage[FILE_COUNT_INDICATOR, bank - BANK_OFFSET, node, 0, :, :, :], node_spectra_storage[FILE_COUNT_INDICATOR, bank - BANK_OFFSET, node, 1, :, :, :], node_spectra_storage[FILE_COUNT_INDICATOR, bank - BANK_OFFSET, node, 2, :, :, :], node_Frequency_Ranges[bank - BANK_OFFSET, node, 0], node_Frequency_Ranges[bank - BANK_OFFSET, node, 1] = spectra_Find_All(dataBuffer[:, 0:NDIMsmall, :], OBSNCHAN, samplesPerTransform, fftsPerIntegration, OBSFREQ, OBSBW)

                x_temp_indices = find_SK_threshold_hits(node_spectra_storage[FILE_COUNT_INDICATOR, bank - BANK_OFFSET, node, 0, :, :, :], fftsPerIntegration)
                np.add.at(THRESHOLD_PERCENTAGES[bank - BANK_OFFSET, node, 0, :], x_temp_indices, 1)
                y_temp_indices = find_SK_threshold_hits(node_spectra_storage[FILE_COUNT_INDICATOR, bank - BANK_OFFSET, node, 1, :, :, :], fftsPerIntegration)
                np.add.at(THRESHOLD_PERCENTAGES[bank - BANK_OFFSET, node, 1, :], y_temp_indices, 1)

                del readIn

            if (OBSERVATION_IS_RUNNING == False):
                break
        if (OBSERVATION_IS_RUNNING == False):
            break

        if (FILE_COUNT_INDICATOR>0):
            clear_full_spectrum()
        ## Done with spectra collection; plot
        for i in range(numberOfNodes):
            if (i!=Plotted_Node):
                plot_otherNodes(node_spectra_storage[FILE_COUNT_INDICATOR, Plotted_Bank, i, 0, :, :, :], node_spectra_storage[FILE_COUNT_INDICATOR, Plotted_Bank, i, 1, :, :, :], OBSNCHAN, samplesPerTransform, fftsPerIntegration, node_Frequency_Ranges[Plotted_Bank, i, 0], node_Frequency_Ranges[Plotted_Bank, i, 1])

        plot_desired(node_spectra_storage[FILE_COUNT_INDICATOR, Plotted_Bank, Plotted_Node, 0, :, :, :], node_spectra_storage[FILE_COUNT_INDICATOR, Plotted_Bank, Plotted_Node, 1, :, :, :], node_spectra_storage[FILE_COUNT_INDICATOR, Plotted_Bank, Plotted_Node, 2, :, :, :], OBSNCHAN, TBIN, samplesPerTransform, fftsPerIntegration, node_Frequency_Ranges[Plotted_Bank, Plotted_Node, 0], node_Frequency_Ranges[Plotted_Bank, Plotted_Node, 1], FILE_COUNT_INDICATOR, THRESHOLD_PERCENTAGES[Plotted_Bank, Plotted_Node, 0, :], THRESHOLD_PERCENTAGES[Plotted_Bank, Plotted_Node, 1, :])

        print(datetime.now().strftime('%H:%M:%S'))

        FILE_COUNT_INDICATOR += 1

        if (FILE_COUNT_INDICATOR >= most_possible_files_read):
            OBSERVATION_IS_RUNNING = False

    plt.close()

    endTime = datetime.now().strftime('%H:%M')


    #
    #
    #

    ###################################################################################
    ####################----EXPORT WATERFALLS----######################################
    ###################################################################################

    #
    #
    #

    ### Fix band identifier
    BAND_IDENTIFIER = ''

    if (numberOfBanks == 3):
        BAND_IDENTIFIER = 'X'
    elif (numberOfBanks == 4):
        BAND_IDENTIFIER = 'C'
    else:
        if ((node_Frequency_Ranges[0,7,0]/10**3) < 1.2):
            BAND_IDENTIFIER = 'L'
        else:
            BAND_IDENTIFIER = 'S'

    pp = PdfPages("../ObservationWaterfalls/" + str(SESSION_IDENTIFIER) + "_" + str(BAND_IDENTIFIER) + "-band_" + str(startTime.replace(":", "")) + "-" + str(endTime.replace(":", "")) + "_waterfall.pdf")

    for export_bank in range(numberOfBanks):
        for export_node in range(numberOfNodes):


            #### Set up data
            export_tempx = np.flip(np.sum(node_spectra_storage[:, export_bank, export_node, 0, :, :, :], axis = 2), 1)
            export_tempy = np.flip(np.sum(node_spectra_storage[:, export_bank, export_node, 1, :, :, :], axis = 2), 1)

            export_waterfall_spectrum_x = np.zeros((FILE_COUNT_INDICATOR, OBSNCHAN * samplesPerTransform))
            export_waterfall_spectrum_y = np.zeros((FILE_COUNT_INDICATOR, OBSNCHAN * samplesPerTransform))

            for id in range(FILE_COUNT_INDICATOR):
                export_waterfall_spectrum_x[id,:] = export_tempx[id, :, :].reshape(-1)
                export_waterfall_spectrum_y[id,:] = export_tempy[id, :, :].reshape(-1)
            ########

            ###### Set up plot
            export_fig = plt.figure()
            plt.suptitle("blc" + str(export_bank + BANK_OFFSET) + str(export_node) + " | " + str(desiredFrequencyResolution/(10**6)) + " MHz, " + str(desiredTimeResolution*(10**3)) + " ms Resolution")

            export_axis1 = plt.subplot2grid((14,5), (0, 0), colspan=2, rowspan=14)
            export_axis1.set_title("X")
            export_axis1.set_xlabel("Frequency (MHz)")
            #export_axis1.set_ylabel("Time (Hours)")
            export_axis1.margins(x=0)

            export_axis2 = plt.subplot2grid((14,5), (0, 3), colspan=2, rowspan=14)
            export_axis2.set_title("Y")
            export_axis2.set_xlabel("Frequency (MHz)")
            #export_axis2.set_ylabel("Time (Hours)")
            export_axis2.margins(x=0)
            ########

            ####### Plot data
            export_im_x = export_axis1.imshow(10*np.log10(export_waterfall_spectrum_x), cmap = 'viridis', aspect = 'auto', extent = [node_Frequency_Ranges[export_bank, export_node, 0], node_Frequency_Ranges[export_bank, export_node, 1], 1, 0])
            export_divider_x = make_axes_locatable(export_axis1)
            export_cax_x = export_divider_x.append_axes('right', size = '5%', pad = 0.05)
            export_colorbar_x = plt.colorbar(export_im_x, cax=export_cax_x, orientation = 'vertical')
            export_colorbar_x.set_label("Power (dB)")
            export_axis1.set_yticks([0,1])
            export_axis1.set_yticklabels([startTime, endTime])

            export_im_y = export_axis2.imshow(10*np.log10(export_waterfall_spectrum_y), cmap = 'viridis', aspect = 'auto', extent = [node_Frequency_Ranges[export_bank, export_node, 0], node_Frequency_Ranges[export_bank, export_node, 1], 1, 0])
            export_divider_y = make_axes_locatable(export_axis2)
            export_cax_y = export_divider_y.append_axes('right', size = '5%', pad = 0.05)
            export_colorbar_y = plt.colorbar(export_im_y, cax=export_cax_y, orientation = 'vertical')
            export_colorbar_y.set_label("Power (dB)")
            export_axis2.set_yticks([0,1])
            export_axis2.set_yticklabels([startTime, endTime])
            ########

            ######## Write to PDF
            plt.close()
            pp.savefig(export_fig)
            ########

    pp.close()
