# Nicholas Joslyn
# Breakthrough Listen UC Berkeley SETI Intern 2018

# Quasi-real time BL RAW data visualization for BL observations

from __future__ import division
import numpy as np
import os

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.collections import LineCollection

################################################################################
######################---Functions---###########################################
################################################################################

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
          DIRECTIO = int(cardString[9:].strip())

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

def clear_full_spectrum():

    global axis1_desired
    del axis1_desired.lines[:]


def plot_real_time_visualization_general(current_axis, bandPass_x):
    """
    Plot the top panel -- full spectrum of all active nodes (except node of interest)
    """
    global axis1_desired, axis2_desired, axis3_desired, axis4_desired, axis5_desired, axis6_desired, axis7_desired
    if(plt.fignum_exists("Test") == False):
        #SET UP Big Plot
        plt.figure("Test")
        plt.suptitle("Real-Time Spectra of Observation")
        plt.ion()
        plt.show()

        # Full observational range
        axis1_desired = plt.subplot2grid((18,5), (0,0), colspan=5, rowspan=3)
        axis1_desired.set_title("Full Observation Spectrum (X)")
        axis1_desired.set_yscale("log")
        axis1_desired.set_ylabel("Power")
        axis1_desired.set_xlabel("Frequency (MHz)")
        axis1_desired.plot(current_axis, bandPass_x, color = 'black')

        # Spectra of compute node
        axis2_desired = plt.subplot2grid((18,5), (5,0), colspan=2, rowspan=3)
        axis2_desired.set_title("Node Spectrum: X")
        axis2_desired.set_xlabel("Frequency (MHz)")
        axis2_desired.set_ylabel("Power")
        axis2_desired.set_yscale('log')
        axis2_desired.margins(x=0)
        #axis2_desired.plot(current_axis, bandPass_x)

        axis3_desired = plt.subplot2grid((18,5), (5, 3), colspan=2, rowspan=3)
        axis3_desired.set_title("Node Spectrum: Y")
        axis3_desired.set_xlabel("Frequency (MHz)")
        axis3_desired.set_ylabel("Power")
        axis3_desired.set_yscale('log')
        axis3_desired.margins(x=0)
        #axis3_desired.plot(current_axis, bandPass_y)

        # Waterfall of compute node
        axis4_desired = plt.subplot2grid((18,5), (10, 0), colspan=2, rowspan=3)
        axis4_desired.set_title("Node Waterfall: X")
        #axis4_desired.imshow(integrated_spectrum_x, cmap = 'viridis', aspect = 'auto', norm = LogNorm(), extent = [lowerBound, upperBound, totalTime, 0])
        axis4_desired.set_xlabel("Frequency (MHz)")
        axis4_desired.set_ylabel("Time (s)")
        axis4_desired.margins(x=0)
        #plt.colorbar(im, ax=ax4)

        axis5_desired = plt.subplot2grid((18,5), (10, 3), colspan=2, rowspan=3)
        axis5_desired.set_title("Node Waterfall: Y")
        #axis5_desired.imshow(integrated_spectrum_y, cmap = 'viridis', aspect = 'auto', norm = LogNorm(), extent = [lowerBound, upperBound, totalTime, 0])
        axis5_desired.set_xlabel("Frequency (MHz)")
        axis5_desired.set_ylabel("Time (s)")
        axis5_desired.margins(x=0)

        # Spectral Kurtosis of compute node
        axis6_desired = plt.subplot2grid((18,5), (15,0), colspan=2, rowspan=3)
        #axis6_desired.plot(current_axis, SK_x)
        axis6_desired.set_title("Spectral Kurtosis: X")
        axis6_desired.margins(x=0)
        axis6_desired.set_xlabel("Frequency (MHz)")

        axis7_desired = plt.subplot2grid((18,5), (15, 3), colspan=2, rowspan=3)
        #axis7_desired.plot(current_axis, SK_y)
        axis7_desired.set_title("Spectral Kurtosis: Y")
        axis7_desired.margins(x=0)
        axis7_desired.set_xlabel("Frequency (MHz)")

    else:
    	axis1_desired.plot(current_axis, bandPass_x, color = 'black')


def real_time_spectra_general(BLOCK, OBSNCHAN, samplesPerTransform, fftsPerIntegration, OBSFREQ, OBSBW):
    """
    Calculate spectra of all active nodes (except node of interest)
    """


    BLOCK = remove_DCoffset(BLOCK)
    spectralData_x, spectralData_y = calculate_spectra(BLOCK, OBSNCHAN, fftsPerIntegration, samplesPerTransform)
    bandPass_x = np.flip(np.sum(spectralData_x, 1),0).reshape(-1)
    #bandPass_y = np.sum(np.flip(np.sum(spectralData_y, 1),0), 0)

    # Helpful plotting values
    lowerBound = OBSFREQ + OBSBW/2
    upperBound = OBSFREQ - OBSBW/2
    current_RAW_axis = np.linspace(lowerBound, upperBound, OBSNCHAN *samplesPerTransform)

    plot_real_time_visualization_general(current_RAW_axis, bandPass_x)


def plot_real_time_visualization_desired(integrated_spectrum_x, integrated_spectrum_y, bandPass_x, bandPass_y, SK_x, SK_y, current_axis, lowerBound, upperBound, samplesPerTransform, fftsPerIntegration, TBIN):
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

    totalTime = samplesPerTransform * fftsPerIntegration * TBIN * 10

    global axis1_desired, axis2_desired, axis3_desired, axis4_desired, axis5_desired, axis6_desired, axis7_desired
    sk_lower_threshold, sk_upper_threshold = SKThresholds.spectralKurtosis_thresholds(fftsPerIntegration)

    if (plt.fignum_exists("Test") == False):
        #SET UP Big Plot
        plt.figure("Test")
        plt.suptitle("Real-Time Spectra of Observation")
        plt.ion()
        plt.show()

        # Full observational range
        axis1_desired = plt.subplot2grid((18,5), (0,0), colspan=5, rowspan=3)
        axis1_desired.set_title("Full Observation Spectrum (X)")
        axis1_desired.set_yscale("log")
        axis1_desired.set_ylabel("Power")
        axis1_desired.set_xlabel("Frequency (MHz)")
        axis1_desired.plot(current_axis, bandPass_x, color = 'red')

        # Spectra of compute node
        axis2_desired = plt.subplot2grid((18,5), (5,0), colspan=2, rowspan=3)
        axis2_desired.set_title("Node Spectrum: X")
        axis2_desired.set_xlabel("Frequency (MHz)")
        axis2_desired.set_ylabel("Power")
        axis2_desired.set_yscale('log')
        axis2_desired.margins(x=0)
        axis2_desired.plot(current_axis, bandPass_x, color = 'C0')

        axis3_desired = plt.subplot2grid((18,5), (5, 3), colspan=2, rowspan=3)
        axis3_desired.set_title("Node Spectrum: Y")
        axis3_desired.set_xlabel("Frequency (MHz)")
        axis3_desired.set_ylabel("Power")
        axis3_desired.set_yscale('log')
        axis3_desired.margins(x=0)
        axis3_desired.plot(current_axis, bandPass_y, color = 'C0')

        # Waterfall of compute node
        axis4_desired = plt.subplot2grid((18,5), (10, 0), colspan=2, rowspan=3)
        axis4_desired.set_title("Node Waterfall: X")
        axis4_desired.imshow(integrated_spectrum_x, cmap = 'viridis', aspect = 'auto', norm = LogNorm(), extent = [lowerBound, upperBound, totalTime, 0])
        axis4_desired.set_xlabel("Frequency (MHz)")
        axis4_desired.set_ylabel("Time (s)")
        axis4_desired.margins(x=0)
        #plt.colorbar(im, ax=ax4)

        axis5_desired = plt.subplot2grid((18,5), (10, 3), colspan=2, rowspan=3)
        axis5_desired.set_title("Node Waterfall: Y")
        axis5_desired.imshow(integrated_spectrum_y, cmap = 'viridis', aspect = 'auto', norm = LogNorm(), extent = [lowerBound, upperBound, totalTime, 0])
        axis5_desired.set_xlabel("Frequency (MHz)")
        axis5_desired.set_ylabel("Time (s)")
        axis5_desired.margins(x=0)

        # Spectral Kurtosis of compute node
        axis6_desired = plt.subplot2grid((18,5), (15,0), colspan=2, rowspan=3)
        axis6_desired.plot(current_axis, SK_x, color = 'C0')
        axis6_desired.set_title("Spectral Kurtosis: X")
        axis6_desired.margins(x=0)
        axis6_desired.axhline(y=sk_upper_threshold, color = 'y')
        axis6_desired.axhline(y=sk_lower_threshold, color = 'y')
        axis6_desired.set_xlabel("Frequency (MHz)")

        axis7_desired = plt.subplot2grid((18,5), (15, 3), colspan=2, rowspan=3)
        axis7_desired.plot(current_axis, SK_y, color = 'C0')
        axis7_desired.set_title("Spectral Kurtosis: Y")
        axis7_desired.margins(x=0)
        axis7_desired.axhline(y=sk_upper_threshold, color = 'y')
        axis7_desired.axhline(y=sk_lower_threshold, color = 'y')
        axis7_desired.set_xlabel("Frequency (MHz)")

    else:
        axis1_desired.plot(current_axis, bandPass_x, color = 'red')
        del axis2_desired.lines[:]
        axis2_desired.plot(current_axis, bandPass_x, color = 'C0')
        del axis3_desired.lines[:]
        axis3_desired.plot(current_axis, bandPass_y, color = 'C0')
        axis4_desired.imshow(integrated_spectrum_x, cmap = 'viridis', aspect = 'auto', norm = LogNorm(), extent = [lowerBound, upperBound, totalTime, 0])
        axis5_desired.imshow(integrated_spectrum_y, cmap = 'viridis', aspect = 'auto', norm = LogNorm(), extent = [lowerBound, upperBound, totalTime, 0])
        del axis6_desired.lines[:]
        axis6_desired.axhline(y=sk_upper_threshold, color = 'y')
        axis6_desired.axhline(y=sk_lower_threshold, color = 'y')
        axis6_desired.plot(current_axis, SK_x, color = 'C0')
        del axis7_desired.lines[:]
        axis7_desired.axhline(y=sk_upper_threshold, color = 'y')
        axis7_desired.axhline(y=sk_lower_threshold, color = 'y')
        axis7_desired.plot(current_axis, SK_y, color = 'C0')

    plt.connect('key_press_event', press)
    plt.pause(0.5)

def real_time_spectra_desired(BLOCK, OBSNCHAN, TBIN, samplesPerTransform, fftsPerIntegration, OBSFREQ, OBSBW, file_index):
    """
    Plot spectra and stats of real-time observational BL data for node of interest.

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
    waterfall_spectrum_x = np.zeros((10, OBSNCHAN * samplesPerTransform))
    waterfall_spectrum_y = np.zeros((10, OBSNCHAN * samplesPerTransform))
    waterfall_spectrum_x[file_index, :] = spectralData_x.reshape(-1)
    waterfall_spectrum_y[file_index, :] = spectralData_y.reshape(-1)

    # Spectrum for plotting
    bandPass_x = np.sum(waterfall_spectrum_x, 0)
    bandPass_y = np.sum(waterfall_spectrum_y, 0)

    # Helpful plotting values
    lowerBound = OBSFREQ + OBSBW/2
    upperBound = OBSFREQ - OBSBW/2
    current_RAW_axis = np.linspace(lowerBound, upperBound, OBSNCHAN *samplesPerTransform)

    plot_real_time_visualization_desired(waterfall_spectrum_x, waterfall_spectrum_y, bandPass_x, bandPass_y, SK_x, SK_y, current_RAW_axis, lowerBound, upperBound, samplesPerTransform, fftsPerIntegration, TBIN)

################################################################################
#######################---Program---############################################
################################################################################

if __name__ == "__main__":

    #User inputted resolutions
    desiredFrequencyResolution = 183105 #16 Bins
    desiredTimeResolution = 0.0003 #54 Integrations
    Plotted_Bank = 0
    Plotted_Node = 0
    #Hardware/band dependent parameters
    numberOfBanks = 1
    numberOfNodes = 8
    dualPolarization = 2
    desiredBank = 0
    desiredNode = 4


    for k in range(10):
	if (k > 0):
            clear_full_spectrum()
        for bank in range(numberOfBanks):
            for node in range(numberOfNodes):
                if (bank!=desiredBank or node!=desiredNode):
                    inputFileName = "/mnt_blc" + str(bank) + str(node) + "/datax/users/eenriquez/AGBT17A_999_56/GUPPI/BLP" + str(bank) + str(node) + "/blc" + str(bank) + str(node) + "_guppi_57872_11280_DIAG_PSR_J1136+1551_0001.000" + str(k) + ".raw"
                    readIn = np.memmap(inputFileName, dtype = 'int8', mode = 'r')
                    fileBytes = os.path.getsize(inputFileName)
                    #Initial location
                    currentBytesPassed = 0

                    #Header Information
                    OBSNCHAN, NPOL, NBITS, BLOCSIZE, OBSFREQ, CHAN_BW, OBSBW, TBIN, headerOffset = extractHeader(readIn, currentBytesPassed)

                    NDIM = int(BLOCSIZE/(OBSNCHAN*NPOL*(NBITS/8))) #time samples per channel per block
                    #Skip header and put data in easily parsed array

                    samplesPerTransform, fftsPerIntegration = convert_resolution(desiredFrequencyResolution, desiredTimeResolution, TBIN)
                    dataBuffer = readIn[(currentBytesPassed + headerOffset):(currentBytesPassed + headerOffset + BLOCSIZE)].reshape(OBSNCHAN, NDIM, NPOL)
                    NDIMsmall = samplesPerTransform * fftsPerIntegration
                    real_time_spectra_general(dataBuffer[:,0:NDIMsmall, :], OBSNCHAN, samplesPerTransform, fftsPerIntegration, OBSFREQ, OBSBW)

                    del readIn


        inputFileName = "/mnt_blc" + str(desiredBank) + str(desiredNode) + "/datax/users/eenriquez/AGBT17A_999_56/GUPPI/BLP" + str(desiredBank) + str(desiredNode) + "/blc" + str(desiredBank) + str(desiredNode) + "_guppi_57872_11280_DIAG_PSR_J1136+1551_0001.000" + str(k) + ".raw"
        readIn = np.memmap(inputFileName, dtype = 'int8', mode = 'r')
        fileBytes = os.path.getsize(inputFileName)
        #Initial location
        currentBytesPassed = 0

        #Header Information
        OBSNCHAN, NPOL, NBITS, BLOCSIZE, OBSFREQ, CHAN_BW, OBSBW, TBIN, headerOffset = header.extractHeader(readIn, currentBytesPassed)

        NDIM = int(BLOCSIZE/(OBSNCHAN*NPOL*(NBITS/8))) #time samples per channel per block
        #Skip header and put data in easily parsed array

        samplesPerTransform, fftsPerIntegration = convert_resolution(desiredFrequencyResolution, desiredTimeResolution, TBIN)
        dataBuffer = readIn[(currentBytesPassed + headerOffset):(currentBytesPassed + headerOffset + BLOCSIZE)].reshape(OBSNCHAN, NDIM, NPOL)

        NDIMsmall = samplesPerTransform * fftsPerIntegration
        real_time_spectra_desired(dataBuffer[:,0:NDIMsmall, :], OBSNCHAN, TBIN, samplesPerTransform, fftsPerIntegration, OBSFREQ, OBSBW, k)

        del readIn
