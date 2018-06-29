# Nicholas Joslyn
# Breakthrough Listen UC Berkeley SETI Intern 2018

# Main program for quasi-real time spectrum estimates of BL RAW data
from __future__ import division
import numpy as np
import os
import time

import spectrumEstimation
import header
import spectrumIntegration
import RealTime
import SKThresholds

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.collections import LineCollection

if __name__ == "__main__":

    startTime = time.time()

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

    #Memmap the RAW file and find the number of bytes
    # Personal desktop RAW file
    #inputFileName = "../../Downloads/blc05_guppi_58100_78802_OUMUAMUA_0011.0000.raw"
    # Green Bank nodes
    for k in range(10):
	if (k > 0):
            RealTime.clear_full_spectrum()
        for bank in range(numberOfBanks):
            for node in range(numberOfNodes):
                if (bank!=desiredBank or node!=desiredNode):
                    inputFileName = "/mnt_blc" + str(bank) + str(node) + "/datax/users/eenriquez/AGBT17A_999_56/GUPPI/BLP" + str(bank) + str(node) + "/blc" + str(bank) + str(node) + "_guppi_57872_11280_DIAG_PSR_J1136+1551_0001.000" + str(k) + ".raw"
                    readIn = np.memmap(inputFileName, dtype = 'int8', mode = 'r')
                    fileBytes = os.path.getsize(inputFileName)
                    #Initial location
                    currentBytesPassed = 0

                    #Header Information
                    OBSNCHAN, NPOL, NBITS, BLOCSIZE, OBSFREQ, CHAN_BW, OBSBW, TBIN, headerOffset = header.extractHeader(readIn, currentBytesPassed)

                    NDIM = int(BLOCSIZE/(OBSNCHAN*NPOL*(NBITS/8))) #time samples per channel per block
                    #Skip header and put data in easily parsed array

                    samplesPerTransform, fftsPerIntegration = RealTime.convert_resolution(desiredFrequencyResolution, desiredTimeResolution, TBIN)
                    dataBuffer = readIn[(currentBytesPassed + headerOffset):(currentBytesPassed + headerOffset + BLOCSIZE)].reshape(OBSNCHAN, NDIM, NPOL)
                    NDIMsmall = samplesPerTransform * fftsPerIntegration
                    RealTime.real_time_spectra_general(dataBuffer[:,0:NDIMsmall, :], OBSNCHAN, samplesPerTransform, fftsPerIntegration, OBSFREQ, OBSBW)

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

        samplesPerTransform, fftsPerIntegration = RealTime.convert_resolution(desiredFrequencyResolution, desiredTimeResolution, TBIN)
        dataBuffer = readIn[(currentBytesPassed + headerOffset):(currentBytesPassed + headerOffset + BLOCSIZE)].reshape(OBSNCHAN, NDIM, NPOL)

        NDIMsmall = samplesPerTransform * fftsPerIntegration
        RealTime.real_time_spectra_desired(dataBuffer[:,0:NDIMsmall, :], OBSNCHAN, TBIN, samplesPerTransform, fftsPerIntegration, OBSFREQ, OBSBW, k)

        del readIn

#    raw_input("Press enter when done")
    endTime = time.time()
    print("Time:" + str(round(endTime - startTime, 8)))
