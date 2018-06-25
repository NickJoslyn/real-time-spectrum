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

if __name__ == "__main__":

    startTime = time.time()
    desiredFrequencyResolution = 183105 #16 Bins
    desiredTimeResolution = 0.0003 #54 Integrations

    #Memmap the RAW file and find the number of bytes
    # Personal desktop RAW file
    #inputFileName = "../../Downloads/blc05_guppi_58100_78802_OUMUAMUA_0011.0000.raw"
    # Green Bank nodes
    inputFileName = "/mnt_blc00/datax/users/eenriquez/AGBT17A_999_56/GUPPI/BLP00/blc00_guppi_57872_11280_DIAG_PSR_J1136+1551_0001.0002.raw"
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
    RealTime.real_time_spectra(dataBuffer[:,0:NDIMsmall, :], OBSNCHAN, CHAN_BW, TBIN, samplesPerTransform, fftsPerIntegration, OBSFREQ, OBSBW)


    del readIn
    endTime = time.time()
    print("Time:" + str(round(endTime - startTime, 8)))
