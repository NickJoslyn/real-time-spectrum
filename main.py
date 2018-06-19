# Nicholas Joslyn
# Breakthrough Listen UC Berkeley SETI Intern 2018

# Main program for quasi-real time spectrum estimates

import numpy as np
import os
import spectrumEstimators
from header import extractHeader

if __name__ == "__main__":
    #Memmap the RAW file and find the number of bytes
    inputFileName = "../../Downloads/blc05_guppi_58100_78802_OUMUAMUA_0011.0000.raw"
    readIn = np.memmap(inputFileName, dtype = 'int8', mode = 'r')
    fileBytes = os.path.getsize(inputFileName)

    #Initial location
    currentBytesPassed = 0
    blockNumber = 0

    while (currentBytesPassed < fileBytes):

        if (blockNumber == 0):
            #Header Information
            OBSNCHAN, NPOL, NBITS, BLOCSIZE, OBSFREQ, CHAN_BW, OBSBW, TBIN, headerOffset = extractHeader(readIn, currentBytesPassed)

            #Number of time samples per channel per block
            NDIM = int(BLOCSIZE/(OBSNCHAN*NPOL*(NBITS/8)))

        #Skip header and put data in easily parsed array
        currentBytesPassed += headerOffset
        dataBuffer = readIn[currentBytesPassed:currentBytesPassed + BLOCSIZE].reshape(OBSNCHAN, NDIM, NPOL)

        for CHANNEL in range(OBSNCHAN):
            centerFrequency = OBSFREQ + (np.abs(OBSBW)/2) - (CHANNEL + 0.5)*np.abs(CHAN_BW)
            spectrumEstimators.RAW_periodogram(dataBuffer[CHANNEL,:,:], TBIN, CHANNEL, 'density')
            spectrumEstimators.RAW_FFT(dataBuffer[CHANNEL,:,:], CHANNEL, centerFrequency, CHAN_BW)

        del readIn
        exit()
