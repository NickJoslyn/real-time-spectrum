# Nicholas Joslyn
# Breakthrough Listen UC Berkeley SETI Intern 2018

# Main program for quasi-real time spectrum estimates of BL RAW data

import numpy as np
import os
import spectrumEstimation
from header import extractHeader
import time
import spectrumIntegration

startTime = time.time()

if __name__ == "__main__":

    #Memmap the RAW file and find the number of bytes
    #inputFileName = "../../Downloads/blc05_guppi_58100_78802_OUMUAMUA_0011.0000.raw"
    inputFileName = "/mnt_blc00/datax/users/eenriquez/AGBT17A_999_56/GUPPI/BLP00/blc00_guppi_57872_11280_DIAG_PSR_J1136+1551_0001.0002.raw"
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
                if (CHANNEL == 3 or CHANNEL == 5):
                    centerFrequency = OBSFREQ + (np.abs(OBSBW)/2) - (CHANNEL + 0.5)*np.abs(CHAN_BW)
		    spectrumIntegration.RAW_waterfall(dataBuffer[CHANNEL, :, :], CHANNEL, centerFrequency, CHAN_BW, TBIN, 5722.6, 0.003)		    
           # Spectrum Integration Function Testing --------------------------------------
            #spectrumIntegration.RAW_PSD(dataBuffer[CHANNEL, :, :], CHANNEL, centerFrequency, CHAN_BW, TBIN)
            #spectrumIntegration.RAW_PSD(dataBuffer[CHANNEL, :, :], CHANNEL, centerFrequency, CHAN_BW, TBIN, 256)
            #spectrumIntegration.RAW_PSD(dataBuffer[CHANNEL, :, :], CHANNEL, centerFrequency, CHAN_BW, TBIN, 4096)
            #spectrumIntegration.RAW_spectrogram(dataBuffer[CHANNEL, :, :], CHANNEL, centerFrequency, CHAN_BW, TBIN)
            #spectrumIntegration.RAW_waterfall(dataBuffer[CHANNEL, :, :], CHANNEL, centerFrequency, CHAN_BW, TBIN, 5722.6, .003)
                    #spectrumIntegration.RAW_integratedFFT(dataBuffer[CHANNEL, :, :], CHANNEL, centerFrequency, CHAN_BW, TBIN, 5722.6)


            # Spectrum Estimation Function Tests-----------------------------------------
            #    spectrumEstimation.RAW_periodogram(dataBuffer[CHANNEL,:,:], CHANNEL, centerFrequency, CHAN_BW, TBIN)
            #    spectrumEstimation.RAW_periodogram(dataBuffer[CHANNEL,:,:], CHANNEL, centerFrequency, CHAN_BW, TBIN, 'spectrum')
                    #spectrumEstimation.RAW_FFT(dataBuffer[CHANNEL,:,:], CHANNEL, centerFrequency, CHAN_BW)
            # spectrumEstimation.RAW_FFT(dataBuffer[CHANNEL,:,:], CHANNEL, centerFrequency, CHAN_BW, 'power spectrum')
            #    spectrumEstimation.RAW_FFT(dataBuffer[CHANNEL,:,:], CHANNEL, centerFrequency, CHAN_BW, 'power normalized')
            # spectrumEstimation.RAW_FFT(dataBuffer[CHANNEL,:,:], CHANNEL, centerFrequency, CHAN_BW, 'magnitude normalized')
            #    spectrumEstimation.RAW_pyplot_magnitude(dataBuffer[CHANNEL,:,:], CHANNEL, centerFrequency, CHAN_BW, TBIN)
        del readIn
        endTime = time.time()
        print("Time:" + str(round(endTime - startTime, 8)))
        exit()
