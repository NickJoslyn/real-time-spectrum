# Nick Joslyn
# Breakthrough Listen UC Berkeley SETI Intern 2018

# Create a waterfall plot with custom time and frequency resolution
# Development: Prototype script, then function call
import numpy as np
import matplotlib.pyplot as plt
import os

###Custom parameters
customFrequencyResolution = 5000 #in Hz
customTimeResolution = 0.1




# Will be passed as parameters to a function
cardLength = 80 #BL Raw Information

inputFileName = "../../Downloads/blc05_guppi_58100_78802_OUMUAMUA_0011.0000.raw"
readIn = np.memmap(inputFileName, dtype = 'int8', mode = 'r')
# CHANNEL = 0
# TBIN = 0.000000341333
# NPOL = 4
# CHAN_BW = -2.9296875
# OBSBW = -187.5
# OBSFREQ = 2120.21484375

#Calculate In function


fileBytes = os.path.getsize(inputFileName)

#Initialize Loop Break to 0 "bytes"
currentBytesPassed = 0

#Track Current Block Number during Loop
blockNumber = 0

#Loop will end when we hit the end of file
while(currentBytesPassed < fileBytes):


    ##-----Header Information------------------------------------------------
	continueLoop = True
	lineCounter = 0     #Ensure the same card is not read twice
	if blockNumber > -1:
		while(continueLoop):

			cardString = ''

			#Get the ASCII value of the card and convert to char
			for index in range(cardLength):
				cardString += chr(readIn[currentBytesPassed + index + lineCounter * cardLength])

			print(cardString)

			#Identify the end of the header
			#If not the end, find other useful parameters from header
			if (cardString[:3] == 'END'):   #reached end of header
				continueLoop = False

			elif(cardString[:8] == 'OBSNCHAN'): #Number of Channels
				OBSNCHAN = int(cardString[9:].strip()) #remove white spaces and convert to int

			elif(cardString[:4] == 'NPOL'):     #Number of Polarizations * 2
				NPOL = int(cardString[9:].strip())

			elif(cardString[:5] == 'NBITS'):    #Number of Bits per Data Sample
				NBITS = int(cardString[9:].strip())

			elif(cardString[:7] == 'OVERLAP'):  #Number of Time Samples that Overlap Between Blocks
				OVERLAP = int(cardString[9:].strip())

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

		#Calculate once -- NDIM is number of samples per channel
		headerOffset = cardLength * lineCounter + DIRECTIO_offset
		NDIM = int(BLOCSIZE/(OBSNCHAN*NPOL*(NBITS/8)))
		print(NDIM)
##-----Done With Header Information Extraction------------------------------

	#Skip header unaltered
	currentBytesPassed += headerOffset
	dataBuffer = readIn[currentBytesPassed:currentBytesPassed + BLOCSIZE].reshape(OBSNCHAN, NDIM, NPOL)
	for CHANNEL in range(OBSNCHAN):
		#Make a writeable copy of the buffer for the time domain MAD
		if (CHANNEL == 0):
			copyBuffer = np.copy(dataBuffer[CHANNEL, :, :])
			xrTime = copyBuffer[:, 0]
			xiTime = copyBuffer[:, 1]
			yrTime = copyBuffer[:, 2]
			yiTime = copyBuffer[:, 3]

			centerFrequency = OBSFREQ + (np.abs(OBSBW)/2) - (CHANNEL + 0.5)*np.abs(CHAN_BW)


			numberOfSamples = int((1/customFrequencyResolution)/TBIN)
			print(numberOfSamples)
			numberOfFFTs = int(customTimeResolution * customFrequencyResolution)
			print(numberOfFFTs)

			lengthOfPlot = 1
			waterfallData = np.zeros((lengthOfPlot, numberOfSamples))

			for integration in range(lengthOfPlot):
				summedFFT = np.zeros(numberOfSamples)
				for window in range(numberOfFFTs):
					FFTxPol = np.fft.fftshift(np.fft.fft(xrTime[window*numberOfSamples:(window+1)*numberOfSamples] + 1j*xiTime[window*numberOfSamples:(window+1)*numberOfSamples]))
					summedFFT += np.absolute(FFTxPol)**2
				waterfallData[integration, :] = summedFFT

			plt.figure()
			plt.imshow(waterfallData, cmap = 'gray', aspect = 'auto', extent = [centerFrequency + CHAN_BW/2, centerFrequency - CHAN_BW/2, 0, customTimeResolution * lengthOfPlot])
			plt.title("Waterfall")
			plt.xlabel("Frequency")
			plt.ylabel("Time")
			plt.tight_layout()
			plt.show()
