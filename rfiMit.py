# In development:
# Identifying and altering applicable code from rfiMitigation.py

## Parts to keep:
## 	Read Raw header --> Possibly use blimpy?
## 	Channelizing the stream of voltages
## 	Running FFTs and plotting spectra

import numpy as np
import matplotlib.pylab as plt
import os
import time

startTime = time.time()

#-------------------------
#Program parameters

inputFileName = "../../Downloads/blc05_guppi_58100_78802_OUMUAMUA_0011.0000.raw"
cardLength = 80 #BL Raw Information


#FFT Information
numberOfTransforms = 800
samplesPerTransform = 4096
#edgeOffset = 500

#------------------------
#Functions

#-------------------------
#Begin Program
#-------------------------

#File Information
readIn = np.memmap(inputFileName, dtype = 'int8', mode = 'r')
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

			lineCounter += 1    #Go to Next Card in Header

		#Padding Bytes
		if (DIRECTIO != 0):
			DIRECTIO_offset = 512 - (cardLength*lineCounter)%512
		else:
			DIRECTIO_offset = 0

		#Calculate once -- NDIM is number of samples per channel
		headerOffset = cardLength * lineCounter + DIRECTIO_offset
		NDIM = int(BLOCSIZE/(OBSNCHAN*NPOL*(NBITS/8)))

	##-----Done With Header Information Extraction------------------------------

	#Skip header unaltered
	currentBytesPassed += headerOffset

	#Put data into an easily parsed array
	dataBuffer = readIn[currentBytesPassed:currentBytesPassed + BLOCSIZE].reshape(OBSNCHAN, NDIM, NPOL)

	for CHANNEL in range(OBSNCHAN):

		#Make a writeable copy of the buffer for the time domain MAD
		copyBuffer = np.copy(dataBuffer[CHANNEL, :, :])
		xrTime = copyBuffer[:, 0]
		xiTime = copyBuffer[:, 1]
		yrTime = copyBuffer[:, 2]
		yiTime = copyBuffer[:, 3]

		tempFFTx = np.fft.fftshift(np.fft.fft(xrTime + 1j*xiTime))
		tempFFTy = np.fft.fftshift(np.fft.fft(yrTime + 1j*yiTime))

		#Plot here#####################################
		centerFrequency = OBSFREQ + (np.abs(OBSBW)/2) - (CHANNEL + 0.5)*np.abs(CHAN_BW)
		print(CHAN_BW)
		print(np.abs(CHAN_BW))
		print(centerFrequency)
		print(OBSFREQ)
		plt.plot(np.linspace(centerFrequency + CHAN_BW/2, centerFrequency - CHAN_BW/2, len(tempFFTx)), np.abs(tempFFTx)**2)
		plt.title("X")
		plt.show()
		plt.plot(np.linspace(centerFrequency + CHAN_BW/2, centerFrequency - CHAN_BW/2, len(tempFFTx)), np.abs(tempFFTy)**2)
		plt.title('Y')
		plt.show()


		###############################################33

		print("STOP")

		for i in range( (NDIM-OVERLAP)//(samplesPerTransform*numberOfTransforms) ):

			FFTx = np.zeros(samplesPerTransform)
			FFTy = np.zeros(samplesPerTransform)
			FFTSTOREx = np.zeros((numberOfTransforms, samplesPerTransform), dtype = 'complex')
			FFTSTOREy = np.zeros((numberOfTransforms, samplesPerTransform), dtype = 'complex')

			for k in range(numberOfTransforms):

				#Keeps track of byte location while looping through FFTs(inner loop)
				#and groups of FFTs (outer loop)
				indexOffset = OVERLAP + i*samplesPerTransform*numberOfTransforms + k*samplesPerTransform

				#Time Domain
				xrTimeMAD = replaceOutliers(copyBuffer[indexOffset:indexOffset + samplesPerTransform, 0])
				xiTimeMAD = replaceOutliers(copyBuffer[indexOffset:indexOffset + samplesPerTransform, 1])
				yrTimeMAD = replaceOutliers(copyBuffer[indexOffset:indexOffset + samplesPerTransform, 2])
				yiTimeMAD = replaceOutliers(copyBuffer[indexOffset:indexOffset + samplesPerTransform, 3])


				tempFFTx = np.fft.fftshift(np.fft.fft(xrTimeMAD + 1j*xiTimeMAD))
				FFTSTOREx[k,:] = tempFFTx

				tempFFTy = np.fft.fftshift(np.fft.fft(yrTimeMAD + 1j*yiTimeMAD))
				FFTSTOREy[k,:] = tempFFTy

				FFTx += np.abs(tempFFTx)**2
				FFTy += np.abs(tempFFTy)**2

			#Find Outlying Bins
			replacerx = findOutliers(FFTx[edgeOffset:-edgeOffset])
			replacery = findOutliers(FFTy[edgeOffset:-edgeOffset])

			#Parameters for Gaussian Random Number Generator
			realMedianx = np.median(FFTSTOREx[:, edgeOffset:-edgeOffset].real)
			realSTDx = findRSTD(FFTSTOREx[:, edgeOffset:-edgeOffset].real)
			imagMedianx = np.median(FFTSTOREx[:, edgeOffset:-edgeOffset].imag)
			imagSTDx = findRSTD(FFTSTOREx[:, edgeOffset:-edgeOffset].imag)
			realMediany = np.median(FFTSTOREy[:, edgeOffset:-edgeOffset].real)
			realSTDy = findRSTD(FFTSTOREy[:, edgeOffset:-edgeOffset].real)
			imagMediany = np.median(FFTSTOREy[:, edgeOffset:-edgeOffset].imag)
			imagSTDy = findRSTD(FFTSTOREy[:, edgeOffset:-edgeOffset].imag)


			#Mitigation
			for xz in replacerx:
				FFTSTOREx[:, xz + edgeOffset].real = np.random.normal(realMedianx, realSTDx, numberOfTransforms)
				FFTSTOREx[:, xz + edgeOffset].imag = np.random.normal(imagMedianx, imagSTDx, numberOfTransforms)

			for yz in replacery:
				FFTSTOREy[:, yz + edgeOffset].real = np.random.normal(realMediany, realSTDy, numberOfTransforms)
				FFTSTOREy[:, yz + edgeOffset].imag = np.random.normal(imagMediany, imagSTDy, numberOfTransforms)

			#Write Out in the correct order
			writeOutBuffer = np.zeros(numberOfTransforms*samplesPerTransform*NPOL)
			for k in range(numberOfTransforms):

				tempX = np.fft.ifft(np.fft.ifftshift(FFTSTOREx[k,:]))
				tempY = np.fft.ifft(np.fft.ifftshift(FFTSTOREy[k,:]))

				writeOutOffset = k*samplesPerTransform*NPOL
				writeOutBuffer[writeOutOffset:writeOutOffset + samplesPerTransform*NPOL] = np.column_stack((tempX.real.astype('int8'), tempX.imag.astype('int8'), tempY.real.astype('int8'), tempY.imag.astype('int8'))).reshape(-1)


			writeOutFile[currentBytesPassed:currentBytesPassed + numberOfTransforms*samplesPerTransform*NPOL] = writeOutBuffer
			currentBytesPassed += numberOfTransforms*samplesPerTransform*NPOL


		#Bytes left over before end of block

		leftOverSamples = NDIM - OVERLAP - ( (NDIM-OVERLAP)//(numberOfTransforms*samplesPerTransform) ) * numberOfTransforms*samplesPerTransform

		dataLeftOver = copyBuffer[-leftOverSamples:, :].reshape(leftOverSamples,NPOL)

		#Number of samplesPerTransform-point FFTs left
		windowsLeft = leftOverSamples//samplesPerTransform
		FFTx = np.zeros(samplesPerTransform)
		FFTy = np.zeros(samplesPerTransform)
		FFTSTOREx = np.zeros((windowsLeft, samplesPerTransform), dtype = 'complex')
		FFTSTOREy = np.zeros((windowsLeft, samplesPerTransform), dtype = 'complex')

		for i in range(windowsLeft):

			#Time Domain
			xrTimeMAD = replaceOutliers(dataLeftOver[i*samplesPerTransform:i*samplesPerTransform + samplesPerTransform, 0])
			xiTimeMAD = replaceOutliers(dataLeftOver[i*samplesPerTransform:i*samplesPerTransform + samplesPerTransform, 1])
			yrTimeMAD = replaceOutliers(dataLeftOver[i*samplesPerTransform:i*samplesPerTransform + samplesPerTransform, 2])
			yiTimeMAD = replaceOutliers(dataLeftOver[i*samplesPerTransform:i*samplesPerTransform + samplesPerTransform, 3])

			tempFFTx = np.fft.fftshift(np.fft.fft(xrTimeMAD + 1j*xiTimeMAD))
			FFTSTOREx[i,:] = tempFFTx

			tempFFTy = np.fft.fftshift(np.fft.fft(yrTimeMAD + 1j*yiTimeMAD))
			FFTSTOREy[i,:] = tempFFTy

			FFTx += np.abs(tempFFTx)**2
			FFTy += np.abs(tempFFTy)**2

		replacerx = findOutliers(FFTx[edgeOffset:-edgeOffset])
		replacery = findOutliers(FFTy[edgeOffset:-edgeOffset])

		realMedianx = np.median(FFTSTOREx[:, edgeOffset:-edgeOffset].real)
		realSTDx = findRSTD(FFTSTOREx[:, edgeOffset:-edgeOffset].real)
		imagMedianx = np.median(FFTSTOREx[:, edgeOffset:-edgeOffset].imag)
		imagSTDx = findRSTD(FFTSTOREx[:, edgeOffset:-edgeOffset].imag)

		realMediany = np.median(FFTSTOREy[:, edgeOffset:-edgeOffset].real)
		realSTDy = findRSTD(FFTSTOREy[:, edgeOffset:-edgeOffset].real)
		imagMediany = np.median(FFTSTOREy[:, edgeOffset:-edgeOffset].imag)
		imagSTDy = findRSTD(FFTSTOREy[:, edgeOffset:-edgeOffset].imag)


		#Mitigation
		for xz in replacerx:
			FFTSTOREx[:, xz + edgeOffset].real = np.random.normal(realMedianx, realSTDx, windowsLeft)
			FFTSTOREx[:, xz + edgeOffset].imag = np.random.normal(imagMedianx, imagSTDx, windowsLeft)

		for yz in replacery:
			FFTSTOREy[:, yz + edgeOffset].real = np.random.normal(realMediany, realSTDy, windowsLeft)
			FFTSTOREy[:, yz + edgeOffset].imag = np.random.normal(imagMediany, imagSTDy, windowsLeft)


		writeOutBuffer = np.zeros(windowsLeft*samplesPerTransform*NPOL)
		for k in range(windowsLeft):

			tempX = np.fft.ifft(np.fft.ifftshift(FFTSTOREx[k,:]))
			tempY = np.fft.ifft(np.fft.ifftshift(FFTSTOREy[k,:]))

			writeOutOffset = k*samplesPerTransform*NPOL
			writeOutBuffer[writeOutOffset:writeOutOffset + samplesPerTransform*NPOL] = np.column_stack((tempX.real.astype('int8'), tempX.imag.astype('int8'), tempY.real.astype('int8'), tempY.imag.astype('int8'))).reshape(-1)


		writeOutFile[currentBytesPassed:currentBytesPassed + windowsLeft*samplesPerTransform*NPOL] = writeOutBuffer
		currentBytesPassed += windowsLeft*samplesPerTransform*NPOL

		#Write out the remaining < 4096 time samples
		bytesRemaining = NPOL * (leftOverSamples - windowsLeft * samplesPerTransform)
		if(bytesRemaining > 0):
			writeOutFile[currentBytesPassed:currentBytesPassed + bytesRemaining] = replaceOutliers(dataLeftOver[-(bytesRemaining/NPOL):, :].reshape(-1))
			currentBytesPassed += bytesRemaining

	writeOutFile.flush()

	if (blockNumber%10 == 0):
		print("Percent Complete (approximate): " + str(round((currentBytesPassed/fileBytes) * 100, 2)) + "%:")

	blockNumber+=1

del readIn
del writeOutFile

endTime = time.time()

print("Time: " + str(round(endTime - startTime)))

#----------------------------------------------------------
#----------------------------------------------------------
#----------------------------------------------------------


#MAD Information
normalDistributionScale = 1.4826
outlierDistance = 3

#Function to identify data points outside of 3 standard deviations
def findOutliers(array):

	#Median of input array
	median = np.median(array)

	# Second array to hold absolute deviations from median
	array2 = abs(array-median)
	mad = np.median(array2)

	# Find Robust Standard Deviation
	rstd = mad*normalDistributionScale

	#Random Number Generator fails with 0
	if rstd == 0:
		rstd = 0.01

	# Identify outlier criteria:
	top = median + outlierDistance*rstd
	bottom = median - outlierDistance*rstd

	replacementList, = np.where(array>top)
	replacementList2, = np.where(array<bottom)

	arrayToReplace = np.concatenate([replacementList, replacementList2])
	return arrayToReplace


#Function to replace data points outside of 3 standard deviations
def replaceOutliers(array):

	median = np.median(array)

	array2 = np.abs(array-median)

	mad = np.median(array2)

	# Find Robust Standard Deviation
	rstd = mad*normalDistributionScale

	# Random Number Generator fails with 0
	if rstd == 0:
		rstd = 0.01

	# Identify outlier criteria:
	top = median + outlierDistance*rstd
	bottom = median - outlierDistance*rstd

	replacementList, = np.where(array>top)
	for i in replacementList:
		array[i] = int(np.random.normal(median,rstd))

	replacementList2, = np.where(array<bottom)
	for i in replacementList2:
		array[i] = int(np.random.normal(median, rstd))

	return array

def findRSTD(array):

	arrayToReplace = []

	#Median of input array
	median = np.median(array)

	# Second array to hold absolute deviations from median
	array2 =abs(array-median)

	mad = np.median(array2)

	# Find Robust Standard Deviation
	rstd = mad*normalDistributionScale

	# Random Number Generator fails with 0
	if rstd == 0:
		rstd = 0.1

	return rstd
