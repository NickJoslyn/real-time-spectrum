# Nicholas Joslyn
# Breakthrough Listen UC Berkeley SETI Intern 2018

# Program for real-time spectra of BL Observations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def convert_resolution(customFrequencyResolution, customTimeResolution, TBIN):
    """
    Convert custom frequency (Hz) resolution and time (s) resolution into FFT parameters

    Return:
    samplesPerTransform (int):  Number of samples for each FFT
    fftsPerIntegration (int):   Number of FFTs to integrate over
    """

    samplesPerTransform = int((1/customFrequencyResolution)/TBIN)
    fftsPerIntegration = int(customTimeResolution * customFrequencyResolution)
    print(samplesPerTransform, fftsPerIntegration)
    return samplesPerTransform, fftsPerIntegration

def remove_DCoffset(BLOCK):

    channels, samples, polarizations = BLOCK.shape
    DC_offset = np.mean(BLOCK, axis = 1).reshape(channels, 1, polarizations)
    _, DC_offset = np.broadcast_arrays(BLOCK, DC_offset)
    return (BLOCK - DC_offset)


def real_time_spectra(BLOCK, OBSNCHAN, CHANNEL, CHAN_BW, TBIN, samplesPerTransform, fftsPerIntegration, OBSFREQ, OBSBW):
    """
    Plot spectra and stats of real-time observational BL data on block.

    """

    #Need to expand from channel analysis to full compute node analysis (i.e. band analysis)

    ###################Prepare Data

    BLOCK = remove_DCoffset(BLOCK)

    waterfallData_x = np.zeros((OBSNCHAN, samplesPerTransform))
    waterfallData_y = np.zeros((OBSNCHAN, samplesPerTransform))

    for channel in range(OBSNCHAN):
        waterfallData_x[channel, :] = np.sum(np.abs(np.fft.fftshift(np.fft.fft(np.split(BLOCK[channel,:, 0] + 1j*BLOCK[channel,:,1], fftsPerIntegration))))**2, axis = 0)
        waterfallData_y[channel, :] = np.sum(np.abs(np.fft.fftshift(np.fft.fft(np.split(BLOCK[channel,:,2] + 1j*BLOCK[channel,:, 3], fftsPerIntegration))))**2, axis = 0)

    # print(waterfallData_x.shape)
    # print(waterfallData_y.shape)

    lowerBound = OBSFREQ + OBSBW/2
    upperBound = OBSFREQ - OBSBW/2
    totalTime = TBIN * samplesPerTransform * fftsPerIntegration
    waterfallData_x = np.flip(waterfallData_x, 0)
    waterfallData_y = np.flip(waterfallData_y, 0)

    integrated_spectrum_x = [waterfallData_x.reshape(-1)]
    integrated_spectrum_y = [waterfallData_y.reshape(-1)]

    plt.figure()
    plt.imshow(integrated_spectrum_x, cmap = 'viridis', aspect = 'auto', norm = LogNorm(), extent = [lowerBound, upperBound, totalTime, 0])
    plt.title("Waterfall Plot: X")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Time (s)")
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.imshow(integrated_spectrum_y, cmap = 'viridis', aspect = 'auto', norm = LogNorm(), extent = [lowerBound, upperBound, totalTime, 0])
    plt.title("Waterfall Plot: Y")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Time (s)")
    plt.colorbar()
    plt.show()

    ###################Channel Spectrum

    bandPass_x = np.sum(integrated_spectrum_x, 0)
    bandPass_y = np.sum(integrated_spectrum_y, 0)

    plt.plot(np.linspace(lowerBound, upperBound, OBSNCHAN *samplesPerTransform), bandPass_x, color = 'black')
    plt.title("Full Spectrum: X")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Power, Log")
    plt.yscale('log')
    plt.show()

    plt.plot(np.linspace(lowerBound, upperBound, OBSNCHAN * samplesPerTransform), bandPass_y, color = 'black')
    plt.title("Full Spectrum: Y")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Power, Log")
    plt.yscale('log')
    plt.show()

    #SET UP Big Plot
    plt.figure("Template")

    # Full Bandpass
    ax1 = plt.subplot2grid((18,5), (0,0), colspan=5, rowspan=3)
    ax1.set_title("Full Observation Spectrum")

    # Spectra
    ax2 = plt.subplot2grid((18,5), (5,0), colspan=2, rowspan=3)
    ax2.set_title("Node Spectrum: X")
    ax2.set_xlabel("Frequency (MHz)")
    ax2.set_ylabel("Power")
    ax2.set_yscale('log')
    ax2.margins(x=0)
    ax2.plot(np.linspace(lowerBound, upperBound, OBSNCHAN * samplesPerTransform), bandPass_x, color = 'black')
    ax3 = plt.subplot2grid((18,5), (5, 3), colspan=2, rowspan=3)
    ax3.set_title("Node Spectrum: Y")
    ax3.set_xlabel("Frequency (MHz)")
    ax3.set_ylabel("Power")
    ax3.set_yscale('log')
    ax3.margins(x=0)
    ax3.plot(np.linspace(lowerBound, upperBound, OBSNCHAN * samplesPerTransform), bandPass_y)

    # Waterfall
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


    # Spectral Kurtosis
    ax6 = plt.subplot2grid((18,5), (15,0), colspan=2, rowspan=3)
    ax6.set_title("Spectral Kurtosis: X")
    ax7 = plt.subplot2grid((18,5), (15, 3), colspan=2, rowspan=3)
    ax7.set_title("Spectral Kurtosis: Y")
    plt.suptitle("Real-Time Spectra of Observation")

    plt.show()


    quit()


def real_time_spectra_multiple_Integrations(BLOCK, OBSNCHAN, CHANNEL, CHAN_BW, TBIN, samplesPerTransform, fftsPerIntegration, numberOfIntegrations, OBSFREQ, OBSBW):
    """
    Plot spectra and stats of real-time observational BL data on block.

    """

    #Need to expand from channel analysis to full compute node analysis (i.e. band analysis)

    ###################Prepare Data

    BLOCK = remove_DCoffset(BLOCK)

    waterfallData_x = np.zeros((OBSNCHAN, numberOfIntegrations, samplesPerTransform))
    waterfallData_y = np.zeros((OBSNCHAN, numberOfIntegrations, samplesPerTransform))

    for channel in range(OBSNCHAN):
        for integration in range(numberOfIntegrations):

            initialIndex = integration * samplesPerTransform * fftsPerIntegration
            finalIndex = (integration + 1) * samplesPerTransform * fftsPerIntegration

            waterfallData_x[channel, integration, :] = np.sum(np.abs(np.fft.fftshift(np.fft.fft(np.split(BLOCK[channel,initialIndex:finalIndex, 0] + 1j*BLOCK[channel,initialIndex:finalIndex,1], fftsPerIntegration))))**2, axis = 0)
            waterfallData_y[channel, integration, :] = np.sum(np.abs(np.fft.fftshift(np.fft.fft(np.split(BLOCK[channel,initialIndex:finalIndex,2] + 1j*BLOCK[channel,initialIndex:finalIndex, 3], fftsPerIntegration))))**2, axis = 0)

    # print(waterfallData_x.shape)
    # print(waterfallData_y.shape)

    lowerBound = OBSFREQ + OBSBW/2
    upperBound = OBSFREQ - OBSBW/2

    integrated_spectra_x = np.zeros((numberOfIntegrations, OBSNCHAN * samplesPerTransform))
    integrated_spectra_y = np.zeros((numberOfIntegrations, OBSNCHAN * samplesPerTransform))

    for i in range(numberOfIntegrations):
        integrated_spectra_x[i,:] = waterfallData_x[:,i,:].reshape(-1)
        integrated_spectra_y[i,:] = waterfallData_y[:,i,:].reshape(-1)

    plt.figure()
    plt.imshow(integrated_spectra_x, cmap = 'viridis', aspect = 'auto', norm = LogNorm())
    plt.title("Waterfall Plot: X")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Time")
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.imshow(integrated_spectra_y, cmap = 'viridis', aspect = 'auto', norm = LogNorm())
    plt.title("Waterfall Plot: Y")
    plt.xlabel("Frequency Bin Number")
    plt.ylabel("Time (currently incorrect scale)")
    plt.colorbar()
    plt.show()

    ###################Channel Spectrum

    integrated_spectrum_x = np.sum(integrated_spectra_x, axis = 0)
    integrated_spectrum_y = np.sum(integrated_spectra_y, axis=0)
    plt.plot(integrated_spectrum_x, color = 'black')
    #plt.plot(np.linspace(lowerBound, upperBound, OBSNCHAN *samplesPerTransform), integrated_spectrum_x, color = 'black')
    plt.title("Full Spectrum: X")
    plt.xlabel("Frequency Bin Number")
    plt.ylabel("Power, Log")
    plt.yscale('log')
    plt.show()

    plt.plot(np.linspace(lowerBound, upperBound, OBSNCHAN * samplesPerTransform), integrated_spectrum_y, color = 'black')
    plt.title("Full Spectrum: Y")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Power, Log")
    plt.yscale('log')
    plt.show()



    quit()




    # Divide channel into polarizations
    xrTime = RAW_CHANNEL[:,0]
    xiTime = RAW_CHANNEL[:,1]
    yrTime = RAW_CHANNEL[:,2]
    yiTime = RAW_CHANNEL[:,3]

    # Remove DC
    xrTime_noDC = xrTime - np.mean(xrTime)
    xiTime_noDC = xiTime - np.mean(xiTime)
    yrTime_noDC = yrTime - np.mean(yrTime)
    yiTime_noDC = yiTime - np.mean(yiTime)

    # Variables for plotting
    samplingFrequency = 1/TBIN
    lowerBound = centerFrequency + CHAN_BW/2
    upperBound = centerFrequency - CHAN_BW/2

    ###################Waterfall

    waterfallData_x = np.zeros((numberOfIntegrations,samplesPerTransform))
    waterfallData_y = np.zeros((numberOfIntegrations,samplesPerTransform))

    for integration in range(numberOfIntegrations):
        index = integration * fftsPerIntegration * samplesPerTransform
        for individualFFT in range(fftsPerIntegration):
            waterfallData_x[integration,:] += np.abs(np.fft.fftshift(np.fft.fft(xrTime_noDC[index + individualFFT*samplesPerTransform:index + (individualFFT+1)*samplesPerTransform] + 1j*xiTime_noDC[index + individualFFT*samplesPerTransform:index + (individualFFT+1)*samplesPerTransform])))**2
            waterfallData_y[integration,:] += np.abs(np.fft.fftshift(np.fft.fft(yrTime_noDC[index + individualFFT*samplesPerTransform:index + (individualFFT+1)*samplesPerTransform] + 1j*yiTime_noDC[index + individualFFT*samplesPerTransform:index + (individualFFT+1)*samplesPerTransform])))**2

    plt.figure()
    plt.imshow(waterfallData_x, cmap = 'viridis', aspect = 'auto', extent = [lowerBound, upperBound, 0, 0.18])
    plt.title("Waterfall Plot: Channel " + str(CHANNEL) + ": X Polarization")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Time")
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.imshow(waterfallData_y, cmap = 'viridis', aspect = 'auto', extent = [lowerBound, upperBound, 0, 0.18])
    plt.title("Waterfall Plot: Channel " + str(CHANNEL) + ": Y Polarization")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Time")
    plt.colorbar()
    plt.show()

    ###################Chanel Spectrum

    integrated_spectrum_x = np.sum(waterfallData_x, axis = 0)
    integrated_spectrum_y = np.sum(waterfallData_y, axis=0)

    plt.plot(np.linspace(lowerBound, upperBound, samplesPerTransform), integrated_spectrum_x)
    plt.title("Channel " + str(CHANNEL) + ": X Polarization")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Spectrum_X")
    plt.show()

    plt.plot(np.linspace(lowerBound, upperBound, samplesPerTransform), integrated_spectrum_y)
    plt.title("Channel " + str(CHANNEL) + ": Y Polarization")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Spectrum_Y")
    plt.show()



    #SET UP Big Plot
    plt.figure(0)

    # Full Bandpass
    ax1 = plt.subplot2grid((18,5), (0,0), colspan=5, rowspan=3)
    ax1.set_title("Full Bandpass")

    # Channel Spectra
    ax2 = plt.subplot2grid((18,5), (5,0), colspan=2, rowspan=3)
    ax2.set_title("Channel Spectrum: X")
    ax2.set_xlabel("Frequency (MHz)")
    ax2.plot(np.linspace(lowerBound, upperBound, samplesPerTransform), integrated_spectrum_x)
    ax3 = plt.subplot2grid((18,5), (5, 3), colspan=2, rowspan=3)
    ax3.set_title("Channel Spectrum: Y")
    ax3.set_xlabel("Frequency (MHz)")
    ax3.plot(np.linspace(lowerBound, upperBound, samplesPerTransform), integrated_spectrum_y)

    # Channel Waterfall
    ax4 = plt.subplot2grid((18,5), (10, 0), colspan=2, rowspan=3)
    ax4.set_title("Waterfall: X")
    ax4.imshow(waterfallData_x, cmap = 'viridis', aspect = 'auto', extent = [lowerBound, upperBound, 0, 0.003])
    ax4.set_xlabel("Frequency (MHz)")
    ax4.set_ylabel("Time")


    ax5 = plt.subplot2grid((18,5), (10, 3), colspan=2, rowspan=3)
    ax5.set_title("Waterfall: Y")
    ax5.imshow(waterfallData_y, cmap = 'viridis', aspect = 'auto', extent = [lowerBound, upperBound, 0, 0.003])
    ax5.set_xlabel("Frequency (MHz)")
    ax5.set_ylabel("Time")


    # Spectral Kurtosis
    ax6 = plt.subplot2grid((18,5), (15,0), colspan=2, rowspan=3)
    ax6.set_title("Spectral Kurtosis: X")
    ax7 = plt.subplot2grid((18,5), (15, 3), colspan=2, rowspan=3)
    ax7.set_title("Spectral Kurtosis: Y")
    plt.suptitle("Real-Time Spectra of Observation")

    plt.show()
    exit()





def real_time_spectra_Channel(RAW_CHANNEL, CHANNEL, centerFrequency, CHAN_BW, TBIN, frequencyResolution, integrationTime, numberOfIntegrations):
    """
    Plot spectra and stats of real-time observational BL data.

    """

    #Need to expand from channel analysis to full compute node analysis (i.e. band analysis)

    ###################Prepare Data

    # Divide channel into polarizations
    xrTime = RAW_CHANNEL[:,0]
    xiTime = RAW_CHANNEL[:,1]
    yrTime = RAW_CHANNEL[:,2]
    yiTime = RAW_CHANNEL[:,3]

    # Remove DC
    xrTime_noDC = xrTime - np.mean(xrTime)
    xiTime_noDC = xiTime - np.mean(xiTime)
    yrTime_noDC = yrTime - np.mean(yrTime)
    yiTime_noDC = yiTime - np.mean(yiTime)

    # Variables for plotting
    samplingFrequency = 1/TBIN
    lowerBound = centerFrequency + CHAN_BW/2
    upperBound = centerFrequency - CHAN_BW/2

    # Convert frequencyResolution and integrationTime to FFT representation
    samplesPerTransform = int((1/frequencyResolution)/TBIN)
    fftsPerIntegration = int(integrationTime * frequencyResolution)

    ###################Waterfall

    waterfallData_x = np.zeros((numberOfIntegrations,samplesPerTransform))
    waterfallData_y = np.zeros((numberOfIntegrations,samplesPerTransform))

    for integration in range(numberOfIntegrations):
        index = integration * fftsPerIntegration * samplesPerTransform
        for individualFFT in range(fftsPerIntegration):
            waterfallData_x[integration,:] += np.abs(np.fft.fftshift(np.fft.fft(xrTime_noDC[index + individualFFT*samplesPerTransform:index + (individualFFT+1)*samplesPerTransform] + 1j*xiTime_noDC[index + individualFFT*samplesPerTransform:index + (individualFFT+1)*samplesPerTransform])))**2
            waterfallData_y[integration,:] += np.abs(np.fft.fftshift(np.fft.fft(yrTime_noDC[index + individualFFT*samplesPerTransform:index + (individualFFT+1)*samplesPerTransform] + 1j*yiTime_noDC[index + individualFFT*samplesPerTransform:index + (individualFFT+1)*samplesPerTransform])))**2

    plt.figure()
    plt.imshow(waterfallData_x, cmap = 'viridis', aspect = 'auto', extent = [lowerBound, upperBound, 0, 0.18])
    plt.title("Waterfall Plot: Channel " + str(CHANNEL) + ": X Polarization")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Time")
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.imshow(waterfallData_y, cmap = 'viridis', aspect = 'auto', extent = [lowerBound, upperBound, 0, 0.18])
    plt.title("Waterfall Plot: Channel " + str(CHANNEL) + ": Y Polarization")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Time")
    plt.colorbar()
    plt.show()

    ###################Chanel Spectrum

    integrated_spectrum_x = np.sum(waterfallData_x, axis = 0)
    integrated_spectrum_y = np.sum(waterfallData_y, axis=0)

    plt.plot(np.linspace(lowerBound, upperBound, samplesPerTransform), integrated_spectrum_x)
    plt.title("Channel " + str(CHANNEL) + ": X Polarization")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Spectrum_X")
    plt.show()

    plt.plot(np.linspace(lowerBound, upperBound, samplesPerTransform), integrated_spectrum_y)
    plt.title("Channel " + str(CHANNEL) + ": Y Polarization")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Spectrum_Y")
    plt.show()



    #SET UP Big Plot
    plt.figure(0)

    # Full Bandpass
    ax1 = plt.subplot2grid((18,5), (0,0), colspan=5, rowspan=3)
    ax1.set_title("Full Bandpass")

    # Channel Spectra
    ax2 = plt.subplot2grid((18,5), (5,0), colspan=2, rowspan=3)
    ax2.set_title("Channel Spectrum: X")
    ax2.set_xlabel("Frequency (MHz)")
    ax2.plot(np.linspace(lowerBound, upperBound, samplesPerTransform), integrated_spectrum_x)
    ax3 = plt.subplot2grid((18,5), (5, 3), colspan=2, rowspan=3)
    ax3.set_title("Channel Spectrum: Y")
    ax3.set_xlabel("Frequency (MHz)")
    ax3.plot(np.linspace(lowerBound, upperBound, samplesPerTransform), integrated_spectrum_y)

    # Channel Waterfall
    ax4 = plt.subplot2grid((18,5), (10, 0), colspan=2, rowspan=3)
    ax4.set_title("Waterfall: X")
    ax4.imshow(waterfallData_x, cmap = 'viridis', aspect = 'auto', extent = [lowerBound, upperBound, 0, 0.003])
    ax4.set_xlabel("Frequency (MHz)")
    ax4.set_ylabel("Time")


    ax5 = plt.subplot2grid((18,5), (10, 3), colspan=2, rowspan=3)
    ax5.set_title("Waterfall: Y")
    ax5.imshow(waterfallData_y, cmap = 'viridis', aspect = 'auto', extent = [lowerBound, upperBound, 0, 0.003])
    ax5.set_xlabel("Frequency (MHz)")
    ax5.set_ylabel("Time")


    # Spectral Kurtosis
    ax6 = plt.subplot2grid((18,5), (15,0), colspan=2, rowspan=3)
    ax6.set_title("Spectral Kurtosis: X")
    ax7 = plt.subplot2grid((18,5), (15, 3), colspan=2, rowspan=3)
    ax7.set_title("Spectral Kurtosis: Y")
    plt.suptitle("Real-Time Spectra of Observation")

    plt.show()
    exit()
