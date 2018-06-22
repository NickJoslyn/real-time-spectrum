# Nicholas Joslyn
# Breakthrough Listen UC Berkeley SETI Intern 2018

# Program for real-time spectra of BL Observations
import numpy as np
import matplotlib.pyplot as plt

def convert_resolution(customFrequencyResolution, customTimeResolution, TBIN):
    """
    Convert custom frequency (Hz) resolution and time (s) resolution into FFT parameters

    Return:
    samplesPerTransform (int):  Number of samples for each FFT
    fftsPerIntegration (int):   Number of FFTs to integrate over
    """

    samplesPerTransform = int((1/customFrequencyResolution)/TBIN)
    fftsPerIntegration = int(customTimeResolution * customFrequencyResolution)

    return samplesPerTransform, fftsPerIntegration

def remove_DCoffset(BLOCK):

    channels, samples, polarizations = BLOCK.shape
    DC_offset = np.mean(BLOCK, axis = 1).reshape(channels, 1, polarizations)
    _, DC_offset = np.broadcast_arrays(BLOCK, DC_offset)
    return (BLOCK - DC_offset)


def real_time_spectra(BLOCK, CHANNEL, CHAN_BW, TBIN, samplesPerTransform, fftsPerIntegration, numberOfIntegrations):
    """
    Plot spectra and stats of real-time observational BL data on block.

    """

    #Need to expand from channel analysis to full compute node analysis (i.e. band analysis)

    ###################Prepare Data
    print(samplesPerTransform, fftsPerIntegration, numberOfIntegrations)
    print(BLOCK.shape)
    print(np.mean(BLOCK, axis =1))
    BLOCK = remove_DCoffset(BLOCK)
    print(BLOCK.shape)
    print(np.mean(BLOCK, axis =1))

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
