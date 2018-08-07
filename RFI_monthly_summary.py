# Nicholas Joslyn
# Breakthrough Listen (BL) UC-Berkeley SETI Intern 2018
# Put in /etc/cron.monthly for Monthly RFI summary statistics

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sys
import subprocess
from datetime import datetime

from scipy import special
from scipy import optimize
from scipy import signal

from argparse import ArgumentParser
from matplotlib.colors import LogNorm
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages


#### Do for all four bands
BAND_NAMES = ['L', 'S', 'C', 'X']

samplesPerTransform = 16
OBSNCHAN = 64

exportPath = "ObservationRFI/" + str(datetime.now().strftime('%b%d%Y')) + "_RFI.pdf"
pp = PdfPages(exportPath)

for individualBand in BAND_NAMES:
    have_we_observed_that_band = "monthlyRFI_" + str(individualBand) + "_X.npy"
    if (int(subprocess.check_output("find -maxdepth 2 -name " + have_we_observed_that_band + " | wc -l", shell=True)) > 0):
        monthly_RFI_hits_x = np.load("ObservationRFI/monthlyRFI_" + str(individualBand) + "_X.npy")
        monthly_RFI_hits_y = np.load("ObservationRFI/monthlyRFI_" + str(individualBand) + "_Y.npy")
        monthly_RFI_counter = np.load("ObservationRFI/monthlyRFI_" + str(individualBand) + "_Counter.npy")
        monthly_RFI_freq_range = np.load("ObservationRFI/" + str(individualBand) + "_FrequencyRange.npy")

        if (individualBand == 'L'):
            numberOfBanks = 1
            numberOfNodes = 8
        elif (individualBand == 'S'):
            numberOfBanks = 1
            numberOfNodes = 8
        elif (individualBand == 'C'):
            numberOfBanks = 4
            numberOfNodes = 8
        else:
            numberOfBanks = 3
            numberOfNodes = 8

        #Convert to 1D w/ Spectral flip
        monthly_RFI_hits_x = monthly_RFI_hits_x/monthly_RFI_counter
        monthly_RFI_hits_x = np.flip(monthly_RFI_hits_x[::-1],1).reshape(-1)

        #Convert to 1D w/ Spectral flip
        monthly_RFI_hits_y = monthly_RFI_hits_y/monthly_RFI_counter
        monthly_RFI_hits_y = np.flip(monthly_RFI_hits_y[::-1],1).reshape(-1)

        raw_axis = np.linspace(monthly_RFI_freq_range[0], monthly_RFI_freq_range[1], numberOfBanks*numberOfNodes*samplesPerTransform*OBSNCHAN)

        export_fig = plt.figure(figsize=(12,10))

        export_axis1 = plt.subplot2grid((2,1), (0, 0))
        export_axis1.set_title(str(individualBand) + " | X")
        export_axis1.set_xlabel("Frequency (MHz)")
        export_axis1.set_ylabel("%")
        export_axis1.set_ylim(0,100)
        export_axis1.margins(x=0)
        export_axis1.plot(raw_axis, monthly_RFI_hits_x)

        export_axis2 = plt.subplot2grid((2,1), (1, 0))
        export_axis2.set_title(str(individualBand) + " | Y")
        export_axis2.set_xlabel("Frequency (MHz)")
        export_axis2.set_ylabel("%")
        export_axis2.margins(x=0)
        export_axis2.set_ylim(0,100)
        export_axis2.plot(raw_axis, monthly_RFI_hits_y)

        plt.close()
        pp.savefig(export_fig)

pp.close()
