"""
Show how to connect to keypress events
"""
from __future__ import print_function
import sys
import numpy as np
import matplotlib.pyplot as plt


def press(event):
    global node, bank

    sys.stdout.flush()
    if event.key == 'x':
        visible = axis1_desired.get_visible()
        axis1_desired.set_visible(not visible)
        #fig.canvas.draw()
    if event.key == 'up':
        bank += 1
        if (bank > 3):
            bank = 0
    if event.key == 'down':
        bank -= 1
        if (bank < 0):
            bank = 3
    if event.key == 'right':
        node += 1
        if (node > 7):
            node = 0
    if event.key == 'left':
        node -= 1
        if (node < 0):
            node = 7

    #ax.plot(np.random.rand(12), np.random.rand(12), 'go')
    plt.suptitle("Observation: >>Grab Name/Date<< | blc" + str(bank) + str(node))
    #fig.canvas.draw()



if __name__ == '__main__':
    #SET UP Big Plot
    bank = 0
    node = 0
    plt.figure("Test")
    plt.suptitle("Observation: >>Grab Name/Date<< | blc" + str(bank) + str(node))
    plt.ion()
    plt.show()

    # Full observational range
    axis1_desired = plt.subplot2grid((18,5), (0,0), colspan=5, rowspan=3)
    axis1_desired.set_title("Full Observation Spectrum (X)")
    axis1_desired.set_yscale("log")
    axis1_desired.set_ylabel("Power")
    axis1_desired.set_xlabel("Frequency (MHz)")
    #axis1_desired.plot(current_axis, bandPass_x, color = 'black')

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
    plt.connect('key_press_event', press)

    plt.pause(10)


    # fig, ax = plt.subplots()
    # node = 0
    # bank = 0
    #
    # fig.canvas.mpl_connect('key_press_event', press)
    #
    # ax.plot(np.random.rand(12), np.random.rand(12), 'go')
    # xl = ax.set_xlabel('easy come, easy go')
    # ax.set_title("blc" + str(bank) + str(node))
    # plt.show()
