# real-time-spectrum
Display spectra of a Breakthrough Listen observation in real-time

During a Breakthrough Listen Observation, this program can be run to both monitor the data quality and produce important observation information (waterfall plots of compute nodes and RFI details). The program will detect when there is no active observation. Also, it will detect when an observation starts up. Therefore, the program can be started at any time, and ideally, left running continuously.

Breakthrough Listen observations utilize multiple compute nodes (organized in banks) to process the very large bandwidth of their observations. Simultaneously, different compute nodes handle different frequency ranges. Thus, in the following discussions, by looking at different compute nodes/banks, the observer looks at different spectral windows.

ObservationRFI and ObservationWaterfalls sub-directories will be created in the same location as the program is called from if they do not previously exist. These sub-directories are where the .pdf's are saved.

### realtime_visualization.py

Green Bank Version

Run this program from a storage node (**_or any node with all compute nodes -- include blc18 -- mounted_**)

Program is most effective if VNC to GBO gateway, then ssh -X to the storage node.

### realtime_visualization_parkes.py

Parkes Version

Run this program from a storage node (**_or any node with all compute nodes mounted_**)

Both versions follow a very similar strategy, but have slight differences. More verbose comments are located in the Green Bank version, but Parkes is well documented as well.

## Visualization Interface

This automated program produces 10 plot figure for real-time visualization of Breakthrough Listen observations

1. Overall bandpass of observation (specified rack of compute nodes in red)
2. Overall bandpass of observation for specified rack of compute nodes (specified compute node in red)
3. Spectrum of the X polarization for specified compute node
4. Spectrum of the Y polarization for specified compute node
5. Spectral Kurtosis of the X polarization for specified compute nodes
6. Spectral Kurtosis of the Y polarization for specified compute nodes
7. Cross Spectrum for specified compute node
8. Spectral Kurtosis of Cross Spectrum for given compute node
9. Waterfall plot showing approximately the last hour's spectral information of X polarization for specified compute node
10. Waterfall plot showing approximately the last hour's spectral information of Y polarization for specified compute node

## User Interactivity

The following keystrokes have effects on the program

* Right/Left Arrow Keys: Cycle through compute nodes of a compute bank
* Up/Down Arrow Keys: Cycle through compute banks
* x: Toggle X/Y polarization of bandpass diagnostic plots
* p: Pause the program. User will be asked in the terminal if they want to restart the program
* q: Abort the program. Quickest way to quit the program.

## Exports

The following PDF files are exported

* Approximately hourly waterfall plots of all compute nodes
* Post-observation waterfall plot of final hour
* Post-observation %RFI per frequency bin for entire observation

These can be sent to Slack if specified by command line arguments.

After the observation, the following numpy binary (.npy) files are exported

* %RFI per frequency bin X polarization
* %RFI per frequency bin Y polarization
* If it does not exist, the frequency range for the observing band
* A counter of how many times this observing band has been used in the past month

These exported binary files will be used in a monthly cron job (which calls **RFI_monthly_summary.py**) to produce monthly RFI statistics.

---

### Green Bank
 ```
 >>>  python realtime_visualization.py -h

usage: realtime_visualization.py [-h] [-f FILES_PER_EXPORT] [-b NODES_IN_BANK]
                                 [-c CHANNELS_PER_NODE]
                                 [-s SAMPLES_PER_TRANSFORM]
                                 [-i FFTS_PER_INTEGRATION] [-t SLACK_TOKEN]
                                 [-u SLACK_CHANNEL]

Produces real-time spectral information display. Creates summary waterfall and
RFI pdfs.

optional arguments:
  -h, --help            show this help message and exit
  -f FILES_PER_EXPORT   Files Per Export. The number of raw files analyzed
                        before exporting waterfall plots. Default: 60
  -b NODES_IN_BANK      Nodes per bank. Program assumes total number of
                        compute nodes is multiple of this value. Default: 8
                        (unlikely to change from default)
  -c CHANNELS_PER_NODE  Channels per node. Default: 64 (standard for GBT)
  -s SAMPLES_PER_TRANSFORM
                        Time Samples per FFT. Default: 16 (Gives 0.183MHz
                        resolution)
  -i FFTS_PER_INTEGRATION
                        Number FFTs to accumulate. Default: 50 (Gives 0.273ms
                        integration time)
  -t SLACK_TOKEN        Slack token. Specifying token allows PDFs to be
                        exported to Slack. Default: No
  -u SLACK_CHANNEL      Slack channel username. Specify active_observations
                        channel. Must specify if using Slack. Default: No
```
### Parkes
```
>>> python realtime_visualization_parkes.py -h

usage: realtime_visualization_parkes.py [-h] [-f FILES_PER_EXPORT]
                                        [-b NODES_IN_BANK]
                                        [-c CHANNELS_PER_NODE]
                                        [-s SAMPLES_PER_TRANSFORM]
                                        [-i FFTS_PER_INTEGRATION]
                                        [-t SLACK_TOKEN] [-u SLACK_CHANNEL]

Produces real-time spectral information display. Creates summary waterfall and
RFI pdfs.

optional arguments:
  -h, --help            show this help message and exit
  -f FILES_PER_EXPORT   Files Per Export. The number of raw files analyzed
                        before exporting waterfall plots. Default: 60
  -b NODES_IN_BANK      Nodes per bank. Program assumes total number of
                        compute nodes is multiple of this value. Default: 13
  -c CHANNELS_PER_NODE  Channels per node. Default: 44 (standard for Parkes)
  -s SAMPLES_PER_TRANSFORM
                        Time Samples per FFT. Default: 16
  -i FFTS_PER_INTEGRATION
                        Number FFTs to accumulate. Default: 50
  -t SLACK_TOKEN        Slack token. Specifying token allows PDFs to be
                        exported to Slack. Default: No
  -u SLACK_CHANNEL      Slack channel username. Specify active_observations
                        channel. Must specify if using Slack. Default: No
```
