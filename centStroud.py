'''
Template for Final Project/Anomaly Detection with StrOUD
Kevin Molloy -- November 2019

This template provides the code to read in the signal data
and apply a FFT to extract the signal data and build
features (the FFT coefs).

Students need to implement the StrOUD method to formulate a
p-value for each test data point, print these p-values
to a file (as described in the project description) and submit
these to Autolab.
'''

from pathlib import Path

import sys
import argparse
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from array import array


def parseArguments():
    parser = argparse.ArgumentParser(
        description='StrOUD')

    parser.add_argument('--signalDir', action='store',
                        dest='signal_dir', default="", required=True,
                        help='directory where signal directories reside')

    parser.add_argument('--pvalueFile', action='store',
                        dest='pvalueFile', default="", required=False,
                        help='file where the test data pvalues are written')
    
    parser.add_argument('--k', action='store',
                        dest='k', default="", type=int, required=True,
                        help='K value for LOF calculations')
    
    return parser.parse_args()


def parse_signals(base_path):

    signals = []

    for d in sorted(Path(base_path).glob('*.txt'), key=lambda p: int(p.stem[4:])):

        samples = [float(v) for v in d.read_text().split()]

        if len(samples) > 0:  # skip records that are empty
            signals.append(samples)
        else:
            print(d + ' file skipped, was empty')

    return np.asanyarray(signals)


def addStrangeness(baseline, k):
    # create distribution using LOF
    lof = LocalOutlierFactor(n_neighbors=k, novelty=True)
    lof.fit(baseline)

    lof_baseline = lof.negative_outlier_factor_ * -1
    lof_baseline = np.sort(lof_baseline)
    # print(lof_baseline)
    return lof_baseline

def StrOUD(t_signals_fft, baseline): # TODO finish method
    pval_array = array('i')
    for signal in t_signals_fft:
        #find LOF at signal
        # find b = number of signals > given signal
        pvalue = 0
        pval_array.append(pvalue)
    
    return pval_array

def main():
    parms = parseArguments()

    a_signals = parse_signals(parms.signal_dir + '/ModeA')
    b_signals = parse_signals(parms.signal_dir + '/ModeB')
    c_signals = parse_signals(parms.signal_dir + '/ModeC')
    d_signals = parse_signals(parms.signal_dir + '/ModeD')
    m_signals = parse_signals(parms.signal_dir + '/ModeM')
    t_signals = parse_signals(parms.signal_dir + '/TestSignals')

    # Fast Fourier Transform (FFT)

    from scipy.fftpack import rfft

    a_signals_fft = rfft(a_signals)
    b_signals_fft = rfft(b_signals)
    c_signals_fft = rfft(c_signals)
    d_signals_fft = rfft(d_signals)
    m_signals_fft = rfft(m_signals)
    t_signals_fft = rfft(t_signals)

    # build tuning set 
    tuning_set_a = rfft(parse_signals(parms.signal_dir + '/ModeAFirstHalf'))
    tuning_set_b = rfft(parse_signals(parms.signal_dir + '/ModeBFirstHalf'))
    tuning_set_c = rfft(parse_signals(parms.signal_dir + '/ModeCFirstHalf'))
    tuning_set_d = rfft(parse_signals(parms.signal_dir + '/ModeDFirstHalf'))
    tuning_set_m = rfft(parse_signals(parms.signal_dir + '/ModeMFirstHalf'))
    tuning_set = np.concatenate((tuning_set_a, tuning_set_b))
    tuning_set = np.concatenate((tuning_set, tuning_set_c, tuning_set_d))
    tuning_set = np.concatenate((tuning_set, tuning_set_m))

    # Create the baseline from the FFT arrays
    baseline = np.concatenate((a_signals_fft, b_signals_fft))
    baseline = np.concatenate((baseline, c_signals_fft, d_signals_fft))

    # Update the basline with distribution of strangeness from LOF
    updated_baseline = addStrangeness(baseline, parms.k)

    if len(parms.pvalueFile) > 0:
        StrOUD(t_signals_fft, updated_baseline)
        

if __name__ == '__main__':
    main()
