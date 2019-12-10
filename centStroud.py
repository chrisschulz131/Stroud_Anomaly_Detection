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
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt


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


def strOUD(baseline, signals, lof):
    pvals = np.zeros(len(signals))
    # find LOF at signal
    scores = lof.score_samples(signals) * -1
    # find b = number of signals > given signal then calc the pvalues
    i = 0
    for score in scores:
        b = np.count_nonzero(baseline > score)
        pvalue = (b + 1) / (baseline.shape[0] + 1)
        pvals[i] = pvalue
        i += 1
    return pvals


def tuningK(baseline, pos_signals, neg_signals, lof):

    pos_pvals = strOUD(baseline, pos_signals, lof)
    neg_pvals = strOUD(baseline, neg_signals, lof)

    # Calculate the F1 Score
    true_pos = np.count_nonzero(pos_pvals > 0.05)
    false_pos = np.count_nonzero(neg_pvals > 0.05)
    recall = true_pos / len(pos_pvals)
    precision = true_pos / (true_pos + false_pos)
    f1 = (2 * precision * recall) / (precision + recall)
    print("F1 =", f1)

    # Arranging the data for the ROC curve   
    full_data = np.concatenate((pos_pvals, neg_pvals))
    good_labels = np.full(pos_pvals.shape,1, dtype=int)
    bad_labels = np.full(neg_pvals.shape,0, dtype=int)
    full_labels = np.concatenate((good_labels, bad_labels))
    fpr, tpr, threshold = roc_curve(full_labels, full_data)
    calc_auc = auc(fpr, tpr)
    print("Calculated AUC =", calc_auc)


    # Plotting the ROC curve
    plt.plot(fpr, tpr, color="orange", label="ROC")
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.title("ROC Curve", fontsize=14)
    plt.ylabel('TPR', fontsize=12)
    plt.xlabel('FPR', fontsize=12)
    plt.show()


def main():
    parms = parseArguments()
    print("For k =", parms.k)
    
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

    # Create the baseline from the FFT arrays
    baseline = np.concatenate((a_signals_fft, b_signals_fft))
    baseline = np.concatenate((baseline, c_signals_fft, d_signals_fft))

    # Update the basline with distribution of strangeness from LOF
    lof = LocalOutlierFactor(
        n_neighbors=parms.k, novelty=True, contamination='auto')
    lof.fit(baseline)
    updated_baseline = lof.negative_outlier_factor_ * -1
    updated_baseline = np.sort(updated_baseline)

    # build tuning set and calculate the F1 statistic
    tuning_set_a = rfft(parse_signals(parms.signal_dir + '/ktuning_a'))
    tuning_set_b = rfft(parse_signals(parms.signal_dir + '/ktuning_b'))
    tuning_set_c = rfft(parse_signals(parms.signal_dir + '/ktuning_c'))
    tuning_set_d = rfft(parse_signals(parms.signal_dir + '/ktuning_d'))
    tuning_set = np.concatenate((tuning_set_a, tuning_set_b))
    tuning_set = np.concatenate((tuning_set, tuning_set_c, tuning_set_d))
    tuningK(updated_baseline, tuning_set, m_signals_fft, lof)
    
    if len(parms.pvalueFile) > 0:
        # compute test p-values and save to a file
        pvals = strOUD(updated_baseline, t_signals_fft, lof)
        filename = parms.pvalueFile + ".txt"
        np.savetxt(filename, pvals, fmt="%.5f")


if __name__ == '__main__':
    main()
