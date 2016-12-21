'''
Created on 25 nov. 2015

@author: Faisal, Jos
'''
import numpy as np


def movingAverage(sequence, window_size=10):
    '''
    Credits to stackoverflow user Lapis for this solution:
    http://stackoverflow.com/questions/13728392/moving-average-or-running-mean
    '''

    moving_averages = np.convolve(sequence,
                                  np.ones((window_size, )) / window_size,
                                  mode="valid")
    prepend = [0.]
    for i in range(1, window_size):
        prepend.append(np.average(sequence[:i]))
    return np.concatenate((prepend, moving_averages[:-1]))
#     print np.random.randn(sequence.shape[0]).shape
#     print np.random.randn(sequence.shape[0])[:10]
#     return np.random.randn(sequence.shape[0])


def movingAverageGrouped(sequence, X, colnames, groupBy, averages=None):
    col_merge_number = [i for i, c in enumerate(colnames) if groupBy in c][0]
    col_merge_unique = list(set(X[:, col_merge_number].flat))

    if averages is None:
        averages = {c: np.average([sequence.reshape(X[:, col_merge_number].shape)[X[:, col_merge_number] == c]])
                    for c in col_merge_unique}
    default_value = np.average(averages.values())
    newcol = [averages.get(X[i, col_merge_number], default_value)
              for i, _ in enumerate(sequence)]
    return np.asarray(newcol), averages
