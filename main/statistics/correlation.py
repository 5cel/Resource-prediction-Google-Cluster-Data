'''
Created on 25 nov. 2015

@author: Faisal, Jos
'''
from scipy.stats.stats import kendalltau
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr
import tabulate

import cPickle as pickle
from main.preprocessing.sequentialData import movingAverage
from main.preprocessing.sequentialData import movingAverageGrouped
import matplotlib.pyplot as plt
import numpy as np
import sklearn.cross_validation as cross_validation


def correlation_of_averagingMethod(t, method, col, statistics=[pearsonr],
                                   windowsizes=[1, 10, 20, 50, 100, 150, 200,
                                                1000, 10000]):
    table = [['Statistic'] + ['n=%i' % n for n in windowsizes]]
    for statistic in statistics:
        row = [statistic.__name__]
        for n in windowsizes:
            newcol = method(col, n)
            r = statistic(t, newcol)[0]
            row += ['%.3f' % r]
        table.append(row)
    print tabulate.tabulate(table)


def histogram_of_single_job(X, t, colnames):
    idx = colnames.index('job_events_job_name_logical')
    job_idx = colnames.index('task_usage_job_ID')
    logical_jobnames = np.unique(X[:, idx].flat)
    stdevs, avs, vals, jobids = [], [], [], []
    for i, name in enumerate(logical_jobnames):
        vals.append(t[X[:, idx].flat == name])
        jobids.append(X[:, job_idx][X[:, idx].flat == name][0])
        stdevs.append(np.std(vals[i]))
        avs.append(np.average(vals[i]))
    print 'im here'

#     enough_stdevs = [stdev for i, stdev in enumerate(stdevs) if len(vals[i]) > 100]
#     enough_averages = [av for i, av in enumerate(avs) if len(vals[i]) > 100]
    
    a = [i for i, _ in enumerate(vals) if len(vals[i]) > 100 and avs[i] > stdevs[i] and stdevs[i] > 1/5 * avs[i]]
    c = [i for i, _ in enumerate(vals) if len(vals[i]) > 100 and avs[i] < 0.005]
    
#     
#     maxstdev = [i for i, stdev in enumerate(enough_stdevs) if stdev == max(enough_stdevs)][0]
#     min_avg = [i for i, av in enumerate(enough_averages) if av == min(enough_averages)][0]
#     most_vals = [i for i, _ in enumerate(stdevs) if len(vals[i]) == max([len(val) for val in vals])][0]

    for i in range(10):
        print jobids[a[i]]
        plt.figure()
        plt.hist(vals[a[i]], bins = 50)
        plt.show()
        plt.figure()
        print str(jobids[c[i]])
        plt.hist(vals[c[i]], bins = 50)
        plt.show()



def show_correlation_of(groupBy='job_name_logical',
                        col_goal_name='task_usage_mean_CPU_usage_rate'):
    print tabulate.tabulate([colnames])
    kf = cross_validation.KFold(len(X), 5, shuffle=True)
    correlations = [['Iteration', 'train', 'test']]
    for i, (train, test) in enumerate(kf):
        X_train, X_test, t_train, t_test = X[train], X[test], t[train], t[test]
#         col_goal_name = 'task_events_resource_request_for_CPU_cores'
        sequence_train = X_train[:, colnames.index(col_goal_name)]
        newcol_train, averages = movingAverageGrouped(sequence_train, X_train,
                                                      colnames, groupBy)
        sequence_test = X_test[:, colnames.index(col_goal_name)].flat
        newcol_test, _ = movingAverageGrouped(sequence_test, X_test, colnames,
                                              groupBy, averages)
        row = [i] + correlation_of_column(t_train, newcol_train) + correlation_of_column(t_test, newcol_test)
        correlations.append(row)
    print tabulate.tabulate(correlations)


def correlation_of_all_columns(X, t):
    kf = cross_validation.KFold(len(X), 10, shuffle=True)
    correlations = [['Column'] + ['train', 'test'] * 5]
    for i, colname in enumerate(colnames):
        row = [colname]
        for i, (train, test) in enumerate(kf):
            kf = cross_validation.KFold(len(X), 5, shuffle=True)
            row.append('%.3f' % correlation_of_column(t[train], X[train][:, i].flat)[0])
            row.append('%.3f' % correlation_of_column(t[test], X[test][:, i].flat)[0])
        correlations.append(row)
    print tabulate.tabulate(correlations)


def correlation_of_column(t, newcol, statistics=[pearsonr]):
    corrs = []
    for statistic in statistics:
        r = statistic(t, newcol)[0]
        corrs.append(r)
    return corrs


def value_of_same_col(X, t, colnames, col='job_ID'):
    col_number = [i for i, c in enumerate(colnames) if col in c][0]
    col_unique = list(set(X[:, col_number].flat))
    boxpl_50min, boxpl_50plus, hist = [], [], []
    number_50plus, number_50min = 0, 0
    for col_value in col_unique:
        t_of_col = [t_val for i, t_val in enumerate(t)
                    if X[i, col_number] == col_value]
        hist.append(len(t_of_col))
        if number_50plus < 200 and len(t_of_col) > 50:
            boxpl_50plus.append([t_val for i, t_val in enumerate(t)
                                 if X[i, col_number] == col_value])
            number_50plus += 1
        elif number_50min < 200 and 10 < len(t_of_col) < 50:
            boxpl_50min.append([t_val for i, t_val in enumerate(t)
                                if X[i, col_number] == col_value])
            number_50min += 1
    plt.hist(hist, bins=50, range=(0, 50), log=True)
    print len([h for h in hist if h == 1])
    plt.show()
    plt.boxplot(boxpl_50min)
    plt.show()
    plt.boxplot(boxpl_50plus)
    plt.show()


def average_of_same_col(X, colnames, col_goal_name, col_merge='job_name_logical'):
    col_merge_number = [i for i, c in enumerate(colnames) if col_merge in c][0]
    col_merge_unique = list(set(X[:, col_merge_number].flat))
    col_goal_number = [i for i, c in enumerate(colnames) if col_goal_name in c][0]

    values = {c: np.average(X[:, col_goal_number][X[:, col_merge_number] == c])
              for c in col_merge_unique}
    newcol = [values[X[i, col_merge_number]]
              for i, _ in enumerate(X[:, col_goal_number])]
    return newcol

if __name__ == '__main__':
    pref = 'pickles/maximum_CPU_usage_iglesias_100000'
    X = pickle.load(open('../../%s_X.p' % pref, "rb"))
    t = pickle.load(open('../../%s_t.p' % pref, "rb"))
    colnames = pickle.load(open('../../%s_col.p' % pref, "rb"))
#     value_of_same_col(X, t, colnames)
    histogram_of_single_job(X, t, colnames)
