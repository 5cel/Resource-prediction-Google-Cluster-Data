'''
Created on 6 nov. 2015

@author: Faisal, Jos
'''

import math

from sklearn import preprocessing
from sklearn.decomposition.pca import PCA
from sklearn.preprocessing.imputation import Imputer

import main.preprocessing.sequentialData as seq
import numpy as np
import pandas as pd


def transformStringsToNumber(table, options):
    cols_to_transform = option(options, 'columns_to_transform', None)
    if cols_to_transform is not None:
        cols = [name for name in table.columns.values
                if name in cols_to_transform]
        le = preprocessing.LabelEncoder()
        for name in cols:
            table.fillna(-1, inplace=True)
            le.fit(table[name])
            table[name] = le.transform(table[name])
    return table


def stringsToNumbers(X, colnames, options):
    cols_to_transform = option(options, 'columns_to_transform', None)
    le = preprocessing.LabelEncoder()
    if cols_to_transform is not None:
        for i, col in enumerate(colnames):
            if col in cols_to_transform:
                X[:, i][pd.isnull(X[:, i])] = -1
                le.fit(X[:, i])
                X[:, i] = le.transform(X[:, i])
    return X


def process_train_test(X_train, X_test, t_train, t_test, colnames, options):
    col_goal_name = option(options, 'average_column', None)
    if col_goal_name is not None:
        groupBy = option(options, 'average_column_groupBy',
                         'job_name_logical')
        if col_goal_name == 't':
            sequence_train = t_train
            sequence_test = t_test
        else:
            sequence_train = X_train[:, colnames.index(col_goal_name)]
            sequence_test = X_test[:, colnames.index(col_goal_name)]
        newcol_train, averages = seq.movingAverageGrouped(sequence_train, X_train,
                                                          colnames, groupBy)
        X_train = np.append(X_train,
                            newcol_train.reshape((newcol_train.shape[0], 1)),
                            1)
        colnames.append('grouped_average_%s' % col_goal_name)
        newcol_test, _ = seq.movingAverageGrouped(sequence_test, X_test, colnames,
                                                  groupBy, averages)
        X_test = np.append(X_test,
                           newcol_test.reshape((newcol_test.shape[0], 1)), 1)
        if option(options, 'delete_groupby_column', False):
            indices_to_delete = [i for i, c in enumerate(colnames)
                                 if groupBy in c]
            X_train = np.delete(X_train, indices_to_delete, 1)
            X_test = np.delete(X_test, indices_to_delete, 1)

    option_removeHigher = option(options, 'remove_training_higher_than', None)
    if option_removeHigher is not None:
        indices_to_delete = [i for i, b in enumerate(t_train)
                             if (b > option_removeHigher)]
        len_before = X_train.shape[0]
        X_train = np.delete(X_train, indices_to_delete, axis=0)
        t_train = np.delete(t_train, indices_to_delete)
        print("\t\t%i/%i train-tasks with value higher than %f deleted"
              % (len(indices_to_delete), len_before, option_removeHigher))
    if option(options, 'remove_task_usage', True):
        cols_to_delete = [j for j, c in enumerate(colnames)
                          if c.startswith('task_usage')]
        X_train = np.delete(X_train, cols_to_delete, 1)
        X_test = np.delete(X_test, cols_to_delete, 1)
        colnames = [c for i, c in enumerate(colnames)
                    if i not in cols_to_delete]
    return X_train, X_test, t_train, t_test, colnames


def process(X, t, colnames, options):
    print
    missing_values_strategy = option(options, 'missing_values', None)
    if (option(options, 'remove_killed_tasks', False)
            and any([s for s in colnames if 'start_time' in s])):
        col_srt_time = [i for i, c in enumerate(colnames) if 'start_time' in c][0]
        col_end_time = [i for i, c in enumerate(colnames) if 'end_time' in c][0]
        task_time = np.diff(X[:, [col_srt_time, col_end_time]], axis=1)

        indices_to_delete = [i for i, b in enumerate(task_time < 0) if b]
        len_before = X.shape[0]
        X = np.delete(X, indices_to_delete, axis=0)
        t = np.delete(t, indices_to_delete)
        print("\t\t%i/%i killed tasks deleted"
              % (len(indices_to_delete), len_before))
    if option(options, 'remove_zero_values', False):
        indices_to_delete = [i for i, b in enumerate(t) if (b == 0)]
        len_before = X.shape[0]
        X = np.delete(X, indices_to_delete, axis=0)
        t = np.delete(t, indices_to_delete)
        print("\t\t%i/%i tasks with zero prediction value deleted"
              % (len(indices_to_delete), len_before))
    option_removeHigher = option(options, 'remove_values_higher_than', None)
    if option_removeHigher is not None:
        indices_to_delete = [i for i, b in enumerate(t)
                             if (b > option_removeHigher)]
        len_before = X.shape[0]
        X = np.delete(X, indices_to_delete, axis=0)
        t = np.delete(t, indices_to_delete)
        print("\t\t%i/%i tasks with value higer than %f deleted"
              % (len(indices_to_delete), len_before, option_removeHigher))
    if (option(options, 'increase_zero_cpu_values', False) and
            'task_events_resource_request_for_CPU_cores' in colnames):
        col = [i for i, c in enumerate(colnames) if c ==
               'task_events_resource_request_for_CPU_cores'][0]
        new_value = np.min(X[:, col][X[:, col] > 0])
        new_value = new_value[0] if isinstance(new_value, list) else new_value
        rows_to_change = (X[:, col] == 0)
        n_changes = len([c for c in rows_to_change if c])
        X[:, col][rows_to_change] = [new_value] * n_changes
        print("\t\t%i/%i cpu values increased from 0 to %f"
              % (n_changes, X.shape[0],
                 new_value))
        assert len([c for c in (X[:, col] == 0) if c]) == 0
    if (option(options, 'increase_zero_ram_values', False) and
            'task_events_resource_request_for_RAM' in colnames):
        col = [i for i, c in enumerate(colnames) if c ==
               'task_events_resource_request_for_RAM'][0]
        new_value = np.min(X[:, col][X[:, col] > 0])
        rows_to_change = (X[:, col] == 0)
        n_changes = len([c for c in rows_to_change if c])
        X[:, col][rows_to_change] = [new_value] * n_changes
        print("\t\t%i/%i ram values increased from 0 to %f"
              % (n_changes, X.shape[0],
                 new_value))
        assert len([c for c in (X[:, col] == 0) if c]) == 0

    if missing_values_strategy is not None:
        print("\t\t%i/%i nan-elements in X"
              % (np.count_nonzero(np.isnan(X)), X.size))
        imp = Imputer(missing_values="NaN", strategy=missing_values_strategy,
                      axis=0)
#         np.savetxt("before_imputer.csv", X, delimiter=";", fmt='%.4e')
        imp = imp.fit(X)
        X = imp.transform(X)

    if option(options, 'scale', False):
        print("\t\tX scaled")
        scaler = preprocessing.StandardScaler()
        scaler = scaler.fit(X)
        X = scaler.transform(X)

    if option(options, 'moving_average', False):
        windowsize = int(option(options, 'moving_average_windowsize', 100))
        moving_average = seq.movingAverage(t, windowsize)
        X = np.append(X, moving_average.reshape((moving_average.shape[0], 1)),
                      1)
        colnames.append('moving_average_n=%i' % windowsize)

    if option(options, 'PCA', False):
        print("\t\tPCA")
        n_components = option(options, 'PCA_ncomponents',
                              int(math.ceil(X.shape[1] / 2)))
        pca = PCA(n_components=n_components, whiten=True)
        pca.fit(X, t)
        X = pca.transform(X)

    if(option(options, 'daily_representation', False)
       and 'task_usage_start_time_of_the_measurement_period' in colnames):
        col = colnames.index('task_usage_start_time_of_the_measurement_period')
        X[:, col] = np.mod(X[:, col], np.full(X[:, col].shape, 86400000000))
        print("\t\tDaily representation used")

    if option(options, 'remove_task_usage', True):
        indices_to_delete = [i for i, c in enumerate(colnames)
                             if c.startswith('task_usage')
                             and 'task_usage_mean_CPU_usage_rate' not in c]
        if option(options, 'keep_start_time', False):
            indices_to_delete = [i for i, c in enumerate(colnames)
                                 if i in indices_to_delete
                                 and 'start_time' not in c]
        X = np.delete(X, indices_to_delete, 1)
        colnames = [c for i, c in enumerate(colnames)
                    if i not in indices_to_delete]
        print("\t\t%i columns deleted from X, containing task_usage info"
              % len(indices_to_delete))
    assert X.shape[1] == len(colnames)
    return(X, t, colnames)


def option(options, name, defaultValue):
    return options[name] if name in options else defaultValue
