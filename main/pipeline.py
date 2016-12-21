'''
Created on 3 nov. 2015

@author: Faisal, Jos
'''
import sys
import time

import cPickle as pickle
import main.filehandling.readCsv as readCsv
import main.machinelearning.regressor as regressor
import main.preprocessing.featureConstruction as featureConstructor
import main.preprocessing.prepro as prepro
import main.visualization.results as visualize
import numpy as np
import sklearn.cross_validation as cross_validation


def runPipeline(n, n_folds, X_attrs, t_attrs, options_preprocess,
                options_machinelearning, options_pickling, folder_data,
                verbose=False, X=None, t=None, colnames=None):
    print('\tRetrieving X and t...'),
    sys.stdout.flush()
    t0 = time.time()

    if options_pickling['Xt']:
        X = _load(options_pickling, 'X_name')
        t = _load(options_pickling, 't_name')
        colnames = _load(options_pickling, 'columns_name')
    else:
        if None in [X, t, colnames]:
            X, t, colnames, i, t0 = _readFromCsv(n, X_attrs, t_attrs,
                                                 options_preprocess,
                                                 folder_data, t0)
        else:
            X = prepro.stringsToNumbers(X, colnames, options_preprocess)
            X = X.astype('float64')
            t = t.astype('float64')
        _save(options_pickling, 'X_name', X)
        _save(options_pickling, 't_name', t)
        _save(options_pickling, 'columns_name', colnames)

    _printStartAndEndTime(X, colnames)
    t0 = _printNewAction(t0, 'Preprocessing')
    X, t, colnames = prepro.process(X, t, colnames, options_preprocess)

    t0 = _printNewAction(t0, 'Training %s'
                         % options_machinelearning['algorithm'])
    print('On %i folds' % n_folds)
    kf = cross_validation.KFold(len(X), n_folds,
                                shuffle=options_machinelearning['shuffle'])
    MAE_errors = []
    RMSE_errors = []
    for i, (train, test) in enumerate(kf):
        X_train, X_test, t_train, t_test, colnames_run\
            = prepro.process_train_test(X[train], X[test], t[train], t[test],
                                        colnames, options_preprocess)

        if 'job_events_job_name_logical' in colnames_run:
            idx = colnames_run.index('job_events_job_name_logical')
            logical_jobnames_train = np.unique(X_train[:, idx].flat)
            logical_jobnames_test = np.unique(X_test[:, idx].flat)
            new_elements = [el for el in logical_jobnames_test
                            if el not in logical_jobnames_train]
            print('number of new elements %i. El in test: %i. in train: %i n: %i'
                  % (len(new_elements), len(logical_jobnames_test),
                     len(logical_jobnames_train), n))
        model = regressor.fit(X_train, t_train, colnames_run,
                              options_machinelearning, verbose)
        y = regressor.predict(X_test, colnames_run, model, options_machinelearning)
        y_smaller_than_zero = np.where(y < 0)[0].shape[0]
        if y_smaller_than_zero > 0:
            print '%i elements smaller than zero' % y_smaller_than_zero
#         y[y > 0.5] = 0.5
        MAE, RMSE = visualize.ty(X_test, t_test, y,
                                 options_machinelearning['folder_results'],
                                 fold=i, verbose=verbose)
        MAE_errors.append(MAE)
        RMSE_errors.append(RMSE)
    _printNewAction(t0, None)
    MAE_average = np.average(MAE_errors)
    MAE_std = np.std(MAE_errors)
    RMSE_average = np.average(RMSE_errors)
    RMSE_std = np.std(RMSE_errors)
    print('Total error on %i fold cross-validation: MAE: %f\t\t RMSE: %f'
          % (n_folds, MAE_average, RMSE_average))
    return MAE_average, MAE_std, RMSE_average, RMSE_std


def _printNewAction(t0, new_action):
    t1 = time.time()
    print '\tDone (%.1f s).' % (t1 - t0)
    if new_action is not None:
        print ('\t%s...' % new_action),
        t0 = time.time()
    sys.stdout.flush()
    return t0


def _save(options, name, data):
    f = "%s/%s.p" % (options['folder'], options[name])
    pickle.dump(data, open(f, "wb+"))


def _load(options, name):
    f = "%s/%s.p" % (options['folder'], options[name])
    return pickle.load(open(f, "rb"))


def _readFromCsv(n, X_attrs, t_attrs, options_preprocess, folder_data, t0):
    table_names = list(set(X_attrs.keys() + t_attrs.keys()))
    print len(table_names)
    filenames = [folder_data + f + '/' for f in table_names]
    tables = [prepro.transformStringsToNumber(readCsv.readFolder(f),
                                              options_preprocess)
              for f in filenames]
    tuple__table_name = [(tables[i], table_names[i])
                         for i in range(len(filenames))]
    t0 = _printNewAction(t0, 'Constructing features')
    i = [i for i, (_, name) in enumerate(tuple__table_name)
         if name == 'task_usage'][0]
    X, t, colnames = featureConstructor.Xt(tuple__table_name, X_attrs,
                                           t_attrs, n)
    return X, t, colnames, i, t0


def _printStartAndEndTime(X, colnames):
    if(any(['start_time' in colname in colname
            for colname in colnames]) and
       any(['end_time' in colname in colname
            for colname in colnames])):
        str_times = set(X[:, [i for i, c in enumerate(colnames)
                              if 'start_time' in c][0]].flat)
        end_times = set(X[:, [i for i, c in enumerate(colnames)
                              if 'end_time' in c][0]].flat)
        durage = np.diff(X[:, [[i for i, c in enumerate(colnames)
                                if 'start_time' in c][0],
                               [i for i, c in enumerate(colnames)
                                if 'end_time' in c][0]
                               ]], axis=1)
        print('\n\t\tstart_time (s) min-max: (%i, %i)'
              % (min(str_times) / 1000000, max(str_times) / 1000000))
        print('\t\tend_time (s) min-max: (%i, %i)'
              % (min(end_times) / 1000000, max(end_times) / 1000000))
        print('\t\taverage durage (s): %.1f' % (np.mean(durage) / 1000000))
        sys.stdout.flush()
