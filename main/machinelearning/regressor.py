'''
Created on 4 nov. 2015

@author: Faisal, Jos
'''
from sklearn.neighbors.regression import KNeighborsRegressor
import tabulate

import numpy as np
import sklearn.ensemble as ensemble
import sklearn.svm as svm
import sklearn.tree as tree


def fit(X, t, colnames, options, verbose):
    alg = option(options, 'algorithm', 'random_forest')
    print 'Using %s' % alg
    if alg == 'random_forest':
        n_estimators = option(options, 'n_estimators', None)
        bootstrap = option(options, 'bootstrap', None)
        criterion = option(options, 'criterion', None)
        max_depth = None
        max_features = int(np.floor(np.sqrt(X.shape[1])))
        rf = ensemble.RandomForestRegressor(n_estimators=n_estimators,
                                            criterion=criterion,
                                            bootstrap=bootstrap,
                                            max_depth=max_depth,
                                            max_features=max_features)
        model = rf.fit(X, t)
        if verbose:
            print tabulate.tabulate([colnames, rf.feature_importances_])
    elif alg == 'adaboost':
        n_estimators = int(option(options, 'n_estimators', 100))
        learning_rate = 0.99
        loss = 'exponential'
        reg = ensemble.AdaBoostRegressor(n_estimators=n_estimators,
                                         learning_rate=learning_rate,
                                         loss=loss)
        model = reg.fit(X, t)
    elif alg == 'baseline_average_overJobid':
        model = None
    elif alg == 'baseline_average':
        model = np.average(t)
    elif alg == 'baseline_users_limit':
        model = None
    elif alg == 'SVM':
        kernel = option(options, 'kernel', 'poly')
        degree = option(options, 'degree', 3)
        gamma = option(options, 'gamma', 0.1)
        tol = option(options, 'tol', 0.01)
        shrinking = option(options, 'schrinking', True)
        max_iter = option(options, 'max_iter', 10000)
        C = option(options, 'kernel', 1.0)
        epsilon = option(options, 'epsilon', 0.1)
        clf = svm.SVR(kernel=kernel, degree=degree, gamma=gamma, coef0=0.0,
                      tol=tol, C=C, epsilon=epsilon, shrinking=shrinking,
                      max_iter=max_iter)
        model = clf.fit(X, t)
    elif alg == 'KNN':
        K = 100
        W = 'distance'
        leaf_size = 30
        knn = KNeighborsRegressor(K, W, leaf_size=leaf_size)
        model = knn.fit(X, t)
    else:
        raise NotImplementedError("alg %s not implemented" % alg)
    return model


def predict(x, colnames, model, options):
    alg = option(options, 'algorithm', 'random_forest')
    if alg in ['random_forest', 'SVM', 'KNN', 'adaboost']:
        y = model.predict(x)
    elif alg == 'baseline_average':
        y = np.array([model] * x.shape[0])
    elif alg == 'baseline_users_limit':
        predicting = option(options, 'predicting', 'maximum_memory_usage')
        if predicting == 'maximum_memory_usage':
            predictor = 'task_events_resource_request_for_RAM'
        elif predicting == 'maximum_CPU_usage':
            predictor = 'task_events_resource_request_for_CPU_cores'
        col = [i for i, name in enumerate(colnames) if name == predictor][0]
        y = x[:, col]
    elif alg == 'baseline_average_overJobid':
        y = x[:, colnames.index('grouped_average_t')]
    else:
        raise NotImplementedError("alg %s not implemented" % alg)
    return y


def option(options, name, defaultValue):
    return options[name] if name in options else defaultValue
