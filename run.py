'''
Created on 12 nov. 2015

@author: Faisal, Jos
'''
import argparse
import os
import sys

import main.pipeline as pipeline
import test.preparedata.prepareData as prepD


def attributesSqlToPaper():
    return {'task_events':
                   {'priority': 'priority',
                    'CPU_request': 'resource_request_for_CPU_cores',
                    'memory_request': 'resource_request_for_RAM',
                    'different_machines_restriction': 'different-machine_constraint',
                    'machine_ID': 'machine_ID',
                    'job_ID': 'job_ID'
                    },
            'job_events':
                   {'JE_scheduling_class': 'scheduling_class',
                    'JE_User': 'user_name',
                    'job_name': 'job_name',
                    'logical_job_name': 'job_name_logical'
                    },
            'task_usage':
                   {'start_time': 'start_time_of_the_measurement_period',
                    'end_time': 'end_time_of_the_measurement_period',
                    'job_ID': 'job_ID',
                    'task_index': 'task_index'}
            }


def _options_machinelearning(_type, predict, shuffle, folder_results,
                             rf_bootstrap, rf_estimators):
    if _type == 'random_forest':
        options_machinelearning = {'algorithm': _type,
                                   'n_estimators': int(rf_estimators),
                                   'bootstrap': rf_bootstrap,
                                   'criterion': 'mse',
                                   'shuffle': shuffle,
                                   'folder_results': folder_results}
    elif _type == 'baseline_users_limit':
        options_machinelearning = {'algorithm': _type,
                                   'predicting': predict,
                                   'shuffle': shuffle,
                                   'folder_results': folder_results}
    else:
        options_machinelearning = {'algorithm': _type,
                                   'shuffle': shuffle,
                                   'folder_results': folder_results}
    return options_machinelearning


def _attributes(type_X, type_T):
    if type_X == 'CPU_only':
        X_attrs = {'task_events': ['CPU_request']}
    elif type_X == 'RAM_only':
        X_attrs = {'task_events': ['memory_request']}
    elif type_X == 'Grouping_only':
        X_attrs = {'job_events': ['logical_job_name']}
    elif type_X == 'Grouping_withTime':
        X_attrs = {'job_events': ['logical_job_name'],
                   'task_usage': ['start_time', 'end_time']}
    elif type_X == 'logicalJobname_Machine_StartTime':
        X_attrs = {'task_events': ['machine_ID'],
                   'job_events': ['logical_job_name'],
                   'task_usage': ['start_time', 'end_time']}
    elif type_X == 'jobid_Machine_StartTime':
        X_attrs = {'task_events': ['machine_ID', 'job_ID'],
                   'task_usage': ['start_time', 'end_time']}
    elif type_X == 'logicalJobname_Machine':
        X_attrs = {'task_events': ['machine_ID'],
                   'job_events': ['logical_job_name']}
    elif type_X == 'logicalJobname_StartTime':
        X_attrs = {'job_events': ['logical_job_name'],
                   'task_usage': ['start_time', 'end_time']}
    elif type_X == 'logicalJobname_only':
        X_attrs = {'job_events': ['logical_job_name']}
    elif type_X == 'iglesias':
        X_attrs = {'task_events':
                   ['priority', 'CPU_request',
                    'memory_request',
                    'different_machines_restriction'],
                   'job_events':
                   ['JE_scheduling_class', 'JE_User', 'job_name',
                    'logical_job_name'],
                   'task_usage':
                   ['start_time',
                    'end_time', 'job_ID',
                    'task_index']}
    else:
        raise NotImplementedError('%s not known' % type_X)
    t_attrs = {'task_usage': [type_T]}
    return X_attrs, t_attrs


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', dest='UseSql', action='store_false', default=True,
                    help="Use csv files instead of sql-database")
    ap.add_argument('--reloaddata', dest='reuse_existing_data',
                    action='store_true', default=False,
                    help="Reuse data stored in pickles folder")
    ap.add_argument('--removeZeroValues', dest='remove_zero_values',
                    action='store_true', default=False,
                    help="Remove rows for which the prediction is zero.")
    ap.add_argument('--shuffle', dest='shuffle',
                    action='store_true', default=False,
                    help="Shuffle the data when making folds")
    ap.add_argument('--verbose', dest='verbose',
                    action='store_true', default=False,
                    help="Print extra info")
    ap.add_argument('--rf_nobootstrap', dest='rf_bootstrap',
                    action='store_false', default=True,
                    help="Random forest: don't bootstrap")

    ap.add_argument('-alg', dest='alg', default='random_forest',
                    help='The used machine learning algorithm.'
                    + 'Defaults to random_forest')
    ap.add_argument('-attr', dest='attributes_to_use', default='iglesias',
                    help='Default: "iglesias"')
    ap.add_argument('-movingaverageWindowsize',
                    dest='moving_average_windowsize', default=None,
                    help="Add column to X, with sliding windowsize"
                    + "(default None)")
    ap.add_argument('-missingvalues',
                    dest='missing_values_strategy', default=None,
                    help="Strategy for handling missing values. "
                    + "'mean', 'median' or 'most_frequent'. "
                    + "(default 'most_frequent')")
    ap.add_argument('-n', dest='n', type=int, default=100000,
                    help="Number of total rows (train and test)")
    ap.add_argument('-nfolds', dest='nfolds', type=int, default=10,
                    help="Number of folds for crossvalidation. Default 10.")
    ap.add_argument('-predict', dest='predict', default='maximum_CPU_usage',
                    help='predict "maximum_CPU_usage"/"maximum_memory_usage"')
    ap.add_argument('-avgcolumn', dest='average_column', default=None,
                    help='Add a column to X, averaging over this column')
    ap.add_argument('-identifier', dest='identifier', default='',
                    help='This string will be added to the results_folder')
    ap.add_argument('-rf_estimators', dest='rf_estimators', default=100,
                    help='Random forest: number of used trees')

    opts = vars(ap.parse_args())

    X_attrs, t_attrs = _attributes(opts['attributes_to_use'], opts['predict'])

    if opts['UseSql'] and not opts['reuse_existing_data']:
        X, t, colnames = prepD.queryData(X_attrs, t_attrs, opts['n'],
                                         attributesSqlToPaper())
    else:
        X, t, colnames = None, None, None
        X_attrs = {key: ['%s' % (attributesSqlToPaper()[key][item])
                         for item in vals]
                   for key, vals in X_attrs.iteritems()}

    prefix = '%s_%s_%i_' % (opts['predict'], opts['attributes_to_use'],
                            opts['n'])
    prefix = prefix if not opts['UseSql'] else prefix + 'sql_'
    folder_results = 'results/%s%s_%s/' % (prefix, opts['alg'],
                                           opts['identifier'])
    if not os.path.exists(folder_results):
            os.makedirs(folder_results)

    # Redirecting output.
    orig_stdout = sys.stdout
    f = file(folder_results + 'out.log', 'w')
    sys.stdout = f

    options_pickling = {'folder': 'pickles',
                        'tables': False,
                        'Xt': opts['reuse_existing_data'],
                        'X_name': prefix + 'X',
                        't_name': prefix + 't',
                        'columns_name': prefix + 'col'}
    folder_data = 'data/'
    options_preprocess = {'missing_values': opts['missing_values_strategy'],
                          'scale': False,
                          'PCA': False, 'remove_killed_tasks': True,
                          'remove_zero_values': opts['remove_zero_values'],
                          'remove_task_usage': True,
                          'remove_values_higher_than': 1.,
                          'remove_training_higher_than': None,
                          'increase_zero_ram_values': True,
                          'increase_zero_cpu_values': True,
                          'moving_average':
                              opts['moving_average_windowsize'] is not None,
                          'moving_average_windowsize':
                              opts['moving_average_windowsize'],
                          'columns_to_transform':
                          ['different-machine_constraint', 'user_name',
                           'job_name', 'job_name_logical', 'logical_job_name',
                           'User',
                           'task_events_different-machine_constraint',
                           'job_events_user_name',
                           'job_events_job_name',
                           'job_events_job_name_logical',
                           'JE_scheduling_class', 'JE_User',
                           'task_events_machine_ID'
                           ],
                          'average_column': opts['average_column'],
                          'average_column_groupBy': 'job_name_logical'
                          }

    options_machinelearning = _options_machinelearning(opts['alg'],
                                                       opts['predict'],
                                                       opts['shuffle'],
                                                       folder_results,
                                                       opts['rf_bootstrap'],
                                                       opts['rf_estimators'])
    pipeline.runPipeline(opts['n'], opts['nfolds'], X_attrs, t_attrs,
                         options_preprocess, options_machinelearning,
                         options_pickling,
                         folder_data, verbose=opts['verbose'],
                         X=X, t=t, colnames=colnames)
    sys.stdout = orig_stdout
    f.close()
