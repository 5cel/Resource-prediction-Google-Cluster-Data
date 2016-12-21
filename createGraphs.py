'''
Created on 12 nov. 2015

@author: Faisal
'''
import argparse
import os
import sys

import tabulate

import main.pipeline as pipeline
import main.visualization.results as vis
import run
import test.preparedata.prepareData as prepD


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-predict', dest='predict', default='maximum_CPU_usage',
                    help='predict "maximum_CPU_usage"/"maximum_memory_usage"')
    ap.add_argument('--csv', dest='UseSql', action='store_false', default=True,
                    help="Use csv files instead of sql-database")
    ap.add_argument('--reloaddata', dest='reuse_existing_data',
                    action='store_true', default=False,
                    help="Reuse data stored in pickles folder")
    ap.add_argument('-missingvalues',
                    dest='missing_values_strategy', default=None,
                    help="Strategy for handling missing values. "
                    + "'mean', 'median' or 'most_frequent'. "
                    + "(default 'most_frequent')")
    ap.add_argument('-movingaverageWindowsize',
                    dest='moving_average_windowsize', default=None,
                    help="Add column to X, with sliding windowsize"
                    + "(default None)")
    ap.add_argument('-avgcolumn', dest='average_column', default=None,
                    help='Add a column to X, averaging over this column')
    ap.add_argument('-avgcolumn_groupby', dest='average_column_group_by',
                    default='logical',
                    help='Add a averaging column to X, grouped by this column')
    ap.add_argument('--delete_groupby_column', dest='delete_groupby_column',
                    default=False, action="store_true",
                    help='Delete the column groupby column before learning')
    ap.add_argument('--removeZeroValues', dest='remove_zero_values',
                    action='store_true', default=False,
                    help="Remove rows for which the prediction is zero.")
    ap.add_argument('--keep_task_usage', dest='remove_task_usage',
                    action='store_false', default=True,
                    help="Do not remove columns from task_usage table")
    ap.add_argument('--keep_start_time', dest='keep_start_time',
                    action='store_true', default=False,
                    help="Do not remove the start time")
    ap.add_argument('-identifier', dest='identifier', default='',
                    help='This string will be added to the results_folder')
    ap.add_argument('--noshuffle', dest='shuffle',
                    action='store_false', default=True,
                    help="Do not shuffle the data when making folds")
    ap.add_argument('-rf_estimators', dest='rf_estimators', default=100,
                    help='Random forest: number of used trees')
    ap.add_argument('--rf_nobootstrap', dest='rf_bootstrap',
                    action='store_false', default=True,
                    help="Random forest: don't bootstrap")
    ap.add_argument('--daily_representation', dest='daily_representation',
                    action='store_true', default=False,
                    help="Use starttime modulo the time in a day")
    ap.add_argument('--large_dataset', dest='large_dataset',
                    action='store_true', default=False,
                    help="Use n=[100000, 1000000, 10000000, 50000000] instead "
                    + "of n=[100000, 200000, 400000, 600000, 800000, 1000000]")
    opts = vars(ap.parse_args())
    useSql = opts['UseSql']
    reuse_existing_data = opts['reuse_existing_data']

    verbose = True
    predict = opts['predict']
    ns = {True: [100000, 1000000, 10000000, 50000000],
          False: [100000, 200000, 400000, 600000, 800000, 1000000]
          }[opts['large_dataset']]

    attributes_to_use = ['logicalJobname_Machine_StartTime']
    algorithms = ['random_forest'] * len(attributes_to_use)
    names = [attr + '_' + opts['identifier'] for attr in attributes_to_use]

    nfolds = 10

    MAE_avgs = {n: [] for n in names}
    MAE_stds = {n: [] for n in names}
    RMSE_avgs = {n: [] for n in names}
    RMSE_stds = {n: [] for n in names}
    for n in ns:
        for i, alg in enumerate(algorithms):
            X_attrs, t_attrs = run._attributes(attributes_to_use[i], predict)
            if useSql:
                X, t, colnames = prepD.queryData(X_attrs, t_attrs, n,
                                                 run.attributesSqlToPaper())
            else:
                X, t, colnames = None, None, None
                X_attrs = {key: ['%s' % (run.attributesSqlToPaper()[key][item])
                                 for item in vals]
                           for key, vals in X_attrs.iteritems()}
            prefix = '%s_%s_%i_' % (predict, attributes_to_use[i], n)

            folder_results = 'results/%s%s_%s/' % (prefix, alg,
                                                   names[i])
            if not os.path.exists(folder_results):
                    os.makedirs(folder_results)

            # Redirecting output.
            orig_stdout = sys.stdout
            f = file(folder_results + 'out.log', 'w')
            sys.stdout = f

            options_pickling = {'folder': 'pickles',
                                'Xt': reuse_existing_data,
                                'X_name': prefix + 'X',
                                't_name': prefix + 't',
                                'columns_name': prefix + 'col'
                                }
            folder_data = 'data/'

            average_column = 't' if alg == 'baseline_average_overJobid' else opts['average_column']
            if attributes_to_use[i] in ['logicalJobname_Machine',
                                        'logicalJobname_Machine_StartTime',
                                        'baseline_average_overJobid']:
                average_column_group_by = 'logical'
            elif attributes_to_use[i] == 'jobid_Machine_StartTime':
                average_column_group_by = 'job_ID'
            else:
                average_column_group_by = opts['average_column_group_by']

            options_preprocess = {'missing_values': opts['missing_values_strategy'],
                                  'scale': False,
                                  'PCA': False, 'remove_killed_tasks': True,
                                  'remove_zero_values': opts['remove_zero_values'],
                                  'remove_task_usage': opts['remove_task_usage'],
                                  'keep_start_time': opts['keep_start_time'],
                                  'remove_values_higher_than': 1.,
                                  'remove_training_higher_than': None,
                                  'increase_zero_ram_values': True,
                                  'increase_zero_cpu_values': True,
                                  'moving_average':
                                      opts['moving_average_windowsize'] is not None,
                                  'moving_average_windowsize':
                                      opts['moving_average_windowsize'],
                                  'daily_representation': opts['daily_representation'],
                                  'columns_to_transform':
                                  ['different-machine_constraint', 'user_name',
                                   'job_name', 'job_name_logical', 'logical_job_name',
                                   'User',
                                   'task_events_different-machine_constraint',
                                   'job_events_user_name',
                                   'job_events_job_name',
                                   'job_events_job_name_logical',
                                   'JE_scheduling_class', 'JE_User',
                                   'task_events_machine_ID', 'machine_ID'
                                   ],
                                  'average_column': average_column,
                                  'average_column_groupBy':
                                      average_column_group_by,
                                      'delete_groupby_column':
                                      opts['delete_groupby_column']
                                  }

            shuffle, rf_bootstrap, rf_estimators \
                = opts['shuffle'], opts['rf_bootstrap'], opts['rf_estimators']

            options_machinelearning \
                = run._options_machinelearning(alg, predict, shuffle,
                                               folder_results, rf_bootstrap,
                                               rf_estimators)
            MAE_average, MAE_std, RMSE_average, RMSE_std \
                = pipeline.runPipeline(n, nfolds, X_attrs, t_attrs,
                                       options_preprocess,
                                       options_machinelearning,
                                       options_pickling,
                                       folder_data, verbose=verbose,
                                       X=X, t=t, colnames=colnames)
            MAE_avgs[names[i]] += [MAE_average]
            MAE_stds[names[i]] += [MAE_std]
            RMSE_avgs[names[i]] += [RMSE_average]
            RMSE_stds[names[i]] += [RMSE_std]

            sys.stdout = orig_stdout
            f.close()

    folder_totals = 'results/%s/' % opts['identifier']
    if not os.path.exists(folder_totals):
        os.makedirs(folder_totals)

    orig_stdout = sys.stdout
    f = file('%scompleteTables.log' % folder_totals, 'w')
    sys.stdout = f

    _toprint = lambda _dict: [[key] + _dict[key]
                              for i, key in enumerate(_dict.keys())
                              ]
    print tabulate.tabulate(_toprint(MAE_avgs), headers=['MAE avgs']+ns) + '\n'
    print tabulate.tabulate(_toprint(MAE_stds), headers=['MAE stds']+ns) + '\n'

    print tabulate.tabulate(_toprint(RMSE_avgs),
                            headers=['RMSE avgs']+ns) + '\n'
    print tabulate.tabulate(_toprint(RMSE_stds),
                            headers=['RMSE stds']+ns) + '\n'

    vis.error_graph(names, MAE_avgs, MAE_stds, ns, "total n", "MAE",
                    "%sMAE" % folder_totals)
    vis.error_graph(names, RMSE_avgs, RMSE_stds, ns, "total n", "RMSE",
                    "%sRMSE" % folder_totals)
    sys.stdout.flush()
    sys.stdout = orig_stdout
    f.close()
