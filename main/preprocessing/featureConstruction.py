'''
Created on 3 nov. 2015

@author: Faisal, Jos
'''
import main.filehandling.csvdataToMatrices as csvToMat
import numpy as np
import pandas as pd


def estimationCPUError(table_taskUsage, table_taskEvents, n=100,
                       task_usage_attr='maximum_memory_usage',
                       task_events_attr='resource_request_for_RAM'):
    tuple__name_table_attibutes = [('task_usage', table_taskUsage,
                                    [task_usage_attr]),
                                   ('task_events', table_taskEvents,
                                    [task_events_attr])
                                   ]
    mat, _ = csvToMat.toMatrix(tuple__name_table_attibutes, n=n)
    difference = np.diff(mat, axis=1)
    mat = np.concatenate((mat, difference), axis=1)
    return mat, pd.DataFrame(mat, columns=[task_usage_attr,
                                           task_events_attr,
                                           "Difference"])


def Xt(tuple__table_name, X_attrs, t_attrs, n=1000):
    tuple__name_table_attributes_X \
        = csvToMat.to_tuple__name_table_attributes(tuple__table_name, X_attrs)
    tuple__name_table_attributes_t \
        = csvToMat.to_tuple__name_table_attributes(tuple__table_name, t_attrs)
    X, keys, column_names \
        = csvToMat.toMatrix(tuple__name_table_attributes_X, n)
    t, _, _ = csvToMat.toMatrix(tuple__name_table_attributes_t, n, keys)

    indexes_to_remove = []
    for i, v in enumerate(t):
        if v == -1:
            indexes_to_remove.append(i)
    X = np.delete(X, indexes_to_remove, 0)
    t = np.delete(t, indexes_to_remove, 0)
    print("%i/%i tasks not found in usage table"
          % (len(indexes_to_remove), len(t))),

    t = np.ravel(t)
    return X, t, column_names
