'''
Created on 3 nov. 2015

@author: Faisal
'''
from sets import Set

import numpy as np


def to_tuple__name_table_attributes(tuple__table_name, _dict):
    tuple__name_table_attibutes = [(name,
                                    [t for (t, n) in tuple__table_name
                                        if n == name][0],
                                    _dict[name])
                                   for name in _dict.keys()]
    return tuple__name_table_attibutes


def toMatrix(tuple__name_table_attibutes, n=1000,
             keys=None):
    task_dict, column_names = toDict(tuple__name_table_attibutes, n=n,
                                     keys=keys)
    keys = task_dict.keys() if keys is None else keys
    task_matrix = np.matrix([task_dict[key] for key in keys])
    return task_matrix, keys, column_names


def toDict(tuple__name_table_attibutes, n=1000, keys=None):
    '''
    Output: dict with
        - key: task_index
        - value: list:
            [usage_memory, usage_cpu, usage_disk, ...]
    '''
    column_names = []
    if keys is None:
        for (table_name, table, _) in tuple__name_table_attibutes:
            if table_name == "task_events":
                task_dict, job_dict = _setup_dictinaries(table, n)
    else:
        task_dict = dict.fromkeys(keys, [])
        job_dict = {}
        for (job_id, task_index) in keys:
            if job_id not in job_dict:
                job_dict[job_id] = [task_index]
            else:
                job_dict[job_id] = job_dict[job_id] + [task_index]
    for (table_name, table, attributes) in tuple__name_table_attibutes:
        column_names += ['%s_%s' % (table_name, a) for a in attributes]
        if table_name == "job_events":
            task_dict = _jobEventsToDict(table, attributes, task_dict,
                                         job_dict)
        else:
            task_dict = _tableToDict(table, attributes, task_dict, job_dict)
    return task_dict, column_names


def _setup_dictinaries(table, n):
    '''
    Output:
        - task_dict.
            key: task_index.
            Value: list of information like event_type
        - job_dict.
            key: job_id.
            Value: list of tasks.
    '''
    job_dict = {}
    task_dict = {}
    i = 0
    size = 0
    while(size < n):
        if not np.isnan(table.resource_request_for_CPU_cores[i]):
            job_id = table.job_ID[i]
            task_index = table.task_index[i]
            if job_id not in job_dict.keys():
                job_dict[job_id] = [task_index]
            else:
                job_dict[job_id] = job_dict[job_id] + [task_index]
            task_dict[(job_id, task_index)] = []
            size += 1
        i += 1
    return task_dict, job_dict


def _tableToDict(table, attributes, task_dict, job_dict):
    for job_id, task_ids in job_dict.iteritems():  # TODO: improve efficiency
        rows_job = table[(table['job_ID'] == job_id)]
        for task_id in task_ids:
            row_table = rows_job[(rows_job['task_index'] == task_id)]
            row_dict = []
            for attr in attributes:
                if row_table[attr].values.shape[0] > 0:
                    row_dict.append(row_table[attr].values[0])
                else:
                    row_dict.append(-1)
            ky = (job_id, task_id)
            task_dict[ky] = task_dict[ky] + row_dict
    return task_dict


def _jobEventsToDict(jobEvents, attributes, task_dict, job_dict):
    job_ids = list(Set([j for (j, _) in task_dict.keys()]))
    for job_id in job_ids:
        row = jobEvents[(jobEvents['job_ID'] == job_id)]

        row_dict = []
        for attr in attributes:
            row_dict.append(row[attr].values[0])
        for task_index in job_dict[job_id]:
            ky = (job_id, task_index)
            task_dict[ky] = task_dict[ky] + row_dict
    return task_dict
