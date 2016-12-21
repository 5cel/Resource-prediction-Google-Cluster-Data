'''
Created on 1 Dec 2015

@author: Faisal
'''

import time

from sqlalchemy import create_engine

import pandas as pd


def queryData(X_attrs, t_attrs, n, _dict):
    t_col = ('maximum_CPU_rate' if 'CPU' in t_attrs.values()[0][0]
             else 'maximum_memory_usage')
    cols = [['%s_%s' % (keys, _dict[keys][item]), '%s AS %s' % (item, item)]
            for keys, vals in X_attrs.iteritems()
            for item in vals]
    colnames = [c for c, _ in cols]
    colList = [c for _, c in cols] + ['%s AS %s'
                                      % (t_col, t_col)]
    if 'different_machines_restriction' in colList:
        idx = colList.index('different_machines_restriction')
        colList[idx] = 'different_machines_restriction+0'
    limit = '' if n == 0 else ' LIMIT %i ' % n

    with open('test/preparedata/SQLScript_python', 'r') as myfile:
        query = myfile.read().replace('\n', '') % (', '.join(colList), limit)
    engine = create_engine('mysql://root:csc2233@localhost/googlecluster')
    with engine.connect() as conn, conn.begin():
        t0 = time.time()
        dataDF = pd.read_sql_query(query, conn)
        t1 = time.time()
        print 'Loading the data took %f sec' % (t1 - t0)
        narray = dataDF.as_matrix()
        X = narray[:, :-1]
        t = narray[:, -1]

    conn.close()
    engine.dispose()
    return X, t, colnames
