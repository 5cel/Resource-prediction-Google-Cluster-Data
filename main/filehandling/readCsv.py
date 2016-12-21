'''
Created on 3 nov. 2015

@author: Faisal
'''

import csv
import os

import pandas


def readCsvFile(filename, headers):
    data = pandas.read_csv(filename, sep=',', names=headers, index_col=False)
    return data


def readFolder(folder):
    files = os.listdir(folder)
    file_header = [folder + f for f in files if f.endswith('headers.csv')][0]
    files_csv = [folder + f for f in files
                 if not f.endswith('headers.csv')and f.endswith('.csv')]

    with open(file_header, 'r') as f:
        headers = list(csv.reader(f, delimiter=','))[0]

    for f in files_csv:
        if 'data' not in locals():
            data = readCsvFile(f, headers)
        else:
            data = pandas.concat[data, readCsvFile(f, headers)]
    return data
