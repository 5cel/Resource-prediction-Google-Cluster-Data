'''
Created on 3 nov. 2015

@author: Faisal, Jos
'''
from sets import Set
import unittest

import main.filehandling.csvdataToMatrices as csvToMat
import main.filehandling.readCsv as readCsv


class Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        table_names = ['task_usage', 'task_events', 'job_events']
        filenames = ['../../data/' + f + '/' for f in table_names]
        tables = [readCsv.readFolder(f) for f in filenames]
        cls.table_name_pairs = [(tables[i], table_names[i])
                                for i in range(len(filenames))]
        print('\tCsv files read...')

    def testToDict(self):
        n = 200
        table_attribute_pairs = \
            csvToMat.table_attribute_pairs_default(self.table_name_pairs)
        task_dict = csvToMat.toDict(table_attribute_pairs, n)

        assert len(task_dict) == n
        assert Set([len(v) for v in task_dict.values()]) == Set([12])

    def testToMatrix(self):
        n = 250
        table_attribute_pairs = \
            csvToMat.table_attribute_pairs_default(self.table_name_pairs)
        task_matrix, _ = csvToMat.toMatrix(table_attribute_pairs, n)
        assert task_matrix.shape[0] == n
        assert task_matrix.shape[1] == 12

if __name__ == "__main__":
    unittest.main()
