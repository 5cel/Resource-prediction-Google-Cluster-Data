'''
Created on 4 nov. 2015

@author: Faisal, Jos
'''
import unittest
import main.filehandling.readCsv as readCsv
import main.preprocessing.featureConstruction as featureConstructor


class Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        table_names = ['task_usage', 'task_events', 'job_events']
        filenames = ['../../data/' + f + '/' for f in table_names]
        tables = [readCsv.readFolder(f) for f in filenames]
        cls.table_taskUsage = tables[0]
        cls.table_taskEvents = tables[1]
        cls.tuple__table_name = [(tables[i], table_names[i])
                                 for i in range(len(table_names))]
        print('\tCsv files read...')

    def testEstimationCPUError(self):
        n = 25
        mat, _ = featureConstructor.estimationCPUError(self.table_taskUsage,
                                                       self.table_taskEvents,
                                                       n)
        assert(mat.shape[0] == n)
        assert(mat.shape[1] == 3)

    def testCpuError_Xt(self):
        n = 100
        X, t = featureConstructor.cpuError_Xt(self.tuple__table_name, n)
        assert(X.shape == (n, 9))
        assert(t.shape == (n, 1))

if __name__ == "__main__":
    unittest.main()
