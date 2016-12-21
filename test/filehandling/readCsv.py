'''
Created on 3 nov. 2015

@author: Faisal, Jos
'''
import unittest
import main.filehandling.readCsv as readCsv


class Test(unittest.TestCase):

    def testReadCsvFile(self):
        filename = '../../data/job_events/part-00000-of-00500.csv'
        headers = ['A', 'B', 'job_ID', 'D', 'E', 'F', 'G', 'H']
        data = readCsv.readCsvFile(filename, headers, False)
        assert data.job_ID[0] == 3418309
        assert data.job_ID[1] == 3418314
        assert data.job_ID[2] == 3418319

    def testReadFolder(self):
        folder = '../../data/job_events/'
        data = readCsv.readFolder(folder)
        headers = ['timestamp', 'missing_info', 'job_ID', 'event_type',
                   'user_name', 'scheduling_class', 'job_name',
                   'job_name_logical']
        assert(list(data.columns.values) == headers)
        assert data.job_ID[0] == 3418309
        assert data.job_ID[1] == 3418314
        assert data.job_ID[2] == 3418319

if __name__ == "__main__":
    unittest.main()
