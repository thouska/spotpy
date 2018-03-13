import unittest
import os
import glob

try:
    import spotpy
except ImportError:
    import sys
    sys.path.append("../../")
    sys.path.append("../../spotpy")
    sys.path.append(".")
    import spotpy
import spotpy.database as db
import numpy as np

#https://docs.python.org/3/library/unittest.html

class TestDatabase(unittest.TestCase):

    def setUp(self):
        self.parnames = ['x1', 'x2', 'x3', 'x4', 'x5']
        self.like = 0.213
        self.randompar = [175.21733934706367, 0.41669126598819262, 0.25265012080652388, 0.049706767415682945, 0.69674090782836173]

        self.simulations_multi = []
        for i in range(5):
            self.simulations_multi.append(np.random.uniform(0, 1, 5).tolist())

        self.simulations = np.random.uniform(0, 1, 5).tolist()

        #print(self.simulations)

    def tearDown(self):
        for filename in glob.glob("UnitTest_tmp*"):
            os.remove(filename)

    def objf(self):
        return np.random.uniform(0, 1, 1)[0]

    def test_csv_multiline(self):
        csv = db.csv("UnitTest_tmp", self.parnames, self.like, self.randompar, simulations=self.simulations_multi, chains=1, save_sim=True)

        csv.save(self.like, self.randompar, self.simulations_multi)
        csv.save(self.like, self.randompar, self.simulations_multi)
        # Save Simulations

        csv.finalize()
        csvdata = csv.getdata()
        self.assertEqual(str(type(csvdata)), str(type(np.array([]))))
        self.assertEqual(len(csvdata[0]), 32)
        self.assertEqual(len(csvdata), 2)
        self.assertEqual(len(csv.header), 32)


    def test_csv_multiline_false(self):
        # Save not Simulations
        csv = db.csv("UnitTest_tmp", self.parnames, self.like, self.randompar, simulations=self.simulations_multi, chains=1, save_sim=False)

        csv.save(self.like, self.randompar, self.simulations_multi)
        csv.save(self.like, self.randompar, self.simulations_multi)

        csv.finalize()
        csvdata = csv.getdata()
        self.assertEqual(str(type(csvdata)), str(type(np.array([]))))
        self.assertEqual(len(csvdata[0]), 7)
        self.assertEqual(len(csvdata), 2)
        self.assertEqual(len(csv.header), 7)

    def test_csv_single(self):
        csv = db.csv("UnitTest_tmp", self.parnames, self.like, self.randompar, simulations=self.simulations,
                     chains=1, save_sim=True)

        csv.save(self.like, self.randompar, self.simulations)
        csv.save(self.like, self.randompar, self.simulations)

        csv.finalize()
        csvdata = csv.getdata()
        self.assertEqual(str(type(csvdata)), str(type(np.array([]))))
        self.assertEqual(len(csvdata[0]), 12)
        self.assertEqual(len(csvdata), 2)
        self.assertEqual(len(csv.header), 12)

    def test_csv_single_false(self):
        csv = db.csv("UnitTest_tmp", self.parnames, self.like, self.randompar, simulations=self.simulations,
                     chains=1, save_sim=False)

        csv.save(self.like, self.randompar, self.simulations)
        csv.save(self.like, self.randompar, self.simulations)

        csv.finalize()
        csvdata = csv.getdata()
        self.assertEqual(str(type(csvdata)), str(type(np.array([]))))
        self.assertEqual(len(csvdata[0]), 7)
        self.assertEqual(len(csvdata), 2)
        self.assertEqual(len(csv.header), 7)


    def test_sql_multiline(self):
        sql = db.sql("UnitTest_tmp", self.parnames, self.like, self.randompar, simulations=self.simulations_multi, chains=1, save_sim=True)
        sql.save(self.like, self.randompar, self.simulations_multi)
        sql.finalize()
        sqldata = sql.getdata()
        self.assertEqual(str(type(sqldata)), str(type(np.array([]))))
        self.assertEqual(len(sqldata[0]), 32)
        self.assertEqual(len(sqldata), 1)
        self.assertEqual(len(sql.header), 32)


    def test_sql_multiline_false(self):
        sql = db.sql("UnitTest_tmp", self.parnames, self.like, self.randompar, simulations=self.simulations_multi, chains=1, save_sim=False)
        sql.save(self.like, self.randompar, self.simulations_multi)
        sql.finalize()
        sqldata = sql.getdata()
        self.assertEqual(str(type(sqldata)), str(type(np.array([]))))
        self.assertEqual(len(sqldata[0]), 7)
        self.assertEqual(len(sqldata), 1)
        self.assertEqual(len(sql.header), 7)

    def test_sql_single(self):
        sql = db.sql("UnitTest_tmp", self.parnames, self.like, self.randompar,
                     simulations=self.simulations, chains=1, save_sim=True)
        sql.save(self.like, self.randompar, self.simulations)
        sql.finalize()
        sqldata = sql.getdata()
        self.assertEqual(str(type(sqldata)), str(type(np.array([]))))
        self.assertEqual(len(sqldata[0]), 12)
        self.assertEqual(len(sqldata), 1)
        self.assertEqual(len(sql.header), 12)

    def test_sql_single_false(self):
        sql = db.sql("UnitTest_tmp", self.parnames, self.like, self.randompar, simulations=self.simulations,
                     chains=1, save_sim=False)
        sql.save(self.like, self.randompar, self.simulations)
        sql.finalize()

        sqldata = sql.getdata()
        self.assertEqual(str(type(sqldata)), str(type(np.array([]))))
        self.assertEqual(len(sqldata[0]), 7)
        self.assertEqual(len(sqldata), 1)
        self.assertEqual(len(sql.header), 7)

    def test_ram_multiline(self):
        ram = db.ram("UnitTest_tmp", self.parnames, self.like, self.randompar, simulations=self.simulations_multi, chains=1, save_sim=True)
        ram.save(self.like, self.randompar, self.simulations_multi)
        ram.finalize()

        ramdata = ram.getdata()
        self.assertEqual(str(type(ramdata)), str(type(np.array([]))))
        self.assertEqual(len(ram.header), 32)
        self.assertEqual(len(ramdata[0]), 32)
        self.assertEqual(len(ramdata), 1)
        self.assertEqual(len(ramdata.dtype), len(ram.header))

    def test_ram_multiline_false(self):
        ram = db.ram("UnitTest_tmp", self.parnames, self.like, self.randompar, simulations=self.simulations_multi, chains=1, save_sim=False)
        ram.save(self.like, self.randompar, self.simulations_multi)

        ram.finalize()
        ramdata = ram.getdata()
        self.assertEqual(str(type(ramdata)), str(type(np.array([]))))
        self.assertEqual(len(ramdata[0]), 7)
        self.assertEqual(len(ramdata), 1)
        self.assertEqual(len(ramdata.dtype), len(ram.header))
        self.assertEqual(len(ram.header), 7)

    def test_ram_single(self):
        ram = db.ram("UnitTest_tmp", self.parnames, self.like, self.randompar, simulations=self.simulations,
                     chains=1, save_sim=True)
        ram.save(self.like, self.randompar, self.simulations)

        ram.finalize()
        ramdata = ram.getdata()
        self.assertEqual(str(type(ramdata)), str(type(np.array([]))))
        self.assertEqual(len(ramdata[0]), 12)
        self.assertEqual(len(ramdata), 1)
        self.assertEqual(len(ramdata.dtype), len(ram.header))
        self.assertEqual(len(ram.header), 12)

    def test_ram_single_false(self):
        ram = db.ram("UnitTest_tmp", self.parnames, self.like, self.randompar, simulations=self.simulations,
                     chains=1, save_sim=False)
        ram.save(self.like, self.randompar, self.simulations)

        ram.finalize()
        ramdata = ram.getdata()
        self.assertEqual(str(type(ramdata)), str(type(np.array([]))))
        self.assertEqual(len(ramdata[0]), 7)
        self.assertEqual(len(ramdata), 1)
        self.assertEqual(len(ramdata.dtype), len(ram.header))
        self.assertEqual(len(ram.header), 7)


if __name__ == '__main__':
    unittest.main()
