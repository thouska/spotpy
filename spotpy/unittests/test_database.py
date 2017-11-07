import unittest
import unittest.mock as mock
import spotpy.database as db
import numpy as np

#https://docs.python.org/3/library/unittest.html

class TestSignatures(unittest.TestCase):

    def setUp(self):
        self.parnames = [b'x1',b'x2',b'x3',b'x4',b'x5']
        self.like = 0.213
        self.randompar = [175.21733934706367, 0.41669126598819262, 0.25265012080652388, 0.049706767415682945, 0.69674090782836173]

        self.simulations_multi = []
        for i in range(5):
            self.simulations_multi.append(np.random.uniform(0, 1, 5).tolist())

        self.simulations = np.random.uniform(0, 1, 5).tolist()

        #print(self.simulations)

    def objf(self):
        return np.random.uniform(0,1,1)[0]


    def test_csv_multiline(self):
        csv = db.csv("UnitTest_tmp", self.parnames, self.like, self.randompar, simulations=self.simulations_multi,
                          chains=1, save_sim=True)

        csv.save(self.like ,self.randompar,self.simulations_multi)
        # Save Simulations

        self.assertEqual(len(csv.header),32)
        csv.finalize()


    def test_csv_multiline_false(self):
        # Save not Simulations
        csv = db.csv("UnitTest_tmp", self.parnames, self.like, self.randompar, simulations=self.simulations_multi,
                     chains=1, save_sim=False)

        csv.save(self.like, self.randompar, self.simulations_multi)
        self.assertEqual(len(csv.header), 7)
        csv.finalize()

    def test_csv_single(self):
        csv = db.csv("UnitTest_tmp", self.parnames, self.like, self.randompar, simulations=self.simulations,
                          chains=1, save_sim=True)

        csv.save(self.like, self.randompar, self.simulations)
        self.assertEqual(len(csv.header), 12)
        csv.finalize()

    def test_csv_single_false(self):
        csv = db.csv("UnitTest_tmp", self.parnames, self.like, self.randompar, simulations=self.simulations,
                     chains=1, save_sim=False)

        csv.save(self.like, self.randompar, self.simulations)
        self.assertEqual(len(csv.header), 7)
        csv.finalize()

    def test_sql_multiline(self):
        sql = db.sql("UnitTest_tmp",self.parnames, self.like, self.randompar, simulations=self.simulations_multi,
                          chains=1, save_sim=True)
        sql.save(self.like, self.randompar, self.simulations_multi)
        sql.finalize()
        self.assertEqual(len(sql.header), 32)


    def test_sql_multiline_false(self):
        sql = db.sql("UnitTest_tmp",self.parnames, self.like, self.randompar, simulations=self.simulations_multi,
                          chains=1, save_sim=False)
        sql.save(self.like, self.randompar, self.simulations_multi)
        sql.finalize()
        self.assertEqual(len(sql.header), 7)

    def test_sql_single(self):
        sql = db.sql("UnitTest_tmp",self.parnames, self.like, self.randompar, simulations=self.simulations,
                          chains=1, save_sim=True)
        sql.save(self.like, self.randompar, self.simulations)
        sql.finalize()
        self.assertEqual(len(sql.header), 12)

    def test_sql_single_false(self):
        sql = db.sql("UnitTest_tmp",self.parnames, self.like, self.randompar, simulations=self.simulations,
                          chains=1, save_sim=False)
        sql.save(self.like, self.randompar, self.simulations)
        sql.finalize()
        self.assertEqual(len(sql.header), 7)

    def test_ram_multiline(self):
        ram = db.ram("UnitTest_tmp", self.parnames, self.like, self.randompar, simulations=self.simulations_multi,
                     chains=1, save_sim=True)
        ram.save(self.like, self.randompar, self.simulations_multi)
        # self.csv.finalize()
        self.assertEqual(len(ram.header), 32)

    def test_ram_multiline_false(self):
        ram = db.ram("UnitTest_tmp", self.parnames, self.like, self.randompar, simulations=self.simulations_multi,
                     chains=1, save_sim=False)
        ram.save(self.like, self.randompar, self.simulations_multi)
        # self.csv.finalize()
        self.assertEqual(len(ram.header), 7)

    def test_ram_single(self):
        ram = db.sql("UnitTest_tmp", self.parnames, self.like, self.randompar, simulations=self.simulations,
                     chains=1, save_sim=True)
        ram.save(self.like, self.randompar, self.simulations)
        # self.csv.finalize()
        self.assertEqual(len(ram.header), 12)

    def test_ram_single_false(self):
        ram = db.sql("UnitTest_tmp", self.parnames, self.like, self.randompar, simulations=self.simulations,
                     chains=1, save_sim=False)
        ram.save(self.like, self.randompar, self.simulations)
        # self.csv.finalize()
        self.assertEqual(len(ram.header), 7)




if __name__ == '__main__':
    unittest.main()