import unittest
import spotpy.database as db
import numpy as np

#https://docs.python.org/3/library/unittest.html

class TestSignatures(unittest.TestCase):

    def setUp(self):
        pass
        self.parnames = [b'x1',b'x2',b'x3',b'x4',b'x5']
        self.like = 0.213
        self.randompar = [175.21733934706367, 0.41669126598819262, 0.25265012080652388, 0.049706767415682945, 0.69674090782836173]




        #print(self.simulations)

    def objf(self):
        return np.random.uniform(0,1,1)[0]

    def test_csv_multiline(self):
        self.simulations = []
        for i in range(5):
            self.simulations.append(np.random.uniform(0,1,5).tolist())
        self.csv = db.csv("UnitTest_tmp", self.parnames, self.like, self.randompar, simulations=self.simulations,
                          chains=1, save_sim=True )

        self.csv.save(self.like ,self.randompar,self.simulations)
        #self.csv.finalize()
        self.assertEqual(len(self.csv.header),32)


    def test_csv_single(self):
        self.simulations = np.random.uniform(0, 1, 5).tolist()

        self.csv = db.csv("UnitTest_tmp", self.parnames, self.like, self.randompar, simulations=self.simulations,
                          chains=1, save_sim=True)

        self.csv.save(self.like, self.randompar, self.simulations)
        self.assertEqual(len(self.csv.header), 12)


if __name__ == '__main__':
    unittest.main()