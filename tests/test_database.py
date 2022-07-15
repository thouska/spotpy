# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 by Tobias Houska
This file is part of Statistical Parameter Optimization Tool for Python(SPOTPY).
:author: Tobias Houska
"""
import glob
import os
import unittest

import numpy as np

import spotpy.database as db

# https://docs.python.org/3/library/unittest.html


class MockSetup:
    """
    Mock class to use the save function of a spotpy setup
    """

    def save(self, *args, **kwargs):
        pass


class TestDatabase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.parnames = ["x1", "x2", "x3", "x4", "x5"]
        self.like = 0.213
        self.randompar = [
            175.21733934706367,
            0.41669126598819262,
            0.25265012080652388,
            0.049706767415682945,
            0.69674090782836173,
        ]

        self.simulations_multi = []
        for i in range(5):
            self.simulations_multi.append(np.random.uniform(0, 1, 5).tolist())

        self.simulations = np.random.uniform(0, 1, 5)

    @classmethod
    def tearDownClass(self):
        for filename in glob.glob("UnitTest_tmp*"):
            os.remove(filename)

    def objf(self):
        return np.random.uniform(0, 1, 1)[0]

    def test_csv_multiline(self):
        csv = db.get_datawriter(
            "csv",
            "UnitTest_tmp",
            self.parnames,
            self.like,
            self.randompar,
            simulations=self.simulations_multi,
            chains=1,
            save_sim=True,
        )

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
        csv = db.get_datawriter(
            "csv",
            "UnitTest_tmp",
            self.parnames,
            self.like,
            self.randompar,
            simulations=self.simulations_multi,
            chains=1,
            save_sim=False,
        )

        csv.save(self.like, self.randompar, self.simulations_multi)
        csv.save(self.like, self.randompar, self.simulations_multi)

        csv.finalize()
        csvdata = csv.getdata()
        self.assertEqual(str(type(csvdata)), str(type(np.array([]))))
        self.assertEqual(len(csvdata[0]), 7)
        self.assertEqual(len(csvdata), 2)
        self.assertEqual(len(csv.header), 7)

    def test_csv_single(self):
        csv = db.get_datawriter(
            "csv",
            "UnitTest_tmp",
            self.parnames,
            self.like,
            self.randompar,
            simulations=self.simulations,
            chains=1,
            save_sim=True,
        )

        csv.save(self.like, self.randompar, self.simulations)
        csv.save(self.like, self.randompar, self.simulations)

        csv.finalize()
        csvdata = csv.getdata()
        self.assertEqual(str(type(csvdata)), str(type(np.array([]))))
        self.assertEqual(len(csvdata[0]), 12)
        self.assertEqual(len(csvdata), 2)
        self.assertEqual(len(csv.header), 12)

    def test_csv_append(self):
        csv = db.get_datawriter(
            "csv",
            "UnitTest_tmp",
            self.parnames,
            self.like,
            self.randompar,
            simulations=self.simulations,
            chains=1,
            save_sim=True,
        )

        csv.save(self.like, self.randompar, self.simulations)
        csv.save(self.like, self.randompar, self.simulations)
        csv.finalize()

        csv_new = db.get_datawriter(
            "csv",
            "UnitTest_tmp",
            self.parnames,
            self.like,
            self.randompar,
            simulations=self.simulations,
            chains=1,
            save_sim=True,
            dbappend=True,
        )

        csv_new.save(self.like, self.randompar, self.simulations)
        csv_new.save(self.like, self.randompar, self.simulations)
        csv_new.finalize()

        csvdata = csv_new.getdata()
        self.assertEqual(len(csvdata), 4)

    def test_csv_single_false(self):
        csv = db.get_datawriter(
            "csv",
            "UnitTest_tmp",
            self.parnames,
            self.like,
            self.randompar,
            simulations=self.simulations,
            chains=1,
            save_sim=False,
        )

        csv.save(self.like, self.randompar, self.simulations)
        csv.save(self.like, self.randompar, self.simulations)

        csv.finalize()
        csvdata = csv.getdata()
        self.assertEqual(str(type(csvdata)), str(type(np.array([]))))
        self.assertEqual(len(csvdata[0]), 7)
        self.assertEqual(len(csvdata), 2)
        self.assertEqual(len(csv.header), 7)

    def test_hdf5_multiline(self):
        hdf5 = db.get_datawriter(
            "hdf5",
            "UnitTest_tmp",
            self.parnames,
            self.like,
            self.randompar,
            simulations=self.simulations_multi,
            chains=1,
            save_sim=True,
        )

        hdf5.save(self.like, self.randompar, self.simulations_multi)
        hdf5.save(self.like, self.randompar, self.simulations_multi)
        # Save Simulations

        hdf5.finalize()
        hdf5data = hdf5.getdata()
        self.assertEqual(str(type(hdf5data)), str(type(np.array([]))))
        self.assertEqual(len(hdf5data[0]), 8)
        self.assertEqual(len(hdf5data), 2)
        self.assertEqual(len(hdf5.header), 32)

    def test_hdf5_multiline_false(self):
        # Save not Simulations
        hdf5 = db.get_datawriter(
            "hdf5",
            "UnitTest_tmp",
            self.parnames,
            self.like,
            self.randompar,
            simulations=self.simulations_multi,
            chains=1,
            save_sim=False,
        )

        hdf5.save(self.like, self.randompar, self.simulations_multi)
        hdf5.save(self.like, self.randompar, self.simulations_multi)

        hdf5.finalize()
        hdf5data = hdf5.getdata()
        self.assertEqual(str(type(hdf5data)), str(type(np.array([]))))
        self.assertEqual(len(hdf5data[0]), 7)
        self.assertEqual(len(hdf5data), 2)
        self.assertEqual(len(hdf5.header), 7)

    def test_hdf5_single(self):
        hdf5 = db.get_datawriter(
            "hdf5",
            "UnitTest_tmp",
            self.parnames,
            self.like,
            self.randompar,
            simulations=self.simulations,
            chains=1,
            save_sim=True,
        )

        hdf5.save(self.like, self.randompar, self.simulations)
        hdf5.save(self.like, self.randompar, self.simulations)

        hdf5.finalize()
        hdf5data = hdf5.getdata()
        self.assertEqual(str(type(hdf5data)), str(type(np.array([]))))
        self.assertEqual(len(hdf5data[0]), 8)
        self.assertEqual(len(hdf5data), 2)
        self.assertEqual(len(hdf5.header), 12)

    def test_hdf5_append(self):
        hdf5 = db.get_datawriter(
            "hdf5",
            "UnitTest_tmp",
            self.parnames,
            self.like,
            self.randompar,
            simulations=self.simulations,
            chains=1,
            save_sim=True,
        )

        hdf5.save(self.like, self.randompar, self.simulations)
        hdf5.save(self.like, self.randompar, self.simulations)
        hdf5.finalize()

        hdf5_new = db.get_datawriter(
            "hdf5",
            "UnitTest_tmp",
            self.parnames,
            self.like,
            self.randompar,
            simulations=self.simulations,
            chains=1,
            save_sim=True,
            dbappend=True,
        )

        hdf5_new.save(self.like, self.randompar, self.simulations)
        hdf5_new.save(self.like, self.randompar, self.simulations)
        hdf5_new.finalize()

        hdf5data = hdf5_new.getdata()
        self.assertEqual(len(hdf5data), 4)

    def test_hdf5_single_false(self):
        hdf5 = db.get_datawriter(
            "hdf5",
            "UnitTest_tmp",
            self.parnames,
            self.like,
            self.randompar,
            simulations=self.simulations,
            chains=1,
            save_sim=False,
        )

        hdf5.save(self.like, self.randompar, self.simulations)
        hdf5.save(self.like, self.randompar, self.simulations)

        hdf5.finalize()
        hdf5data = hdf5.getdata()
        self.assertEqual(str(type(hdf5data)), str(type(np.array([]))))
        self.assertEqual(len(hdf5data[0]), 7)
        self.assertEqual(len(hdf5data), 2)
        self.assertEqual(len(hdf5.header), 7)

    def test_sql_multiline(self):
        sql = db.get_datawriter(
            "sql",
            "UnitTest_tmp",
            self.parnames,
            self.like,
            self.randompar,
            simulations=self.simulations_multi,
            chains=1,
            save_sim=True,
        )
        sql.save(self.like, self.randompar, self.simulations_multi)
        sql.finalize()
        sqldata = sql.getdata()
        self.assertEqual(str(type(sqldata)), str(type(np.array([]))))
        self.assertEqual(len(sqldata[0]), 32)
        self.assertEqual(len(sqldata), 1)
        self.assertEqual(len(sql.header), 32)

    def test_sql_multiline_false(self):
        sql = db.get_datawriter(
            "sql",
            "UnitTest_tmp",
            self.parnames,
            self.like,
            self.randompar,
            simulations=self.simulations_multi,
            chains=1,
            save_sim=False,
        )
        sql.save(self.like, self.randompar, self.simulations_multi)
        sql.finalize()
        sqldata = sql.getdata()
        self.assertEqual(str(type(sqldata)), str(type(np.array([]))))
        self.assertEqual(len(sqldata[0]), 7)
        self.assertEqual(len(sqldata), 1)
        self.assertEqual(len(sql.header), 7)

    def test_sql_single(self):
        sql = db.get_datawriter(
            "sql",
            "UnitTest_tmp",
            self.parnames,
            self.like,
            self.randompar,
            simulations=self.simulations,
            chains=1,
            save_sim=True,
        )
        sql.save(self.like, self.randompar, self.simulations)
        sql.finalize()
        sqldata = sql.getdata()
        self.assertEqual(str(type(sqldata)), str(type(np.array([]))))
        self.assertEqual(len(sqldata[0]), 12)
        self.assertEqual(len(sqldata), 1)
        self.assertEqual(len(sql.header), 12)

    def test_sql_single_false(self):
        sql = db.get_datawriter(
            "sql",
            "UnitTest_tmp",
            self.parnames,
            self.like,
            self.randompar,
            simulations=self.simulations,
            chains=1,
            save_sim=False,
        )
        sql.save(self.like, self.randompar, self.simulations)
        sql.finalize()

        sqldata = sql.getdata()
        self.assertEqual(str(type(sqldata)), str(type(np.array([]))))
        self.assertEqual(len(sqldata[0]), 7)
        self.assertEqual(len(sqldata), 1)
        self.assertEqual(len(sql.header), 7)

    def test_ram_multiline(self):
        ram = db.get_datawriter(
            "ram",
            "UnitTest_tmp",
            self.parnames,
            self.like,
            self.randompar,
            simulations=self.simulations_multi,
            chains=1,
            save_sim=True,
        )
        ram.save(self.like, self.randompar, self.simulations_multi)
        ram.finalize()

        ramdata = ram.getdata()
        self.assertEqual(str(type(ramdata)), str(type(np.array([]))))
        self.assertEqual(len(ram.header), 32)
        self.assertEqual(len(ramdata[0]), 32)
        self.assertEqual(len(ramdata), 1)
        self.assertEqual(len(ramdata.dtype), len(ram.header))

    def test_ram_multiline_false(self):
        ram = db.get_datawriter(
            "ram",
            "UnitTest_tmp",
            self.parnames,
            self.like,
            self.randompar,
            simulations=self.simulations_multi,
            chains=1,
            save_sim=False,
        )
        ram.save(self.like, self.randompar, self.simulations_multi)

        ram.finalize()
        ramdata = ram.getdata()
        self.assertEqual(str(type(ramdata)), str(type(np.array([]))))
        self.assertEqual(len(ramdata[0]), 7)
        self.assertEqual(len(ramdata), 1)
        self.assertEqual(len(ramdata.dtype), len(ram.header))
        self.assertEqual(len(ram.header), 7)

    def test_ram_single(self):
        ram = db.get_datawriter(
            "ram",
            "UnitTest_tmp",
            self.parnames,
            self.like,
            self.randompar,
            simulations=self.simulations,
            chains=1,
            save_sim=True,
        )
        ram.save(self.like, self.randompar, self.simulations)

        ram.finalize()
        ramdata = ram.getdata()
        self.assertEqual(str(type(ramdata)), str(type(np.array([]))))
        self.assertEqual(len(ramdata[0]), 12)
        self.assertEqual(len(ramdata), 1)
        self.assertEqual(len(ramdata.dtype), len(ram.header))
        self.assertEqual(len(ram.header), 12)

    def test_ram_single_false(self):
        ram = db.get_datawriter(
            "ram",
            "UnitTest_tmp",
            self.parnames,
            self.like,
            self.randompar,
            simulations=self.simulations,
            chains=1,
            save_sim=False,
        )
        ram.save(self.like, self.randompar, self.simulations)

        ram.finalize()
        ramdata = ram.getdata()
        self.assertEqual(str(type(ramdata)), str(type(np.array([]))))
        self.assertEqual(len(ramdata[0]), 7)
        self.assertEqual(len(ramdata), 1)
        self.assertEqual(len(ramdata.dtype), len(ram.header))
        self.assertEqual(len(ram.header), 7)

    def test_not_existing_dbformat(self):
        with self.assertRaises(AttributeError):
            _ = db.get_datawriter(
                "xxx",
                "UnitTest_tmp",
                self.parnames,
                self.like,
                self.randompar,
                simulations=self.simulations,
                chains=1,
                save_sim=True,
            )

    def test_noData(self):
        nodata = db.get_datawriter(
            "noData",
            "UnitTest_tmp",
            self.parnames,
            np.array(self.like),
            self.randompar,
            simulations=self.simulations,
            chains=1,
            save_sim=True,
        )
        nodata.save(self.like, self.randompar, self.simulations)
        nodata.finalize()
        self.assertEqual(nodata.getdata(), None)

    def test_custom(self):
        custom = db.get_datawriter(
            "custom",
            "UnitTest_tmp",
            self.parnames,
            self.like,
            self.randompar,
            setup=MockSetup(),
            simulations=self.simulations,
            chains=1,
            save_sim=True,
        )
        custom.save(self.like, self.randompar, self.simulations)
        custom.finalize()
        self.assertEqual(custom.getdata(), None)

    def test_custom_no_setup(self):
        with self.assertRaises(ValueError):
            _ = db.get_datawriter(
                "custom",
                "UnitTest_tmp",
                self.parnames,
                self.like,
                self.randompar,
                simulations=self.simulations,
                chains=1,
                save_sim=True,
            )

    def test_custom_wrong_setup(self):
        with self.assertRaises(AttributeError):
            _ = db.get_datawriter(
                "custom",
                "UnitTest_tmp",
                self.parnames,
                self.like,
                self.randompar,
                setup=[],
                simulations=self.simulations,
                chains=1,
                save_sim=True,
            )


if __name__ == "__main__":
    unittest.main()
