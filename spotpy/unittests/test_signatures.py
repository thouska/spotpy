import unittest

try:
    import spotpy
except ImportError:
    import sys
    sys.path.append(".")
    import spotpy



import spotpy.signatures as sig
from spotpy.examples.spot_setup_hymod_python import spot_setup
import numpy as np
import os

try:
    import pandas as pd
    test = True
except ImportError:
    print('Please install Pandas to use these signature functions')
    test = False

#https://docs.python.org/3/library/unittest.html



class TestSignatures(unittest.TestCase):

    def setUp(self):
        self.data = np.random.gamma(0.7,2,500)
        self.spot_setup = spot_setup()
        self.parameterset = self.spot_setup.parameters()['random']
        self.simulation = self.spot_setup.simulation(self.parameterset)
        self.observation = self.spot_setup.evaluation()

        self.timespanlen = self.simulation.__len__()
        try:

            self.ddd = pd.date_range("2015-01-01 11:00", freq="5min", periods=self.timespanlen)
            self.dd_daily = pd.date_range("2015-05-01", periods=self.timespanlen)
            self.usepandas = True
        except NameError:
            print('Please install Pandas to use these signature functions')
            self.usepandas = False

    def test_getSlopeFDC(self):
        sig_val = sig.getSlopeFDC(self.simulation,self.observation, mode="get_signature")
        sig_raw = sig.getSlopeFDC(self.simulation, self.observation, mode="get_raw_data")
        sig_dev = sig.getSlopeFDC(self.simulation, self.observation, mode="calc_Dev")
        self.assertEqual(type(float(sig_val)), type(1.0))
        self.assertEqual(type(float(sig_raw)), type(1.0))
        self.assertEqual(type(float(sig_dev)), type(1.0))


    def test_getAverageFloodOverflowPerSection(self):
        if self.usepandas:
            for th in range(-10,10):

                sig_val = sig.getAverageFloodOverflowPerSection(self.simulation, self.observation, mode="get_signature", datetime_series=self.dd_daily,
                                                            threshold_value=th)
                sig_raw = sig.getAverageFloodOverflowPerSection(self.simulation, self.observation, mode="get_raw_data", datetime_series=self.dd_daily,
                                                        threshold_value=th)
                sig_dev = sig.getAverageFloodOverflowPerSection(self.simulation, self.observation, mode="calc_Dev", datetime_series=self.dd_daily,
                                                        threshold_value=th)
                self.assertEqual(type(float(sig_val.astype(float))),type(1.0))



                self.assertEqual(sig_raw.dtypes[0],"float64")
                self.assertEqual(sig_raw["flood"].__len__(), 1461)
                #self.assertEqual(str(type(sig_raw.index.tolist()[0])),"<class 'pandas.tslib.Timestamp'>")
                self.assertEqual(type(float(sig_dev.astype(float))), type(1.0))


    def test_getMeanFlow(self):
        self.assertEqual(type(1.0),type(float(sig.getMeanFlow(self.simulation,self.observation,mode="get_signature"))))
        self.assertEqual(type(1.0), type(float(sig.getMeanFlow(self.simulation, None, mode="get_signature"))))
        self.assertEqual(type(1.0), type(float(sig.getMeanFlow(self.simulation, self.observation, mode="get_raw_data"))))
        self.assertEqual(type(1.0), type(float(sig.getMeanFlow(self.simulation,None, mode="get_raw_data"))))
        self.assertEqual(type(1.0), type(float(sig.getMeanFlow(self.simulation, self.observation, mode="calc_Dev"))))


    def test_getMedianFlow(self):
        self.assertEqual(type(1.0), type(float(sig.getMedianFlow(self.simulation, self.observation, mode="get_signature"))))
        self.assertEqual(type(1.0), type(float(sig.getMedianFlow(self.simulation, None, mode="get_signature"))))
        self.assertEqual(type(1.0), type(float(sig.getMedianFlow(self.simulation, self.observation, mode="get_raw_data"))))
        self.assertEqual(type(1.0), type(float(sig.getMedianFlow(self.simulation, None, mode="get_raw_data"))))
        self.assertEqual(type(1.0), type(float(sig.getMedianFlow(self.simulation, self.observation, mode="calc_Dev"))))

    def test_getSkewness(self):
        self.assertEqual(type(1.0), type(float(sig.getSkewness(self.simulation, self.observation, mode="get_signature"))))
        self.assertEqual(type(1.0), type(float(sig.getSkewness(self.simulation, None, mode="get_signature"))))
        self.assertEqual(type(1.0), type(float(sig.getSkewness(self.simulation, self.observation, mode="get_raw_data"))))
        self.assertEqual(type(1.0), type(float(sig.getSkewness(self.simulation, None, mode="get_raw_data"))))
        self.assertEqual(type(1.0), type(float(sig.getSkewness(self.simulation, self.observation, mode="calc_Dev"))))


    def test_getCoeffVariation(self):
        self.assertEqual(type(1.0), type(float(sig.getCoeffVariation(self.simulation, self.observation, mode="get_signature"))))
        self.assertEqual(type(1.0), type(float(sig.getCoeffVariation(self.simulation, None, mode="get_signature"))))
        self.assertEqual(type(1.0), type(float(sig.getCoeffVariation(self.simulation, self.observation, mode="get_raw_data"))))
        self.assertEqual(type(1.0), type(float(sig.getCoeffVariation(self.simulation, None, mode="get_raw_data"))))
        self.assertEqual(type(1.0), type(float(sig.getCoeffVariation(self.simulation, self.observation, mode="calc_Dev"))))


    def test_getQ001(self):
        self.assertEqual(type(1.0), type(float(sig.getQ001(self.simulation, self.observation, mode="get_signature"))))
        self.assertEqual(type(1.0), type(float(sig.getQ001(self.simulation, None, mode="get_signature"))))
        self.assertEqual(type(1.0), type(float(sig.getQ001(self.simulation, self.observation, mode="get_raw_data"))))
        self.assertEqual(type(1.0), type(float(sig.getQ001(self.simulation, None, mode="get_raw_data"))))
        self.assertEqual(type(1.0), type(float(sig.getQ001(self.simulation, self.observation, mode="calc_Dev"))))


    def test_getQ01(self):
        self.assertEqual(type(1.0), type(float(sig.getQ01(self.simulation, self.observation, mode="get_signature"))))
        self.assertEqual(type(1.0), type(float(sig.getQ01(self.simulation, None, mode="get_signature"))))
        self.assertEqual(type(1.0), type(float(sig.getQ01(self.simulation, self.observation, mode="get_raw_data"))))
        self.assertEqual(type(1.0), type(float(sig.getQ01(self.simulation, None, mode="get_raw_data"))))
        self.assertEqual(type(1.0), type(float(sig.getQ01(self.simulation, self.observation, mode="calc_Dev"))))

    def test_getQ1(self):
        self.assertEqual(type(1.0), type(float(sig.getQ1(self.simulation, self.observation, mode="get_signature"))))
        self.assertEqual(type(1.0), type(float(sig.getQ1(self.simulation, None, mode="get_signature"))))
        self.assertEqual(type(1.0), type(float(sig.getQ1(self.simulation, self.observation, mode="get_raw_data"))))
        self.assertEqual(type(1.0), type(float(sig.getQ1(self.simulation, None, mode="get_raw_data"))))
        self.assertEqual(type(1.0), type(float(sig.getQ1(self.simulation, self.observation, mode="calc_Dev"))))

    def test_getQ5(self):
        self.assertEqual(type(1.0), type(float(sig.getQ5(self.simulation, self.observation, mode="get_signature"))))
        self.assertEqual(type(1.0), type(float(sig.getQ5(self.simulation, None, mode="get_signature"))))
        self.assertEqual(type(1.0), type(float(sig.getQ5(self.simulation, self.observation, mode="get_raw_data"))))
        self.assertEqual(type(1.0), type(float(sig.getQ5(self.simulation, None, mode="get_raw_data"))))
        self.assertEqual(type(1.0), type(float(sig.getQ5(self.simulation, self.observation, mode="calc_Dev"))))

    def test_getQ10(self):
        self.assertEqual(type(1.0), type(float(sig.getQ10(self.simulation, self.observation, mode="get_signature"))))
        self.assertEqual(type(1.0), type(float(sig.getQ10(self.simulation, None, mode="get_signature"))))
        self.assertEqual(type(1.0), type(float(sig.getQ10(self.simulation, self.observation, mode="get_raw_data"))))
        self.assertEqual(type(1.0), type(float(sig.getQ10(self.simulation, None, mode="get_raw_data"))))
        self.assertEqual(type(1.0), type(float(sig.getQ10(self.simulation, self.observation, mode="calc_Dev"))))


    def test_getQ20(self):
        self.assertEqual(type(1.0), type(float(sig.getQ20(self.simulation, self.observation, mode="get_signature"))))
        self.assertEqual(type(1.0), type(float(sig.getQ20(self.simulation, None, mode="get_signature"))))
        self.assertEqual(type(1.0), type(float(sig.getQ20(self.simulation, self.observation, mode="get_raw_data"))))
        self.assertEqual(type(1.0), type(float(sig.getQ20(self.simulation, None, mode="get_raw_data"))))
        self.assertEqual(type(1.0), type(float(sig.getQ20(self.simulation, self.observation, mode="calc_Dev"))))


    def test_getQ85(self):
        self.assertEqual(type(1.0), type(float(sig.getQ85(self.simulation, self.observation, mode="get_signature"))))
        self.assertEqual(type(1.0), type(float(sig.getQ85(self.simulation, None, mode="get_signature"))))
        self.assertEqual(type(1.0), type(float(sig.getQ85(self.simulation, self.observation, mode="get_raw_data"))))
        self.assertEqual(type(1.0), type(float(sig.getQ85(self.simulation, None, mode="get_raw_data"))))
        self.assertEqual(type(1.0), type(float(sig.getQ85(self.simulation, self.observation, mode="calc_Dev"))))



    def test_getQ99(self):
        self.assertEqual(type(1.0), type(float(sig.getQ99(self.simulation, self.observation, mode="get_signature"))))
        self.assertEqual(type(1.0), type(float(sig.getQ99(self.simulation, None, mode="get_signature"))))
        self.assertEqual(type(1.0), type(float(sig.getQ99(self.simulation, self.observation, mode="get_raw_data"))))
        self.assertEqual(type(1.0), type(float(sig.getQ99(self.simulation, None, mode="get_raw_data"))))
        self.assertEqual(type(1.0), type(float(sig.getQ99(self.simulation, self.observation, mode="calc_Dev"))))


    def test_getAverageFloodFrequencyPerSection(self):
        if self.usepandas:
            for th in range(-10, 10):
                th = th + np.random.uniform(-1, 1, 1)[0]
                sig_val = sig.getAverageFloodFrequencyPerSection(self.simulation, self.observation, datetime_series=self.dd_daily, threshold_value=th,
                                                           mode="get_signature")

                sig_raw = sig.getAverageFloodFrequencyPerSection(self.simulation, self.observation, datetime_series=self.dd_daily, threshold_value=th,
                                                           mode="get_raw_data")

                sig_dev = sig.getAverageFloodFrequencyPerSection(self.simulation, self.observation, datetime_series=self.dd_daily, threshold_value=th,
                                                       mode="calc_Dev")

                self.assertEqual(sig_raw.dtypes[0], "float64")
                self.assertEqual(sig_raw["flood"].__len__(), 1461)

                #self.assertEqual(str(type(sig_raw.index.tolist()[0])), "<class 'pandas.tslib.Timestamp'>")

                self.assertEqual(type(sig_dev), type(1.0))
                self.assertEqual(type(sig_val), type(1.0))


    def test_getAverageFloodDuration(self):
        if self.usepandas:
            for th in range(-10, 10):
                th = th + np.random.uniform(-1, 1, 1)[0]
                sig_val = sig.getAverageFloodDuration(self.simulation, self.observation, datetime_series=self.dd_daily, threshold_value=th,
                                              mode="get_signature")
                sig_raw = sig.getAverageFloodDuration(self.simulation, self.observation, datetime_series=self.dd_daily, threshold_value=th,
                                              mode="get_raw_data")
                sig_dev = sig.getAverageFloodDuration(self.simulation, self.observation, datetime_series=self.dd_daily, threshold_value=th,
                                              mode="calc_Dev")

                self.assertEqual(sig_raw.dtypes[0], "float64")
                self.assertEqual(sig_raw["flood"].__len__(), 1461)

                #self.assertEqual(str(type(sig_raw.index.tolist()[0])), "<class 'pandas.tslib.Timestamp'>")

                self.assertEqual(type(sig_dev), type(1.0))
                self.assertEqual(type(sig_val), type(1.0))


    def test_getAverageBaseflowUnderflowPerSection(self):
        if self.usepandas:
            for th in range(-10, 10):
                th = th + np.random.uniform(-1, 1, 1)[0]
                sig_val =sig.getAverageBaseflowUnderflowPerSection(self.simulation, self.observation, datetime_series=self.dd_daily,
                                                                threshold_value=th, mode="get_signature")
                sig_raw = sig.getAverageBaseflowUnderflowPerSection(self.simulation, self.observation, datetime_series=self.dd_daily,
                                                                threshold_value=th, mode="get_raw_data")
                sig_dev = sig.getAverageBaseflowUnderflowPerSection(self.simulation, self.observation, datetime_series=self.dd_daily,
                                                                threshold_value=th, mode="calc_Dev")

                self.assertTrue(sig_raw.dtypes[0] == "int64" or sig_raw.dtypes[0] == "float64")
                self.assertEqual(sig_raw["baseflow"].__len__(), 1461)

                #self.assertEqual(str(type(sig_raw.index.tolist()[0])), "<class 'pandas.tslib.Timestamp'>")

                self.assertEqual(type(float(sig_dev.astype(float))), type(1.0))

                self.assertEqual(type(float(sig_val.astype(float))), type(1.0))

    def test_getAverageBaseflowFrequencyPerSection(self):
        if self.usepandas:
            for th in range(-10, 10):
                th = th + np.random.uniform(-1,1,1)[0]
                sig_val = sig.getAverageBaseflowFrequencyPerSection(self.simulation, self.observation, datetime_series=self.dd_daily,
                                                                threshold_value=th, mode="get_signature")
                sig_raw = sig.getAverageBaseflowFrequencyPerSection(self.simulation, self.observation, datetime_series=self.ddd, threshold_value=th,
                                                                mode="get_raw_data")
                sig_dev = sig.getAverageBaseflowFrequencyPerSection(self.simulation, self.observation, datetime_series=self.dd_daily,
                                                                threshold_value=th, mode="calc_Dev")

                self.assertTrue(sig_raw.dtypes[0] ==  "int64" or sig_raw.dtypes[0] ==  "float64")
                self.assertEqual(sig_raw["baseflow"].__len__(), 1461)

                #self.assertEqual(str(type(sig_raw.index.tolist()[0])), "<class 'pandas.tslib.Timestamp'>")

                self.assertEqual(type(sig_dev), type(1.0))
                self.assertEqual(type(sig_val), type(1.0))

    def test_getAverageBaseflowDuration(self):
        if self.usepandas:
            for th in range(-10, 10):
                th = th + np.random.uniform(-1,1,1)[0]

                sig_val = sig.getAverageBaseflowDuration(self.simulation, self.observation, datetime_series=self.dd_daily,
                                                                threshold_value=th, mode="get_signature")

                sig_raw = sig.getAverageBaseflowDuration(self.simulation, self.observation, datetime_series=self.ddd, threshold_value=th,
                                                                mode="get_raw_data")
                sig_dev = sig.getAverageBaseflowDuration(self.simulation, self.observation, datetime_series=self.dd_daily,
                                                                threshold_value=th, mode="calc_Dev")

                self.assertTrue(sig_raw.dtypes[0] == "int64" or sig_raw.dtypes[0] == "float64")
                self.assertEqual(sig_raw["baseflow"].__len__(), 1461)

                #self.assertEqual(str(type(sig_raw.index.tolist()[0])), "<class 'pandas.tslib.Timestamp'>")

                self.assertEqual(type(sig_dev), type(1.0))
                self.assertEqual(type(sig_val), type(1.0))




    def test_getFloodFrequency(self):
        if self.usepandas:
            for th in range(-10, 10):
                th = th + np.random.uniform(-1, 1, 1)[0]
                sig_val = sig.getFloodFrequency(self.simulation, self.observation,
                                        datetime_series=pd.date_range("2015-05-01", periods=self.timespanlen), threshold_value=th,
                                        mode="get_signature")
                sig_raw_dd = sig.getFloodFrequency(self.simulation, self.observation,
                                        datetime_series=self.dd_daily, threshold_value=th,
                                        mode="get_raw_data")
                sig_raw_ddd = sig.getFloodFrequency(self.simulation, self.observation,
                                               datetime_series=self.ddd, threshold_value=th,
                                               mode="get_raw_data")

                sig_dev = sig.getFloodFrequency(self.simulation, self.observation,
                                        datetime_series=pd.date_range("2015-05-01", periods=self.timespanlen), threshold_value=th,
                                        mode="calc_Dev")




                self.assertEqual(sig_raw_dd.dtypes[0], "int64")
                self.assertEqual(sig_raw_dd["count"].__len__(), 1461)

                self.assertEqual(sig_raw_ddd.dtypes[0], "int64")
                self.assertEqual(sig_raw_ddd["count"].__len__(), 122)


                #self.assertEqual(str(type(sig_raw_dd.index.tolist()[0])), "<class 'pandas.tslib.Timestamp'>")
                #self.assertEqual(str(type(sig_raw_ddd.index.tolist()[0])), "<class 'pandas.tslib.Timestamp'>")


                self.assertEqual(type(float(sig_dev.astype(float))), type(42.0) )
                self.assertEqual(type(float(sig_val.astype(float))), type(1.0))

    def test_getBaseflowFrequency(self):
        if self.usepandas:
            for th in range(-10, 10):
                th = th + np.random.uniform(-1, 1, 1)[0]
                sig_val = sig.getBaseflowFrequency(self.simulation, self.observation,
                                        datetime_series=pd.date_range("2015-05-01", periods=self.timespanlen), threshold_value=th,
                                        mode="get_signature")
                sig_raw_dd = sig.getBaseflowFrequency(self.simulation, self.observation,
                                        datetime_series=self.dd_daily, threshold_value=th,
                                        mode="get_raw_data")
                sig_raw_ddd = sig.getBaseflowFrequency(self.simulation, self.observation,
                                               datetime_series=self.ddd, threshold_value=th,
                                               mode="get_raw_data")

                sig_dev = sig.getBaseflowFrequency(self.simulation, self.observation,
                                        datetime_series=pd.date_range("2015-05-01", periods=self.timespanlen), threshold_value=th,
                                        mode="calc_Dev")

                self.assertEqual(sig_raw_dd.dtypes[0], "int64")
                self.assertEqual(sig_raw_dd["count"].__len__(), 1461)

                self.assertEqual(sig_raw_ddd.dtypes[0], "int64")
                self.assertEqual(sig_raw_ddd["count"].__len__(), 122)


                #self.assertEqual(str(type(sig_raw_dd.index.tolist()[0])), "<class 'pandas.tslib.Timestamp'>")
                #self.assertEqual(str(type(sig_raw_ddd.index.tolist()[0])), "<class 'pandas.tslib.Timestamp'>")


                self.assertEqual(type(float(sig_dev)), type(42.0) )
                self.assertEqual(type(float(sig_val)), type(1.0))

    def test_getLowFlowVar(self):
        if self.usepandas:
            sig_sig_1 = sig.getLowFlowVar(self.simulation, self.observation, datetime_series=pd.date_range("2015-05-01", periods=self.timespanlen),mode="get_signature")
            sig_sig_2 = sig.getLowFlowVar(self.simulation, None,
                                                datetime_series=pd.date_range("2015-05-01", periods=self.timespanlen),
                                                mode="get_signature")
            sig_raw = sig.getLowFlowVar(self.simulation, self.observation, datetime_series=pd.date_range("2015-05-01", periods=self.timespanlen),mode="get_raw_data")
            sig_raw_2 = sig.getLowFlowVar(self.simulation, None,
                                        datetime_series=pd.date_range("2015-05-01", periods=self.timespanlen),
                                        mode="get_raw_data")
            sig_dev = sig.getLowFlowVar(self.simulation, self.observation, datetime_series=pd.date_range("2015-05-01", periods=self.timespanlen),mode="calc_Dev")

            self.assertEqual(type(1.0), type(float(sig_sig_1)))
            self.assertEqual(type(1.0), type(float(sig_sig_2)))
            self.assertEqual(type(1.0), type(float(sig_raw)))
            self.assertEqual(type(1.0), type(float(sig_raw_2)))
            self.assertEqual(type(1.0), type(float(sig_dev)))

    def test_getHighFlowVar(self):
        if self.usepandas:
            sig_sig_1 = sig.getHighFlowVar(self.simulation, self.observation,
                                          datetime_series=pd.date_range("2015-05-01", periods=self.timespanlen),
                                          mode="get_signature")
            sig_sig_2 = sig.getHighFlowVar(self.simulation, None,
                                          datetime_series=pd.date_range("2015-05-01", periods=self.timespanlen),
                                          mode="get_signature")
            sig_raw = sig.getHighFlowVar(self.simulation, self.observation,
                                        datetime_series=pd.date_range("2015-05-01", periods=self.timespanlen),
                                        mode="get_raw_data")
            sig_raw_2 = sig.getHighFlowVar(self.simulation, None,
                                          datetime_series=pd.date_range("2015-05-01", periods=self.timespanlen),
                                          mode="get_raw_data")
            sig_dev = sig.getHighFlowVar(self.simulation, self.observation,
                                        datetime_series=pd.date_range("2015-05-01", periods=self.timespanlen),
                                        mode="calc_Dev")

            self.assertEqual(type(1.0), type(float(sig_sig_1)))
            self.assertEqual(type(1.0), type(float(sig_sig_2)))
            self.assertEqual(type(1.0), type(float(sig_raw)))
            self.assertEqual(type(1.0), type(float(sig_raw_2)))
            self.assertEqual(type(1.0), type(float(sig_dev)))

    def test_getBaseflowIndex(self):
        if self.usepandas:
            sig_raw = sig.getBaseflowIndex(self.simulation, self.observation,
                                       datetime_series=self.dd_daily,
                                       mode="get_raw_data")

            self.assertEqual(type(sig_raw) , type({}))
            self.assertGreater(sig_raw.__len__(),0)

            sig_sig_1 = sig.getBaseflowIndex(self.simulation, self.observation,
                                           datetime_series=pd.date_range("2015-05-01", periods=self.timespanlen),
                                           mode="get_signature")
            sig_sig_2 = sig.getBaseflowIndex(self.simulation, None,
                                           datetime_series=pd.date_range("2015-05-01", periods=self.timespanlen),
                                           mode="get_signature")

            sig_dev = sig.getBaseflowIndex(self.simulation, self.observation,
                                        datetime_series=pd.date_range("2015-05-01", periods=self.timespanlen),
                                        mode="calc_Dev")

            self.assertEqual(type(1.0), type(float(sig_sig_1)))
            self.assertEqual(type({}), type(sig_sig_2))
            self.assertEqual(type(1.0), type(float(sig_dev)))

if __name__ == '__main__':
    if test==True: # Happens if Pandas can be imported (not on travis so far)
        unittest.main()