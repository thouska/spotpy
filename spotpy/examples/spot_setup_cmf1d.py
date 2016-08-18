'''
Copyright 2015 by Tobias Houska
This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Tobias Houska

A one dimensional cmf model analysing data from the FACE experiment.
You need to have cmf installed on your system: svn checkout svn://fb09-pasig.umwelt.uni-giessen.de/cmf/trunk cmf
'''

import cmf
from datetime import timedelta, datetime
import Load_Data                as loader
import numpy as np
import spotpy

class spot_setup(object):
    def __init__(self):
        datastart     = datetime(1998,6,1)
        dataend       = datetime(2000,1,1)
        analysestart  = datetime(1999,1,1)
        self.cmfmodel = model(datastart,dataend,analysestart)
        self.params = [spotpy.parameter.Normal('alpha',0.3,0.1,0.02,0.2),
                       spotpy.parameter.Normal('n',1.2,0.035,0.01,1.22),
                       spotpy.parameter.Normal('ksat',1,0.3,0.1,2.0),
                       spotpy.parameter.Normal('porosity',.55,0.04,0.02,0.6),
                       ]
    
    def parameters(self):
        return spotpy.parameter.generate(self.params)        
    
    def simulation(self,vector):
        simulations= self.cmfmodel._run(alpha=vector[0],n=vector[1],ksat=vector[2],porosity=vector[3])
        return simulations
        
    def evaluation(self,evaldates=False):
        if evaldates:
            return self.cmfmodel.eval_dates
        else:
            return list(self.cmfmodel.observations)
    
    def objectivefunction(self,simulation,evaluation):
        objectivefunction= -spotpy.objectivefunctions.rmse(evaluation=evaluation,simulation=simulation)
        return objectivefunction
        
        
class model(object):
    '''
    Input: datastart:    e.g. datetime(1998,6,1)
           dataend:      e.g. datetime(2000,1,1)
           analysestart: e.g. datetime(1999,1,1)
    
    Output: Initialised model instance with forcing data (climate, groundwater) and evaluation data (soil moisture)
    '''
    def __init__(self,datastart,dataend,analysestart):
        self.datastart=datastart
        self.dataend=dataend
        self.analysestart=analysestart
        self.bound= [[0.0001,0.6],[0.01,3],[1.05,1.4],[0.4,0.7]]
        DataLoader   = loader.load_data(analysestart,datastart,dataend)
        cmf.set_parallel_threads(1)
        
        ###################### Forcing data ####################################
        ClimateFilename     = 'Climate_Face_new2.csv'
        try:
            self.meteoarray=np.load(ClimateFilename+str(datastart.date())+str(dataend.date())+'.npy')
            self.rain      = cmf.timeseries.from_array(begin = self.datastart, step = timedelta(hours=1), data=self.meteoarray['Nd_mm_day'])#in mm/day
            self.rHmean    = cmf.timeseries.from_array(begin = self.datastart, step = timedelta(hours=1), data=self.meteoarray['Rh'])
            self.Windspeed = cmf.timeseries.from_array(begin = self.datastart, step = timedelta(hours=1), data=self.meteoarray['Wind'])
            self.Rs        = cmf.timeseries.from_array(begin = self.datastart, step = timedelta(hours=1), data=self.meteoarray['Rs_meas'])
            self.T         = cmf.timeseries.from_array(begin = self.datastart, step = timedelta(hours=1), data=self.meteoarray['Temp'])
        
        except:
            DataLoader.climate_pickle(ClimateFilename)
            self.meteoarray=np.load(ClimateFilename+str(datastart.date())+str(dataend.date())+'.npy')
            self.rain      = cmf.timeseries.from_array(begin = self.datastart, step = timedelta(hours=1), data=self.meteoarray['Nd_mm_day'])#in mm/day
            self.rHmean    = cmf.timeseries.from_array(begin = self.datastart, step = timedelta(hours=1), data=self.meteoarray['Rh'])
            self.Windspeed = cmf.timeseries.from_array(begin = self.datastart, step = timedelta(hours=1), data=self.meteoarray['Wind'])
            self.Rs        = cmf.timeseries.from_array(begin = self.datastart, step = timedelta(hours=1), data=self.meteoarray['Rs_meas'])
            self.T         = cmf.timeseries.from_array(begin = self.datastart, step = timedelta(hours=1), data=self.meteoarray['Temp'])
        
        
        self.piezometer          = 'P4'
        self.gw_array            = DataLoader.groundwater(self.piezometer)
        ###########################################################################
        
        ###################### Evaluation data ####################################    
        eval_soil_moisture = DataLoader.soil_moisture('A1')
        self.eval_dates    = eval_soil_moisture['Date']
        self.observations  = eval_soil_moisture['A1']        
        ###########################################################################        
    

    def _load_meteo(self,project):
            #Create meteo station for project
            meteo=project.meteo_stations.add_station('FACE',position = (0,0,0),timezone=1,timestep=cmf.h)       
            rain            = self.rain
            meteo.rHmean    = self.rHmean
            meteo.Windspeed = self.Windspeed
            meteo.Rs        = self.Rs
            meteo.T         = self.T
            meteo.Tmax      = meteo.T.reduce_max(begin = self.datastart, step = timedelta(days=1))
            meteo.Tmin      = meteo.T.reduce_min(begin = self.datastart, step = timedelta(days=1))
            project.rainfall_stations.add('FACE',rain,(0,0,0))    
            project.use_nearest_rainfall()
            # Use the meteorological station for each cell of the project
            project.use_nearest_meteo()  
    
    def run(self,args):
        return self._run(*args)
        
    def _run(self,alpha=None,n=None,porosity=None,ksat=None):
        #return alpha,n,porosity,ksat
        '''
        Runs the model instance
        
        Input: Parameter set (in this case VAN-Genuchten Parameter alpha,n,porosity,ksat)
        Output: Simulated values on given observation days
        '''
        #Check if given parameter set is in realistic boundaries
        if alpha<self.bound[0][0] or alpha>self.bound[0][1] or ksat<self.bound[1][0] \
        or ksat>self.bound[1][1] or n<self.bound[2][0] or n>self.bound[2][1] or \
        porosity<self.bound[3][0] or porosity>self.bound[3][1]:
            print('The following combination was ignored')
            text='n= '+str(n)
            print(text)
            text='alpha='+str(alpha)
            print(text)
            text='ksat= '+str(ksat)
            print(text)
            text='porosity= '+str(porosity)
            print(text)
            print('##############################')
            return  self.observations*-np.inf
        else:
            project=cmf.project()
            cell = project.NewCell(x=0,y=0,z=0,area=1000, with_surfacewater=True)
            text='n= '+str(n)
            print(text)
            text='alpha='+str(alpha)
            print(text)
            text='ksat= '+str(ksat)
            print(text)
            text='porosity= '+str(porosity)
            print(text)
            print('##############################')
            r_curve = cmf.VanGenuchtenMualem(Ksat=ksat,phi=porosity,alpha=alpha,n=n)
            layers=5
            ldepth=.01     
            for i in range(layers):
                depth = (i+1) * ldepth
                cell.add_layer(depth,r_curve)
            cell.install_connection(cmf.Richards)
            cell.install_connection(cmf.ShuttleworthWallace)
            cell.saturated_depth =.5
            solver = cmf.CVodeIntegrator(project,1e-6)                  
            self._load_meteo(project)
            gw = project.NewOutlet('groundwater',x=0,y=0,z=.9)#layers*ldepth)
            cmf.Richards(cell.layers[-1],gw)
            gw.potential = -.5 #IMPORTANT
            gw.is_source=True
            solver.t = self.datastart
            Evalstep,evallist=0,[]
            rundays=(self.dataend-self.datastart).days
            for t in solver.run(solver.t,solver.t + timedelta(days=rundays),timedelta(hours=1)):
                if self.gw_array['Date'].__contains__(t)==True:
                   Gw_Index=np.where(self.gw_array['Date']==t)               
                   gw.potential=self.gw_array[self.piezometer][Gw_Index]
                   print(gw.potential) #TO DO: CHECK IF SOMETHING HAPPENS HERE!!!!
                if t > self.analysestart:
                    if Evalstep !=len(self.eval_dates) and t == self.eval_dates[Evalstep]:
                        evallist.append(cell.layers.wetness[0]*cell.layers.porosity[0]*100)
                        Evalstep+=1
            return evallist