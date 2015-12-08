# -*- coding: utf-8 -*-
'''
Copyright 2015 by Tobias Houska
This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Tobias Houska

This tool holds functions for statistic analysis. It takes Python-lists and
returns the objective function value of interest.
'''

import numpy as np

def bias(evalution,simulation):
    """
    Bias
    
        .. math::
        
         Bias=\\frac{1}{N}\\sum_{i=1}^{N}(e_{i}-s_{i})

    :evaluation: Observed data to compared with simulation data.
    :type: list
    
    :simulation: simulation data to compared with evalution data
    :type: list
    
    :return: Bias
    :rtype: float
    """    
    if len(evalution)==len(simulation):   
        bias_values=[]       
        for i in range(len(evalution)):
            if evalution[i] == -99999:
                '''
                Cleans out No Data values
                '''
                print 'Wrong Results! Clean out No Data Values'                 
                pass
             
            else:            
                bias_values.append(float(simulation[i]) - float(evalution[i]))
        bias_sum = np.sum(bias_values[0:len(bias_values)])       
        bias = bias_sum/len(bias_values)       
        return float(bias)
    
    else:
        print "Error: evalution and simulation lists does not have the same length."

        
def nashsutcliff(evalution,simulation):   
    """
    Nash-Sutcliff model efficinecy
    
        .. math::

         NSE = 1-\\frac{\\sum_{i=1}^{N}(e_{i}-s_{i})^2}{\\sum_{i=1}^{N}(e_{i}-\\bar{e})^2} 
    
    :evaluation: Observed data to compared with simulation data.
    :type: list
    
    :simulation: simulation data to compared with evalution data
    :type: list
    
    :return: Nash-Sutcliff model efficiency
    :rtype: float
    
    """   
    if len(evalution)==len(simulation):
        s,e=np.array(simulation),np.array(evalution)
        #s,e=simulation,evalution       
        mean_observed = np.mean(e)
        # compute numerator and denominator
        numerator = sum((e - s) ** 2)
        denominator = sum((e - mean_observed)**2)
        # compute coefficient
        return 1 - (numerator/denominator)
         #return coefficient
        #return float(1 - sum((s-e)**2)/sum((e-np.mean(e))**2))
        
    else:
        print "Error: evalution and simulation lists does not have the same length."

        
def lognashsutcliff(evalution,simulation):
    """
    log Nash-Sutcliff model efficiency
   
        .. math::

         NSE = 1-\\frac{\\sum_{i=1}^{N}(log(e_{i})-log(s_{i}))^2}{\\sum_{i=1}^{N}(log(e_{i})-log(\\bar{e})^2}-1)*-1
 
    :evaluation: Observed data to compared with simulation data.
    :type: list
    
    :simulation: simulation data to compared with evalution data
    :type: list
    
    :return: log Nash-Sutcliff model efficiency
    :rtype: float
    
    """   
    return float(1 - sum((np.log(simulation)-np.log(evalution))**2)/sum((np.log(evalution)-np.mean(np.log(evalution)))**2))

    
def log_p(evalution,simulation):
    from scipy import stats    
    """
    Logarithmic probability distribution
    
    :evaluation: Observed data to compared with simulation data.
    :type: list
    
    :simulation: simulation data to compared with evalution data
    :type: list
    
    :return: Logarithmic probability distribution
    :rtype: float
    """ 
    logLik = np.mean( stats.norm.logpdf(evalution, loc=simulation, scale=.1) )
    return logLik

    
def correlationcoefficient(evalution,simulation):
    """
    Correlation Coefficient
    
        .. math::
        
         r = \\frac{\\sum ^n _{i=1}(e_i - \\bar{e})(s_i - \\bar{s})}{\\sqrt{\\sum ^n _{i=1}(e_i - \\bar{e})^2} \\sqrt{\\sum ^n _{i=1}(s_i - \\bar{s})^2}}
    
    :evaluation: Observed data to compared with simulation data.
    :type: list
    
    :simulation: simulation data to compared with evalution data
    :type: list
    
    :return: Corelation Coefficient
    :rtype: float
    """ 
    if len(evalution)==len(simulation):
        Corelation_Coefficient = np.corrcoef(evalution,simulation)[0,1]
        return Corelation_Coefficient
    else:
        return "Error: evalution and simulation lists does not have the same length."
  
  
def rsquared(evalution,simulation):
    """
    Coefficient of Determination
    
        .. math::
        
         r^2=(\\frac{\\sum ^n _{i=1}(e_i - \\bar{e})(s_i - \\bar{s})}{\\sqrt{\\sum ^n _{i=1}(e_i - \\bar{e})^2} \\sqrt{\\sum ^n _{i=1}(s_i - \\bar{s})^2}})^2
    
    :evaluation: Observed data to compared with simulation data.
    :type: list
    
    :simulation: simulation data to compared with evalution data
    :type: list
    
    :return: Coefficient of Determination
    :rtype: float
    """
    if len(evalution)==len(simulation):
        return correlationcoefficient(evalution,simulation)**2
    else:
        return "Error: evalution and simulation lists does not have the same length."
        

def mse(evalution,simulation):
    """
    Mean Squared Error
    
        .. math::
        
         MSE=\\frac{1}{N}\\sum_{i=1}^{N}(e_{i}-s_{i})^2
    
    :evaluation: Observed data to compared with simulation data.
    :type: list
    
    :simulation: simulation data to compared with evalution data
    :type: list
    
    :return: Mean Squared Error
    :rtype: float
    """
    
    if len(evalution)==len(simulation):
        
        MSE_values=[]
                
        for i in range(len(evalution)):
            MSE_values.append((simulation[i] - evalution[i])**2)        
        
        MSE_sum = np.sum(MSE_values[0:len(evalution)])
        
        MSE=MSE_sum/(len(evalution))
        return MSE
    else:
        return "Error: evalution and simulation lists does not have the same length."

        
def rmse(evalution,simulation):
    """
    Root Mean Squared Error
    
        .. math::
        
         RMSE=\\sqrt{\\frac{1}{N}\\sum_{i=1}^{N}(e_{i}-s_{i})^2}
        
    :evaluation: Observed data to compared with simulation data.
    :type: list
    
    :simulation: simulation data to compared with evalution data
    :type: list
    
    :return: Root Mean Squared Error
    :rtype: float
    """
    if len(evalution)==len(simulation):
        return np.sqrt(mse(evalution,simulation))
    else:
        return "Error: evalution and simulation lists does not have the same length."    


def mae(evalution,simulation):
    """
    Mean Absolute Error

        .. math::
            
         MAE=\\frac{1}{N}\\sum_{i=1}^{N}(\\left |  e_{i}-s_{i} \\right |)
   
    :evaluation: Observed data to compared with simulation data.
    :type: list
    
    :simulation: simulation data to compared with evalution data
    :type: list
    
    :return: Mean Absolute Error
    :rtype: float
    """
    if len(evalution)==len(simulation):
        
        MAE_values=[]
                
        for i in range(len(evalution)):
            MAE_values.append(np.abs(simulation[i] - evalution[i]))        
        
        MAE_sum = np.sum(MAE_values[0:len(evalution)])
        
        MAE = MAE_sum/(len(evalution))
        
        return MAE
    else:
        return "Error: evalution and simulation lists does not have the same length."  


def rrmse(evalution,simulation):
    """
    Relative Root Mean Squared Error
    
        .. math::   

         RRMSE=\\frac{\\sqrt{\\frac{1}{N}\\sum_{i=1}^{N}(e_{i}-s_{i})^2}}{\\bar{e}}
           
    :evaluation: Observed data to compared with simulation data.
    :type: list
    
    :simulation: simulation data to compared with evalution data
    :type: list
    
    :return: Relative Root Mean Squared Error
    :rtype: float
    """
    
    if len(evalution)==len(simulation):

        RRMSE = rmse(evalution,simulation)/np.mean(simulation)
        return RRMSE
        
    else:
        return "Error: evalution and simulation lists does not have the same length."   
    
        
def agreementindex(evalution,simulation):
    """
    Agreement Index (d) developed by Willmott (1981)

        .. math::   
    
         d = 1 - \\frac{\\sum_{i=1}^{N}(e_{i} - s_{i})^2}{\\sum_{i=1}^{N}(\\left | s_{i} - \\bar{e} \\right | + \\left | e_{i} - \\bar{e} \\right |)^2}  
                
    
    :evaluation: Observed data to compared with simulation data.
    :type: list
    
    :simulation: simulation data to compared with evalution data
    :type: list
    
    :return: Agreement Index
    :rtype: float
    """
    if len(evalution)==len(simulation):
        simulation,evalution=np.array(simulation),np.array(evalution)
        Agreement_index=1 -(np.sum((evalution-simulation)**2))/(np.sum(
                (np.abs(simulation-np.mean(evalution))+np.abs(evalution-np.mean(evalution)))**2))
        return Agreement_index
    else:
        return "Error: evalution and simulation lists does not have the same length." 
    

def _variance(evalution):
    """
    Variance
    
        .. math:: 
        
         Variance = \\frac{1}{N}\\sum_{i=1}^{N}(e_{i}-\\bar{e})^2
            
    :evaluation: Observed data to compared with simulation data.
    :type: list
    
    :return: Variance
    :rtype: float
    """
    Variance_values = []    
    for i in range(len(evalution)):
        Variance_values.append((evalution[i]-np.mean(evalution))**2)            
    Variance = np.sum(Variance_values)/len(evalution) 
    return Variance
    

def covariance(evalution,simulation):
    """
    Covariance
    
        .. math::
         Covariance = \\frac{1}{N} \\sum_{i=1}^{N}((e_{i} - \\bar{e}) * (s_{i} - \\bar{s}))
    
    :evaluation: Observed data to compared with simulation data.
    :type: list
    
    :simulation: simulation data to compared with evalution data
    :type: list
    
    :return: Covariance
    :rtype: float
    """
    if len(evalution)==len(simulation):
        Covariance_values = []
        
        for i in range(len(evalution)):
            Covariance_values.append((evalution[i]-np.mean(evalution))*(simulation[i]-np.mean(simulation)))
            
        Covariance = np.sum(Covariance_values)/(len(evalution))
        return Covariance
    else:
        return "Error: evalution and simulation lists does not have the same length." 

def _standarddeviation(evalution):
    """
    Standard Derivation (sigma)
    
        .. math::
         sigma = \\sqrt{\\frac{1}{N}\\sum_{i=1}^{N}(e_{i}-\\bar{e})^2}
    
    :evaluation: Observed data to compared with simulation data.
    :type: list
    
    :simulation: simulation data to compared with evalution data
    :type: list
    
    :return: Standard Derivation
    :rtype: float
    """
    
    return np.sqrt(_variance(evalution))


def decomposed_mse(evalution,simulation):
    """
    Decomposed MSE developed by Kobayashi and Salam (2000)
    
        .. math ::
         dMSE = (\\frac{1}{N}\\sum_{i=1}^{N}(e_{i}-s_{i}))^2 + SDSD + LCS
    
         SDSD = (\\sigma(e) - \\sigma(s))^2
         
         LCS = 2 \\sigma(e) \\sigma(s) * (1 - \\frac{\\sum ^n _{i=1}(e_i - \\bar{e})(s_i - \\bar{s})}{\\sqrt{\\sum ^n _{i=1}(e_i - \\bar{e})^2} \\sqrt{\\sum ^n _{i=1}(s_i - \\bar{s})^2}})
    
    :evaluation: Observed data to compared with simulation data.
    :type: list
    
    :simulation: simulation data to compared with evalution data
    :type: list
    
    :return: Decomposed MSE
    :rtype: float
    """
    
    if len(evalution)==len(simulation):
        
        Decomposed_MSE = str(round((bias(evalution,simulation))**2,2))+'(bias**2) + '+str(round((_standarddeviation(evalution)-_standarddeviation(simulation))**2,2))+'(SDSD) + '+str(round(2*_standarddeviation(evalution)*_standarddeviation(simulation)*(1-correlationcoefficient(evalution,simulation)),2))+'(LCS)'         
        
        return Decomposed_MSE
    else:
        return "Error: evalution and simulation lists does not have the same length."