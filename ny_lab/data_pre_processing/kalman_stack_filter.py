# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 10:16:46 2021

@author: sp3660
"""
import numpy as np

def kalman_stack_filter(imageStack,gain= 0.9,percentvar = 0.5):
# function imageStack=Kalman_Stack_Filter(imageStack,percentvar,gain)
#
# Purpose
# Implements a predictive Kalman-like filter in the time domain of the image
# stack. Algorithm taken from Java code by C.P. Mauer.
# http://rsb.info.nih.gov/ij/plugins/kalman.html
#
# Inputs
# imageStack - a 3d matrix comprising of a noisy image sequence. Time is
#              the 3rd dimension. 
# gain - the strength of the filter [0 to 1]. Larger gain values means more
#        aggressive filtering in time so a smoother function with a lower 
#        peak. Gain values above 0.5 will weight the predicted value of the 
#        pixel higher than the observed value.
# percentvar - the initial estimate for the noise [0 to 1]. Doesn't have
#              much of an effect on the algorithm. 
#
# Output
# imageStack - the filtered image stack
#
# Note:
# The time series will look noisy at first then become smoother as the
# filter accumulates evidence. 
# 
# Rob Campbell, August 2009


#Copy the last frame onto the end so that we filter the whole way
#through

    NewImageStack=np.append(imageStack, [imageStack[-1,:,:]], axis=0)
    
    #Set up variables
    width = NewImageStack.shape[1]
    height = NewImageStack.shape[2]
    stacksize = NewImageStack.shape[0]
    
    tmp=np.ones((width,height));
    
    #Set up priors
    predicted = NewImageStack[0,:,:]; 
    predictedvar = tmp*percentvar;
    noisevar=predictedvar
    
    #Now conduct the Kalman-like filtering on the image stack
    for i in range(1,stacksize-1):
      stackslice = NewImageStack[i+1,:,:]; 
      observed = stackslice;
      
      Kalman =np.divide( predictedvar,predictedvar+noisevar)
      corrected = gain*predicted + (1.0-gain)*observed + np.multiply(Kalman, observed-predicted)        
      correctedvar = np.multiply(predictedvar,tmp - Kalman)
     
      predictedvar = correctedvar;
      predicted = corrected;
      NewImageStack[i,:,:]=corrected;
    
    Kalman_Filtered_Array=np.delete(NewImageStack,-1, axis=0)

    return Kalman_Filtered_Array


