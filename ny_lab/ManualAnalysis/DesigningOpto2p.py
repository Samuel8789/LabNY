# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 07:42:02 2023

@author: sp3660
"""
import numpy as np
import matplotlib.pyplot as plt
freq=20#hz
reps=10
optotime=20#ms
postoptotime=10000#ms
cellnumber=5
preoptodelay=10000
postopto=10000
trialnum=20
intertrialtime=10000





period=int(1*1000/freq)#ms
stimtime=period*reps
isi=period-optotime
reptime=period
totalcelltime=reptime*reps
trialduration=(totalcelltime+postoptotime)*cellnumber
exptime=(trialduration+intertrialtime)*trialnum+preoptodelay+postopto
exptimemin=exptime/1000/60

expvolt=np.zeros(exptime)

allupp=np.zeros((cellnumber,reps,trialnum),dtype='int')
for trial in range(trialnum):
    for celll in range(cellnumber):
    
       allupp[celll,:,trial]=np.arange(celll*(postoptotime+totalcelltime),
                                 celll*(postoptotime+totalcelltime)+totalcelltime,
                                 period,
                                 dtype='int')
       
    allupp[:,:,trial]=allupp[:,:,trial]+(trialduration+intertrialtime)*trial
    
    
allupp= allupp+preoptodelay
    
for trial in range(trialnum):
    for celll in range(cellnumber):
        for rep in range(reps):
            expvolt[allupp[celll,rep,trial]:allupp[celll,rep,trial]+stimtime]=1
            
            
plt.plot(expvolt)
