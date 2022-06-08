# -*- coding: utf-8 -*-
"""
Created on Fri May 27 10:11:11 2022

@author: sp3660
"""

import json
import numpy as np

# the file to be converted to 
# json format
filename = r'G:\Projects\TemPrairireSSH\20220524 Hakim\Mice\SPKW\UnprocessedFaceCameras\220524_SPKW_AllenA_Hakim_1\220524_SPKW_AllenA_Hakim_1_MMStack_Default_metadata.txt'

# dictionary where the lines from
# text will be stored
dict1 = {}
# creating dictionary

with open(filename) as json_file:
    data = json.load(json_file)
    
    
    
framenames=list(data.keys())

usefuldata={}
for name, frame in data.items():
    if 'FrameKey' in name: 
        usefuldata[name]={}
        tosave=['Exposure-ms','ReceivedTime', 'ImageNumber', 'ElapsedTime-ms', 'UserData']
        for k in tosave:
            if k=='UserData':
                usefuldata[name][k]={}
                usefuldata[name][k]['StartTime-ms']=frame[k]['StartTime-ms']
                usefuldata[name][k]['TimeReceivedByCore']=frame[k]['TimeReceivedByCore']
            
            else:
                usefuldata[name][k]=frame[k]
        
    
timstamps=[ i['ElapsedTime-ms'] for i in usefuldata.values()]
interval=np.diff(timstamps)
   

