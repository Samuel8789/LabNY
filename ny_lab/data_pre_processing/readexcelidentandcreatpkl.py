# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 11:51:31 2022

@author: sp3660
"""
import os
import pandas as pd
import time
import pickle
import numpy as np

timestr = time.strftime("%Y%m%d-%H%M%S")
aqname=''
datapath=r'D:\Projects\LabNY\Full_Mice_Pre_Processed_Data\Mice_Projects\Interneuron_Imaging\G2C\Ai14\SPKG\data\211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000'
identfile='SPKGcellidentiity.xlsx'
fullfilepath=os.path.join(datapath, identfile)

pyr_int_identif_path_name='_'.join(['211015_SPKG_FOV1_3planeallenA_920_50024_narrow_without-000', timestr,'pyr_int_identification.pkl'])  




planes=['Plane1', 'Plane2', 'Plane3']
pyr_int_identification={}
df1 = pd.read_excel(fullfilepath, sheet_name="Plane1", engine="openpyxl")
df2 = pd.read_excel(fullfilepath, sheet_name="Plane2", engine="openpyxl")
df3 = pd.read_excel(fullfilepath, sheet_name="Plane3", engine="openpyxl")
dfs=[df1,df2,df3]

pyramidal_count={}
interneuron_count={}
for i,plane in enumerate(planes):
    pyr_int_identification[plane]={'interneuron':{'matlab':np.array(dfs[i][(dfs[i]['Accepted']=='+')& (dfs[i]['Tomato accepted only']=='+')].iloc[:,0].tolist()),
                                                   'python':np.array(dfs[i][(dfs[i]['Accepted']=='+')& (dfs[i]['Tomato accepted only']=='+')].iloc[:,0].tolist())-1
                                                   },
                                    'pyramidals':{'matlab':np.array(dfs[i][(dfs[i]['Accepted']=='+')& (dfs[i]['Tomato accepted only']=='-')].iloc[:,0].tolist()),
                                                  'python':np.array(dfs[i][(dfs[i]['Accepted']=='+')& (dfs[i]['Tomato accepted only']=='-')].iloc[:,0].tolist())-1
                                                  }
                                    }



    pyramidal_count[plane]=pyr_int_identification[plane]['pyramidals']['python'].shape[0]
    interneuron_count[plane]=pyr_int_identification[plane]['interneuron']['python'].shape[0]





datapath=os.path.join(datapath, pyr_int_identif_path_name)
if not os.path.isfile(datapath):
    with open(datapath, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(pyr_int_identification, f, pickle.HIGHEST_PROTOCOL)