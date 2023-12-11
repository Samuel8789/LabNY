# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 10:17:22 2023

@author: sp3660
"""

cnmf=ext.cnm_object
dff=cnmf.estimates.detrend_df_f()

alldterended=dff.F_dff[analysis.full_data['imaging_data']['Plane1']['CellIds'],:]
cellid=0

plt.plot(alldterended[cellid,:])

plt.plot(full_traces[cellid,:])