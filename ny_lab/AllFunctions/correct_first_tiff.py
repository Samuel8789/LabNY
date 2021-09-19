# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 17:20:51 2021

@author: sp3660
"""

"""
add at he end of the movies file of caiman to correct for prairie mismanaging
"""
import numpy as np

def correct_first_tiff(array):
    """
    Custom made by me samuel to correct for paririw managing tiffs wrong
    """
    full=array
    if len(full.shape)>2:
        print('Something wrong with this tiff')
    
        first_elim=np.delete(full,1,axis=0)
        second_elim=np.delete(first_elim,np.array(list(range(32))[1:31]),axis=1)
        final=np.squeeze(second_elim)
       
        return final
    else:
        return full