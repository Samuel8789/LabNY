# -*- coding: utf-8 -*-
"""
Created on Thu May 13 14:14:22 2021

@author: sp3660
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 12 11:22:35 2021

@author: sp3660
"""

"""
adding all experimental tables to database




"""

import pandas as pd
import numpy as np
import sqlite3
from sqlite3 import Error


excelfile=r'C:\Users\sp3660\Desktop\experimentakaddsecond.xlsx'
databasefile=r'C:\Users\sp3660\Documents\Projects\LabNY\4. Mouse Managing\MouseDatabase.db'


#%%
db_conn = sqlite3.connect(databasefile)
c = db_conn.cursor()
#%%

c.execute(
    """
CREATE TABLE SavedPaths_table (
    ID INTEGER PRIMARY KEY AUTOINCREMENT ,
    DataFolder TEXT,
    Path TEXT
    
    );
"""
)





#%%







#%%
db_conn.close()

#%%