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
CoverSize_table = pd.read_excel(
    excelfile, 
    sheet_name='Coverslip Size',
    header=0)
CoverSize_table.dtypes

CranectomyType_table = pd.read_excel(
    excelfile, 
    sheet_name='Craniectomy Types',
    header=0)
CranectomyType_table.dtypes

Covertype_table = pd.read_excel(
    excelfile, 
    sheet_name='CoverType',
    header=0)
Covertype_table.dtypes

Craniosize_table = pd.read_excel(
    excelfile, 
    sheet_name='Craniotomy Size',
    header=0)
Craniosize_table.dtypes

BrainProcessing_table = pd.read_excel(
    excelfile, 
    sheet_name='Brain Processing',
    header=0)
BrainProcessing_table.dtypes
BrainProcessing_table['HistologyID'] = BrainProcessing_table['HistologyID'].astype('Int64')


ColdStorage_table = pd.read_excel(
    excelfile, 
    sheet_name='ColdStorageLocations',
    header=0)
ColdStorage_table.dtypes

FreezingLocations_table = pd.read_excel(
    excelfile, 
    sheet_name='FreezingLocations',
    header=0)
FreezingLocations_table.dtypes

#%%
db_conn = sqlite3.connect(databasefile)
c = db_conn.cursor()
#%%

c.execute(
    """
CREATE TABLE BrainProcessing_table (
    ID INTEGER PRIMARY KEY AUTOINCREMENT ,
    Mouse_ID INTEGER,
    BrainDate TEXT,
    PerfusionPFA TEXT,
    PostfixationPFA TEXT,
    PosxfixationTime TEXT,
    PostfixationTemperature TEXT,
    PrehistologyStorageDate TEXT,
    PrehistologyStorageSolution TEXT,
    PrehistologyStorageLocation INTEGER,
    HistologyDate TEXT,
    HistologyID INTEGER
    );
"""
)



c.execute(
    """
CREATE TABLE CoverSize_table (
    ID INTEGER PRIMARY KEY AUTOINCREMENT ,
    CoverslipSize TEXT
    );
"""
)

c.execute(
    """
CREATE TABLE CranectomyType_table (
    ID INTEGER PRIMARY KEY AUTOINCREMENT ,
    CraniectomyType TEXT
    );
"""
)

c.execute(
    """
CREATE TABLE Covertype_table (
    ID INTEGER PRIMARY KEY AUTOINCREMENT ,
    WindowType TEXT
    );
"""
)
c.execute(
    """
CREATE TABLE Craniosize_table (
    ID INTEGER PRIMARY KEY AUTOINCREMENT ,
    CraniectomySize TEXT
    );
"""
)
c.execute(
    """
CREATE TABLE ColdStorage_table (
    ID INTEGER PRIMARY KEY AUTOINCREMENT ,
    Fridges TEXT
    );
"""
)

c.execute(
    """
CREATE TABLE FreezingLocations_table (
    ID INTEGER PRIMARY KEY AUTOINCREMENT ,
    Freezers TEXT
    );
"""
)




#%%



CoverSize_table.to_sql('CoverSize_table', db_conn, if_exists='append', index=False)
CranectomyType_table.to_sql('CranectomyType_table', db_conn, if_exists='append', index=False)
Covertype_table.to_sql('Covertype_table', db_conn, if_exists='append', index=False)
Craniosize_table.to_sql('Craniosize_table', db_conn, if_exists='append', index=False)
BrainProcessing_table.to_sql('BrainProcessing_table', db_conn, if_exists='append', index=False)
ColdStorage_table.to_sql('ColdStorage_table', db_conn, if_exists='append', index=False)
FreezingLocations_table.to_sql('FreezingLocations_table', db_conn, if_exists='append', index=False)




#%%
db_conn.close()

#%%