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


excelfile=r'C:\Users\sp3660\Desktop\ExperimentalFirstAdd.xlsx'
databasefile=r'C:\Users\sp3660\Documents\Projects\LabNY\4. Mouse Managing\MouseDatabase.db'

#%%
Projects_table = pd.read_excel(
    excelfile, 
    sheet_name='Projects',
    header=0)

Brain_areas_status_table = pd.read_excel(
    excelfile, 
    sheet_name='Brain Areas',
    header=0)

Coordinates_table = pd.read_excel(
    excelfile, 
    sheet_name='Stereotactic corordinates',
    header=0)

Serotypes_table = pd.read_excel(
    excelfile, 
    sheet_name='Serotype',
    header=0)
Sensor_table = pd.read_excel(
    excelfile, 
    sheet_name='Sensors',
    header=0)

Opto_table = pd.read_excel(
    excelfile, 
    sheet_name='Optogenetic Actuators',
    header=0)
Reporter_table = pd.read_excel(
    excelfile, 
    sheet_name='Reporters',
    header=0)
Recombinase_table = pd.read_excel(
    excelfile, 
    sheet_name='Recombinase Types',
    header=0)
Promoter_table = pd.read_excel(
    excelfile, 
    sheet_name='Promoter',
    header=0)

experimental_status_table = pd.read_excel(
    excelfile, 
    sheet_name='Experimental_status',
    header=0)

#%%
Experimental_table = pd.read_excel(
    excelfile, 
    sheet_name='Experimental Animals',
    header=0)

Experimental_table.dtypes
Experimental_table['Mouse_ID'] = Experimental_table['Mouse_ID'].astype('Int64')
Experimental_table['EarMark'] = Experimental_table['EarMark'].astype('Int64')
Experimental_table['Experimental_status'] = Experimental_table['Experimental_status'].astype('Int64')
Experimental_table['NumberOfInjections'] = Experimental_table['NumberOfInjections'].astype('Int64')
Experimental_table['Injection1ID'] = Experimental_table['Injection1ID'].astype('Int64')
Experimental_table['Injection2ID'] = Experimental_table['Injection2ID'].astype('Int64')
Experimental_table['Injection3ID'] = Experimental_table['Injection3ID'].astype('Int64')
Experimental_table['WindowID'] = Experimental_table['WindowID'].astype('Int64')
Experimental_table['BrainProcessingID'] = Experimental_table['BrainProcessingID'].astype('Int64')
Experimental_table['Injection2Date'] = Experimental_table['Injection2Date'].astype('datetime64[ns]')
Experimental_table['Injection3Date'] = Experimental_table['Injection3Date'].astype('datetime64[ns]')

#%%

Injections_table = pd.read_excel(
    excelfile, 
    sheet_name='Injections',
    header=0)

Injections_table.dtypes
Injections_table['CorticalArea'] = Injections_table['CorticalArea'].astype('Int64')
Injections_table['InjectionSites'] = Injections_table['InjectionSites'].astype('Int64')
Injections_table['InjectionSite1Coordinates'] = Injections_table['InjectionSite1Coordinates'].astype('Int64')
Injections_table['InjectionSite2Coordinates'] = Injections_table['InjectionSite2Coordinates'].astype('Int64')






#%%
Windows_table = pd.read_excel(
    excelfile, 
    sheet_name='CranialQIndows',
    header=0)

Windows_table.dtypes
Windows_table['CorticalArea'] = Windows_table['CorticalArea'].astype('Int64')
Windows_table['HeadPlateCoordinates'] = Windows_table['HeadPlateCoordinates'].astype('Int64')
Windows_table['WindowType'] = Windows_table['WindowType'].astype('Int64')
Windows_table['CranioSize'] = Windows_table['CranioSize'].astype('Int64')
Windows_table['CoverType'] = Windows_table['CoverType'].astype('Int64')
Windows_table['CoverSize'] = Windows_table['CoverSize'].astype('Int64')
Windows_table['Durotomy'] = Windows_table['Durotomy'].astype('Int64')



#%%

Virus_table = pd.read_excel(
    excelfile, 
    sheet_name='Virus',
    header=0)

Virus_table.dtypes
Virus_table['CurrentAliquots'] = Virus_table['CurrentAliquots'].astype('Int64')
Virus_table['InitialAliquots'] = Virus_table['InitialAliquots'].astype('Int64')
Virus_table['Reporter'] = Virus_table['Reporter'].astype('Int64')
Virus_table['Sensor'] = Virus_table['Sensor'].astype('Int64')
Virus_table['Opto'] = Virus_table['Opto'].astype('Int64')


#%%

Virus_combinations_table = pd.read_excel(
    excelfile, 
    sheet_name='Virus Combinations',
    header=0)

Virus_combinations_table.dtypes
Virus_combinations_table['Virus1'] = Virus_combinations_table['Virus1'].astype('Int64')
Virus_combinations_table['Virus2'] = Virus_combinations_table['Virus2'].astype('Int64')
Virus_combinations_table['Virus3'] = Virus_combinations_table['Virus3'].astype('Int64')

#%%
db_conn = sqlite3.connect(databasefile)
c = db_conn.cursor()
#%%
c.execute(
    """
CREATE TABLE ExperimentalStatus_table (
    ID INTEGER PRIMARY KEY AUTOINCREMENT ,
    Status TEXT
    );
"""
)



c.execute(
    """
CREATE TABLE Projects_table (
    ID INTEGER PRIMARY KEY AUTOINCREMENT ,
    Projects TEXT
    );
"""
)

c.execute(
    """
CREATE TABLE Brain_Areas_table (
    ID INTEGER PRIMARY KEY AUTOINCREMENT ,
    BrainAreas TEXT
    );
"""
)

c.execute(
    """
CREATE TABLE Serotypes_table (
    ID INTEGER PRIMARY KEY AUTOINCREMENT ,
    Serotypes TEXT
    );
"""
)
c.execute(
    """
CREATE TABLE Sensors_table (
    ID INTEGER PRIMARY KEY AUTOINCREMENT ,
    Sensors TEXT
    );
"""
)
c.execute(
    """
CREATE TABLE Optos_table (
    ID INTEGER PRIMARY KEY AUTOINCREMENT ,
    Optos TEXT
    );
"""
)
c.execute(
    """
CREATE TABLE Reporters_table (
    ID INTEGER PRIMARY KEY AUTOINCREMENT ,
    Reporters TEXT
    );
"""
)
c.execute(
    """
CREATE TABLE Recombinase_table (
    ID INTEGER PRIMARY KEY AUTOINCREMENT ,
    Recombinases TEXT
    );
"""
)
c.execute(
    """
CREATE TABLE Promoter_table (
    ID INTEGER PRIMARY KEY AUTOINCREMENT ,
    Promoters TEXT
    );
"""
)
c.execute(
    """
CREATE TABLE Sterocoordinates_table (
    ID INTEGER PRIMARY KEY AUTOINCREMENT ,
    X TEXT,
    Y TEXT,
    Z TEXT,
    );
"""
)

c.execute(
    """
CREATE TABLE VirusCombinations_table (
    ID INTEGER PRIMARY KEY AUTOINCREMENT ,
    Combination TEXT,
    Virus1 INTEGER,
    Virus2 INTEGER,
    Virus3 INTEGER,
    
    FOREIGN KEY (Virus1) REFERENCES Virus_table (ID)
    FOREIGN KEY (Virus2) REFERENCES Virus_table (ID)
    FOREIGN KEY (Virus3) REFERENCES Virus_table (ID)
    
    );
"""
)

c.execute(
    """
CREATE TABLE Virus_table (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    VirusCode TEXT,
    VirusName TEXT,
    Origin TEXT,
    CurrentAliquots INTEGER,
    Storage INTEGER,
    Location TEXT,
    Genomesml REAL,
    AliquotDate TEXT,
    InitialAliquots INTEGER,
    Reporter INTEGER,
    Sensor INTEGER,
    Opto INTEGER,
    Recombinase INTEGER,
    Promoter INTEGER,
    Serotype INTEGER,
    StockNumber TEXT,
    Lot TEXT,
    RequestDate TEXT,
    OrderDate TEXT,
    ArrivalDate TEXT,
    Notes TEXT,
    
    FOREIGN KEY (Reporter) REFERENCES Reporters_table (ID)
    FOREIGN KEY (Sensor) REFERENCES Sensors_table (ID)
    FOREIGN KEY (Opto) REFERENCES Optos_table (ID)
    FOREIGN KEY (Recombinase) REFERENCES Recombinase_table (ID)
    FOREIGN KEY (Promoter) REFERENCES Promoter_table (ID)
    FOREIGN KEY (Serotype) REFERENCES Serotypes_table (ID)
    
    );
"""
)


c.execute('''ALTER TABLE Virus_table ADD COLUMN AliquotVolume REAL''')

c.execute(
    """
CREATE TABLE ExperimentalAnimals_table (
    ID INTEGER PRIMARY KEY AUTOINCREMENT ,
    Code TEXT,
    Project INTEGER,
    Mouse_ID INTEGER,
    EarMark INTEGER,
    Experimental_status INTEGER,
    Experiment INTEGER,
    NumberOfInjections INTEGER,
    Injection1Date TEXT,
    Injection1ID INTEGER,
    Injection2Date TEXT,
    Injection2ID INTEGER,
    Injection3Date TEXT,
    Injection3ID INTEGER,
    WindowDate TEXT,
    WindowID INTEGER,
    BrainProcessingDate TEXT,
    BrainProcessingID INTEGER,
    Notes TEXT,
    
    FOREIGN KEY (Project) REFERENCES Projects_table (ID)
    FOREIGN KEY (Mouse_ID) REFERENCES MICE_table (ID)
    FOREIGN KEY (EarMark) REFERENCES Labels_table (ID)
    FOREIGN KEY (Experimental_status) REFERENCES ExperimentalStatus_table (ID)
    FOREIGN KEY (Experiment) REFERENCES Experimental_table (ID)
    FOREIGN KEY (Injection1ID) REFERENCES Injections_table (ID)
    FOREIGN KEY (Injection2ID) REFERENCES Injections_table (ID)
    FOREIGN KEY (Injection3ID) REFERENCES Injections_table (ID)
    FOREIGN KEY (WindowID) REFERENCES Windows_table (ID)
    FOREIGN KEY (BrainProcessingID) REFERENCES BrainProcessing_table (ID)
    );
"""
)
c.execute(
    """
CREATE TABLE Injections_table (
    ID INTEGER PRIMARY KEY AUTOINCREMENT ,
    ExpID INTEGER,
    InjDate TEXT,
    VirusCombination INTEGER,
    DilutionSensor1 REAL,
    DilutionSensor2 REAL,
    DilutionOpto REAL,
    CorticalArea INTEGER,
    InjectionSites INTEGER,
    InjectionSite1Coordinates INTEGER,
    InjectionSite1volume REAL,
    InjectionSite1speed REAL,
    InjectionSite1pretime REAL,
    InjectionSite1posttime REAL,
    InjectionSite1goodvolume TEXT,
    InjectionSite1bleeding TEXT,
    InjectionSite2Coordinates INTEGER,
    InjectionSite2volume REAL,
    InjectionSite2speed REAL,
    InjectionSite2pretime REAL,
    InjectionSite2posttime REAL,
    InjectionSite2goodvolume TEXT,
    InjectionSite2bleeding TEXT,
    PostInjection1 TEXT, 
    PostInjection2 TEXT,
    Notes TEXT,
    
    FOREIGN KEY (ExpID) REFERENCES ExperimentalAnimals_table (ID)
    FOREIGN KEY (VirusCombination) REFERENCES VirusCombinations_table (ID)
    FOREIGN KEY (CorticalArea) REFERENCES Brain_Areas_table (ID)
    FOREIGN KEY (InjectionSite1Coordinates) REFERENCES Sterocoordinates_table (ID)
    FOREIGN KEY (InjectionSite2Coordinates) REFERENCES Sterocoordinates_table (ID)

    );
"""
)
c.execute(
    """
CREATE TABLE Windows_table (
    ID INTEGER PRIMARY KEY AUTOINCREMENT ,
    ExpID INTEGER,
    WindDate TEXT,
    CorticalArea INTEGER,
    HeadPlateCoordinates INTEGER,
    WindowType INTEGER,
    CranioSize INTEGER,
    CoverType INTEGER,
    CoverSize INTEGER,
    Durotomy INTEGER,
    DamagedAreas TEXT,
    PostInjection1 TEXT, 
    PostInjection2 TEXT,
    Notes TEXT,
    
    FOREIGN KEY (ExpID) REFERENCES ExperimentalAnimals_table (ID)
    FOREIGN KEY (CorticalArea) REFERENCES Brain_Areas_table (ID)
    FOREIGN KEY (HeadPlateCoordinates) REFERENCES Sterocoordinates_table (ID)
    FOREIGN KEY (WindowType) REFERENCES WindowTypes_table (ID)
    FOREIGN KEY (CranioSize) REFERENCES CranioSize_table (ID)
    FOREIGN KEY (CoverType) REFERENCES CoverType_table (ID)
    FOREIGN KEY (CoverSize) REFERENCES CoverSIze_table (ID)

    );
"""
)

#%%

experimental_status_table.to_sql('ExperimentalStatus_table', db_conn, if_exists='append', index=False)
Projects_table.to_sql('Projects_table', db_conn, if_exists='append', index=False)
Brain_areas_status_table.to_sql('Brain_Areas_table', db_conn, if_exists='append', index=False)
Serotypes_table.to_sql('Serotypes_table', db_conn, if_exists='append', index=False)
Sensor_table.to_sql('Sensors_table', db_conn, if_exists='append', index=False)
Opto_table.to_sql('Optos_table', db_conn, if_exists='append', index=False)
Reporter_table.to_sql('Reporters_table', db_conn, if_exists='append', index=False)
Recombinase_table.to_sql('Recombinase_table', db_conn, if_exists='append', index=False)
Promoter_table.to_sql('Promoter_table', db_conn, if_exists='append', index=False)
Coordinates_table.to_sql('Sterocoordinates_table', db_conn, if_exists='append', index=False)
Virus_combinations_table.to_sql('VirusCombinations_table', db_conn, if_exists='replace', index=False)
Virus_table.to_sql('Virus_table', db_conn, if_exists='append', index=False)
Experimental_table.to_sql('ExperimentalAnimals_table', db_conn, if_exists='append', index=False)
Injections_table.to_sql('Injections_table', db_conn, if_exists='append', index=False)
Windows_table.to_sql('Windows_table', db_conn, if_exists='append', index=False)








#%%
db_conn = sqlite3.connect(databasefile)
c = db_conn.cursor()
db_conn.close()

#%%