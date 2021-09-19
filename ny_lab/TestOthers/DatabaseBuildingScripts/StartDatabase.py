# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 14:44:18 2021

@author: sp3660
"""

import pandas as pd
import numpy as np
import sqlite3
from sqlite3 import Error


excelfile=r'C:\Users\sp3660\Desktop\ForMouseDatabase.xlsx'
databasefile=r'C:\Users\sp3660\Desktop\MouseDatabase.db'

#%%
Rooms_table = pd.read_excel(
    excelfile, 
    sheet_name='Rooms',
    header=0)

Action_types_table = pd.read_excel(
    excelfile, 
    sheet_name='Action Types',
    header=0)

Lines_table = pd.read_excel(
    excelfile, 
    sheet_name='Lines',
    header=0)

Genes_table = pd.read_excel(
    excelfile, 
    sheet_name='Genes',
    header=0)

Genotypes_table = pd.read_excel(
    excelfile, 
    sheet_name='Genotypes',
    header=0)

Genotyping_status_table = pd.read_excel(
    excelfile, 
    sheet_name='Genotypings',
    header=0)

Breeding_status_table = pd.read_excel(
    excelfile, 
    sheet_name='Breeders',
    header=0)

Experimental_status_table = pd.read_excel(
    excelfile, 
    sheet_name='Experimental',
    header=0)

Labels_table = pd.read_excel(
    excelfile, 
    sheet_name='Labels',
    header=0)

Sex_table = pd.read_excel(
    excelfile, 
    sheet_name='Sex',
    header=0)

Mice_table = pd.read_excel(
    excelfile, 
    sheet_name='Mouse Table',
    header=0)

Mice_table.dtypes
Mice_table['Cage'] = Mice_table['Cage'].astype('Int64')
Mice_table.dtypes

Breedings_table = pd.read_excel(
    excelfile, 
    sheet_name='Breedings',
    header=0)
Breedings_table.dtypes
Breedings_table['Cage'] = Breedings_table['Cage'].astype('Int64')
Breedings_table['Male'] = Breedings_table['Male'].astype('Int64')
Breedings_table['Female1'] = Breedings_table['Female1'].astype('Int64')
Breedings_table['Female2'] = Breedings_table['Female2'].astype('Int64')
Breedings_table['Male_Cage'] = Breedings_table['Male_Cage'].astype('Int64')
Breedings_table['Requires_Genotyping'] = Breedings_table['Requires_Genotyping'].astype('Int64')
Breedings_table.dtypes


Litters_table = pd.read_excel(
    excelfile, 
    sheet_name='Litters',
    header=0)
Litters_table.dtypes
Litters_table['NumberDead'] = Litters_table['NumberDead'].astype('Int64')



Actions_table = pd.read_excel(
    excelfile, 
    sheet_name='Actions',
    header=0)
Actions_table.dtypes
Actions_table['Mouse_2'] = Actions_table['Mouse_2'].astype('Int64')
Actions_table['Mouse_3'] = Actions_table['Mouse_3'].astype('Int64')
Actions_table['Mouse_4'] = Actions_table['Mouse_4'].astype('Int64')
Actions_table['Mouse_5'] = Actions_table['Mouse_5'].astype('Int64')
Actions_table.dtypes

#%%

db_conn = sqlite3.connect(databasefile)
c = db_conn.cursor()
#%%
c.execute(
    """
CREATE TABLE Sex_table (
    ID INTEGER PRIMARY KEY AUTOINCREMENT ,
    Sex_types TEXT
    );
"""
)


c.execute(
    """
CREATE TABLE Labels_table (
    ID INTEGER PRIMARY KEY AUTOINCREMENT ,
    Labels_types TEXT
    );
"""
)


c.execute(
    """
CREATE TABLE Experimental_table (
    ID INTEGER PRIMARY KEY AUTOINCREMENT ,
    Experimental_types TEXT
    );
"""
)


c.execute(
    """
CREATE TABLE Breeders_table (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Breeders_types TEXT
    );
"""
)


c.execute(
    """
CREATE TABLE Genotypings_table (
    ID INTEGER PRIMARY KEY AUTOINCREMENT ,
    Genotypings_types TEXT
    );
"""
)


c.execute(
    """
CREATE TABLE Genotypes_table (
    ID INTEGER PRIMARY KEY AUTOINCREMENT ,
    Genotypes_types TEXT
    );
"""
)


c.execute(
    """
CREATE TABLE Genes_table (
    ID INTEGER PRIMARY KEY AUTOINCREMENT ,
    Genes_types TEXT
    );
"""
)


c.execute(
    """
CREATE TABLE Lines_table (
    ID INTEGER PRIMARY KEY AUTOINCREMENT ,
    Line_Short TEXT,
    Line_Long TEXT
    );
"""
)


c.execute(
    """
CREATE TABLE Action_types_table (
    ID INTEGER PRIMARY KEY AUTOINCREMENT ,
    Action_types  TEXT
    );
"""
)


c.execute(
    """
CREATE TABLE Rooms_table (
    ID INTEGER PRIMARY KEY AUTOINCREMENT ,
    Room_types TEXT
    );
"""
)
#%%
c.execute(
    """
CREATE TABLE Breedings_table (
    ID INTEGER PRIMARY KEY AUTOINCREMENT ,
    Cage INTEGER,
    Male INTEGER,
    Female1 INTEGER,
    Female2 INTEGER,
    Line INTEGER NOT NULL,
    StartDate TEXT,
    EndDate TEXT,
    Male_Cage INTEGER,
    Breeding_Status INTEGER,
    Requires_Genotyping INTEGER,
    Notes TEXT,
    
    FOREIGN KEY (Line) REFERENCES Lines_table (ID)
    FOREIGN KEY (Male) REFERENCES MICE_table (ID)
    FOREIGN KEY (Female1) REFERENCES MICE_table (ID)
    FOREIGN KEY (Female2) REFERENCES MICE_table (ID)
    FOREIGN KEY (Breeding_Status) REFERENCES Breeders_table (ID)

    
    );
"""
)

c.execute(
    """
CREATE TABLE Litters_table (
    ID INTEGER PRIMARY KEY AUTOINCREMENT ,
    Cage INTEGER NOT NULL,
    Breeding_Parents INTEGER NOT NULL,
    Date_seen TEXT NOT NULL,
    Age INTEGER NOT NULL,
    Line INTEGER NOT NULL,
    Date_Weaned TEXT,
    NumberAlive INTEGER,
    NumberDead INTEGER,
    Notes TEXT,
    
    FOREIGN KEY (Line) REFERENCES Lines_table (ID)
    FOREIGN KEY (Breeding_Parents) REFERENCES Breedings_table(ID)

    );
"""
)

c.execute(
    """
CREATE TABLE Actions_table (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Date TEXT NOT NULL,
    Action_Type INTEGER NOT NULL,
    Cage_start INT,
    Cage_end INT,
    Mouse_1 INTEGER NOT NULL,
    Mouse_2 INTEGER,
    Mouse_3 INTEGER,
    Mouse_4 INTEGER,
    Mouse_5 INTEGER,
    Notes TEXT,
    
    FOREIGN KEY (Action_Type) REFERENCES Action_types_table (ID)
    FOREIGN KEY (Mouse_1) REFERENCES MICE_table(ID)
    FOREIGN KEY (Mouse_2) REFERENCES MICE_table(ID)
    FOREIGN KEY (Mouse_3) REFERENCES MICE_table(ID)
    FOREIGN KEY (Mouse_4) REFERENCES MICE_table(ID)
    FOREIGN KEY (Mouse_5) REFERENCES MICE_table(ID)

    );
"""
)


#%%
c.execute(
    """
CREATE TABLE MICE_table (
    ID INTEGER PRIMARY KEY AUTOINCREMENT ,
    Lab_Number  INTEGER NOT NULL,
    Sex INTEGER NOT NULL,
    Cage INTEGER,
    Breeding_status INTEGER NOT NULL,
    DOB TEXT,
    Label INTEGER NOT NULL,
    Parent_Breeding INTEGER NOT NULL,
    Ai14 INTEGER NOT NULL,
    Ai65 INTEGER NOT NULL,
    Ai75 INTEGER NOT NULL,
    Ai80 INTEGER NOT NULL,
    Ai148 INTEGER NOT NULL,
    Ai162 INTEGER NOT NULL,
    G2C INTEGER NOT NULL,
    VGC INTEGER NOT NULL,
    VRC INTEGER NOT NULL,
    PVF INTEGER NOT NULL,
    SLF INTEGER NOT NULL,
    Line INTEGER NOT NULL,
    Room INTEGER NOT NULL,
    Experimental_Status INTEGER NOT NULL,
    Genotyping_Status INTEGER NOT NULL,
    Alive INTEGER NOT NULL,
    Notes TEXT,
    
    FOREIGN KEY (Sex) REFERENCES Sex_table (ID)
    FOREIGN KEY (Breeding_status) REFERENCES Breeders_table (ID)
    FOREIGN KEY (Label) REFERENCES Labels_table (ID)
    FOREIGN KEY (Parent_Breeding) REFERENCES Breedings_table (ID)
    FOREIGN KEY (Ai14) REFERENCES Genotypes_table (ID)
    FOREIGN KEY (Ai65) REFERENCES Genotypes_table (ID)
    FOREIGN KEY (Ai75) REFERENCES Genotypes_table (ID)
    FOREIGN KEY (Ai80) REFERENCES Genotypes_table (ID)
    FOREIGN KEY (Ai148) REFERENCES Genotypes_table (ID)
    FOREIGN KEY (Ai162) REFERENCES Genotypes_table (ID)
    FOREIGN KEY (VGC) REFERENCES Genotypes_table (ID)
    FOREIGN KEY (VRC) REFERENCES Genotypes_table (ID)
    FOREIGN KEY (PVF) REFERENCES Genotypes_table (ID)
    FOREIGN KEY (SLF) REFERENCES Genotypes_table (ID)
    FOREIGN KEY (Line) REFERENCES Lines_table (ID)
    FOREIGN KEY (Room) REFERENCES Rooms_table (ID)
    FOREIGN KEY (Experimental_Status) REFERENCES Experimental_table (ID)
    FOREIGN KEY (Genotyping_Status) REFERENCES Genotypings_table (ID)

    );
"""
)
#%%

Sex_table.to_sql('Sex_table', db_conn, if_exists='append', index=False)
Rooms_table.to_sql('Rooms_table', db_conn, if_exists='append', index=False)
Action_types_table.to_sql('Action_types_table', db_conn, if_exists='append', index=False)
Lines_table.to_sql('Lines_table', db_conn, if_exists='append', index=False)
Genes_table.to_sql('Genes_table', db_conn, if_exists='append', index=False)
Genotypes_table.to_sql('Genotypes_table', db_conn, if_exists='append', index=False)
Genotyping_status_table.to_sql('Genotypings_table', db_conn, if_exists='append', index=False)
Breeding_status_table.to_sql('Breeders_table', db_conn, if_exists='append', index=False)
Experimental_status_table.to_sql('Experimental_table', db_conn, if_exists='append', index=False)
Labels_table.to_sql('Labels_table', db_conn, if_exists='append', index=False)
Breedings_table.to_sql('Breedings_table', db_conn, if_exists='append', index=False)
Litters_table.to_sql('Litters_table', db_conn, if_exists='append', index=False)
Actions_table.to_sql('Actions_table', db_conn, if_exists='append', index=False)
Mice_table.to_sql('MICE_table', db_conn, if_exists='append', index=False)







#%%
db_conn = sqlite3.connect(databasefile)
c = db_conn.cursor()
db_conn.close()

#%%

# query3="""
# SELECT 
#     Lab_number,
#     Sex_types, 
#     Cage, 
#     round(julianday('now') - julianday(DOB)) AS DaysOld,
#     round(round(julianday('now') - julianday(DOB))/7) AS WeeksOld,
#     Breeders_types,
#     Labels_types, 
#     Line_Short, 
#     Genotypings_types

# FROM MICE_table 


# LEFT JOIN Breeders_table ON Breeders_table.ID=MICE_table.Breeding_status
# LEFT JOIN Labels_table ON Labels_table.ID=MICE_table.Label
# LEFT JOIN Sex_table ON Sex_table.ID=MICE_table.Sex
# LEFT JOIN Experimental_table ON Experimental_table.ID=MICE_table.Experimental_Status
# LEFT JOIN Genotypings_table ON Genotypings_table.ID=MICE_table.Genotyping_Status
# LEFT JOIN Lines_table ON Lines_table.ID=MICE_table.Line
# LEFT JOIN Rooms_table ON Rooms_table.ID=MICE_table.Room

# WHERE Alive=1 AND (Experimental_Status=1 OR Experimental_Status=3)  AND Breeding_status=3

# """


# Alive_non_exp = pd.read_sql_query(query3,db_conn)

# Alive_non_exp.sort_values(by=['Sex_types'],ascending=False)
# new=    Alive_non_exp.sort_values(by=['Sex_types'],ascending=False)

# gk = new.groupby(['Breeders_types', 'Cage', 'Line_Short'])['Lab_Number'].apply(list)
# zz=gk.to_frame()
# tt=pd.DataFrame(zz.Lab_Number.values.tolist(),zz.index).add_prefix('Mouse_')
# %% changing the breeding mouse codes to indexes

# Breedings_table = pd.read_excel(
#     excelfile, 
#     sheet_name='Breedings',
#     header=0)

# ChangeBreedings=Breedings_table
#%%

#             #%%
# unique=list(ChangeBreedings.Male.unique())
# list_to_replace=unique[:-2]
# list_to_replace.remove(np.nan)
# new_list=[Mice_table._get_value(Mice_table.index[Mice_table['Lab_Number'] == mouse].tolist()[0],  'ID') for  mouse in list_to_replace ]
# ChangeBreedings[['Male']]=ChangeBreedings[['Male']].replace(list_to_replace,new_list)


# unique=list(ChangeBreedings.Female1.unique())
# list_to_replace=unique[:-2]
# list_to_replace.remove(np.nan)
# new_list=[Mice_table._get_value(Mice_table.index[Mice_table['Lab_Number'] == mouse].tolist()[0],  'ID') for  mouse in list_to_replace ]
# ChangeBreedings[['Female1']]=ChangeBreedings[['Female1']].replace(list_to_replace,new_list)

  
# unique=list(ChangeBreedings.Female2.unique())
# list_to_replace=unique[:-2]
# list_to_replace.remove(np.nan)
# new_list=[Mice_table._get_value(Mice_table.index[Mice_table['Lab_Number'] == mouse].tolist()[0],  'ID') for  mouse in list_to_replace ]
# ChangeBreedings[['Female2']]=ChangeBreedings[['Female2']].replace(list_to_replace,new_list)

#%% change action lab number to codes

# Actions_table = pd.read_excel(
#     excelfile, 
#     sheet_name='Actions',
#     header=0)


# unique=list(Actions_table.Mouse_1.unique())
# list_to_replace=unique
# new_list=[Mice_table._get_value(Mice_table.index[Mice_table['Lab_Number'] == mouse].tolist()[0],  'ID') for  mouse in list_to_replace ]
# Actions_table[['Mouse_1']]=Actions_table[['Mouse_1']].replace(list_to_replace,new_list)


# unique=list(Actions_table.Mouse_2.unique())
# list_to_replace=unique
# list_to_replace.pop(9)
# new_list=[Mice_table._get_value(Mice_table.index[Mice_table['Lab_Number'] == mouse].tolist()[0],  'ID') for  mouse in list_to_replace ]
# Actions_table[['Mouse_2']]=Actions_table[['Mouse_2']].replace(list_to_replace,new_list)

# unique=list(Actions_table.Mouse_3.unique())
# list_to_replace=unique
# list_to_replace.pop(3)
# new_list=[Mice_table._get_value(Mice_table.index[Mice_table['Lab_Number'] == mouse].tolist()[0],  'ID') for  mouse in list_to_replace ]
# Actions_table[['Mouse_3']]=Actions_table[['Mouse_3']].replace(list_to_replace,new_list)

# unique=list(Actions_table.Mouse_4.unique())
# list_to_replace=unique
# list_to_replace.pop(2)
# new_list=[Mice_table._get_value(Mice_table.index[Mice_table['Lab_Number'] == mouse].tolist()[0],  'ID') for  mouse in list_to_replace ]
# Actions_table[['Mouse_4']]=Actions_table[['Mouse_4']].replace(list_to_replace,new_list)

# unique=list(Actions_table.Mouse_5.unique())
# list_to_replace=unique
# list_to_replace.pop(0)
# new_list=[Mice_table._get_value(Mice_table.index[Mice_table['Lab_Number'] == mouse].tolist()[0],  'ID') for  mouse in list_to_replace ]
# Actions_table[['Mouse_5']]=Actions_table[['Mouse_5']].replace(list_to_replace,new_list)


# Actions_table['Mouse_2'] = Actions_table['Mouse_2'].astype('Int64')
# Actions_table['Mouse_3'] = Actions_table['Mouse_3'].astype('Int64')
# Actions_table['Mouse_4'] = Actions_table['Mouse_4'].astype('Int64')
# Actions_table['Mouse_5'] = Actions_table['Mouse_5'].astype('Int64')
