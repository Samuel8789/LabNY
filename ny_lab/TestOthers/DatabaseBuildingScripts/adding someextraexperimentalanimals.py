# -*- coding: utf-8 -*-
"""
Created on Tue May  4 09:55:09 2021

@author: sp3660
"""

# adding_experimentals_extra
import sys
sys.path.insert(0, r'C:/Users/sp3660/Documents/Github/LabNY/AllFunctions')

import pandas as pd
import numpy as np
import sqlite3
from sqlite3 import Error
from select_values_gui import select_values_gui


databasefile=r'C:\Users\sp3660\Desktop\MouseDatabase.db'
db_conn = sqlite3.connect(databasefile)
c = db_conn.cursor()
excelfile=r'C:\Users\sp3660\Documents\Projects\LabNY\5. Surgeries\Mouse Surgeries.xlsx'

#%%
Rooms_table = pd.read_excel(
    excelfile, 
    sheet_name='Sheet2',
    header=0)


Rooms_table['DOB'].dt.date

#%% adding missing experimental to database

experimental_to_add=Rooms_table['Lab_Number'].values.tolist()
sex=Rooms_table['Sex'].values.tolist()
dob=Rooms_table['DOB'].dt.date.tolist()

label=Rooms_table['Label'].values.tolist()
ai162=Rooms_table['Ai162'].values.tolist()
vGC=Rooms_table['VGC'].values.tolist()
line=Rooms_table['Line'].values.tolist()
genotyping=Rooms_table['Genotyping_Status'].values.tolist()
notes=Rooms_table['Notes'].values.tolist()

new_exp_mice= ''' INSERT INTO MICE_table( Lab_Number,Sex,Breeding_status,DOB,Label,Parent_Breeding,Ai14,Ai65,Ai75,Ai80,Ai148,Ai162,G2C,VGC,VRC,PVF,SLF,Line,Room,Experimental_Status,Genotyping_Status,Alive)
          VALUES(?,?,3,?,?,158,3,3,3,3,3,?,3,?,3,3,3,?,1,8,?,0) '''


for i, mouse in enumerate(experimental_to_add):
        full_values=(mouse,sex[i],dob[i],label[i],ai162[i],vGC[i],line[i],genotyping[i])
        c.execute(new_exp_mice,full_values)

db_conn.commit()     


allmice="""SELECT * FROM MICE_table  WHERE  Experimental_Status=8 AND ( Line=24 OR Line=17)"""
allmicetable=pd.read_sql_query(allmice, db_conn)
ttt=allmicetable['Lab_Number'].values.tolist()
#%% correct somestuff
for i, mouse in enumerate(experimental_to_add):
    updatemouse="UPDATE MICE_table SET DOB=datetime(?), Notes=? WHERE Lab_Number=?"
    update_values=(dob[i],notes[i],mouse)
    c.execute(updatemouse, update_values)
    db_conn.commit()



