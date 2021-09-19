# -*- coding: utf-8 -*-
"""
Created on Fri May 14 14:22:18 2021

@author: sp3660
"""

import tkinter as Tkinter
import random
from tkinter import *
import pandas as pd
import datetime

class update_window_params(Tkinter.Toplevel):
    def __init__(self,  gui, cage, codes_selected, date_performed, database_object=None):
        Tkinter.Toplevel.__init__(self, gui)

        query_mice_exp_info="""
        SELECT 
            ExperimentalAnimals_table.ID,
            MICE_table.ID, 
            Lab_number,Code,
            Labels_types,
            Room,
            ExperimentalAnimals_table.Experimental_status,
            MICE_table.Experimental_status, 
            Experiment,
            round(round(julianday('now') - julianday(Injection1Date))/7) AS WeeksFromInjection,
            WindowID ,
            Windows_table.*   
        FROM ExperimentalAnimals_table 
        LEFT JOIN MICE_table ON MICE_table.ID = ExperimentalAnimals_table.Mouse_ID 
        LEFT JOIN Windows_table ON Windows_table.ID = ExperimentalAnimals_table.WindowID 
        LEFT JOIN Labels_table d ON d.ID=Label
        WHERE Cage=? AND ExperimentalAnimals_table.Experimental_status=1 AND ExperimentalAnimals_table.Experiment=4 AND Code IN (%s)""" % ','.join('?' for i in codes_selected)  
        query_mice_exp_info="""
        SELECT 
            ExperimentalAnimals_table.ID,
            MICE_table.ID, 
            Lab_number,Code,
            Labels_types,
            Room,
            ExperimentalAnimals_table.Experimental_status,
            MICE_table.Experimental_status, 
            Experiment,
            round(round(julianday('now') - julianday(Injection1Date))/7) AS WeeksFromInjection,
            WindowID ,
            Windows_table.*   
        FROM ExperimentalAnimals_table 
        LEFT JOIN MICE_table ON MICE_table.ID = ExperimentalAnimals_table.Mouse_ID 
        LEFT JOIN Windows_table ON Windows_table.ID = ExperimentalAnimals_table.WindowID 
        LEFT JOIN Labels_table d ON d.ID=Label
        WHERE Cage=? AND Code IN (%s)""" % ','.join('?' for i in codes_selected)            
        params=(cage,)+tuple(codes_selected)          
        params=(cage,)+tuple(codes_selected)
        mice_exp = database_object.arbitrary_query_to_df(query_mice_exp_info, params).values.tolist()
               
        
        mixe_exp_extended={}
        self.total_rows=len(codes_selected)
        for exp in mice_exp:
            window_params={'Mouse_Code':exp[3],
                              'Date':date_performed,
                              'Labels':exp[4],
                              'CorticalArea':exp[14],
                              'HeadPlateCoordinates':exp[15],
                              'WindowType':exp[16],
                              'CranioSize':exp[17],
                              'CoverType':exp[18],
                              'CoverSize':exp[19],
                              'Durotomy':exp[20],
                              'DamagedAreas':'',
                              'Notes':'',                     
                              }
            corrected_window_params={}
            for key,value in window_params.items():
                if value==None:
                    corrected_window_params[key]=''
                else:
                    corrected_window_params[key]=window_params[key]

            mixe_exp_extended[str(exp[3])]=corrected_window_params
        
        
        
        
        self.values=list()
        self.b = list()
        self.b.append(list())  
        self.values.append(list())
        col_labels=list(mixe_exp_extended[list(mixe_exp_extended.keys())[0]].keys())
        
        for j in range(len(col_labels)): #Columns
           self.b[0].append( Label( self, text =col_labels[j], relief=RAISED)) # b[i][j]
           self.b[0][j].grid(row=1, column=j)
           self.values[0].append(col_labels[j])
        # pair_counter=0   
        for i in range(1,self.total_rows+1,1): #Rows
            self.b.append(list())  
            self.values.append(list())
            row_defaults=list(mixe_exp_extended[list(mixe_exp_extended.keys())[i-1]].values())
            for j in range(len(col_labels)): #Columns
                if j in [1]:
                    self.b[i].append(Entry(self, text="", width=10)) # b[i][j]
                elif j in [10,11]:
                    self.b[i].append(Entry(self, text="", width=40)) # b[i][j]
                else:
                    self.b[i].append(Entry(self, text="", width=5)) # b[i][j]
                    
                self.b[i][j].grid(row=i+1, column=j)
                self.b[i][j].insert(0, row_defaults[j])
                
                
                
                # if not (i % 2) == 0 :
                #     if j==0:
                #         self.b[i][j].insert(0, mice_codes[i-i+pair_counter])
                #     elif j>0:
                #         self.b[i][j].insert(0, row_defaults[j])
                # if (i % 2) == 0 :
                #     if j==0:
                #         self.b[i][j].insert(0, mice_codes[i-i+pair_counter])
                #     elif j>0:
                #         self.b[i][j].insert(0, row_defaults[j]) 
                        
                self.values[i].append(self.b[i][j].get())   
                
            # if (i % 2) == 0 :
            #     pair_counter=pair_counter+1
                        
                


        def retrieve_input():
            for i in range(1,self.total_rows+1,1): #Rows
                for j in range(len(col_labels)): #Columns            
                    self.values[i][j]=self.b[i][j].get()
            self.destroy()
            self.update()
            self.database_object.Experimental_class.update_performed_window(self.values, self.cage, self.selected_codes, self.date_performed )
            
            
        enter_button = Button(self, text="Enter", command=retrieve_input)
        enter_button.grid(row=1,column=26)
        
if __name__ == "__main__":
    cage=125
    selectedanimals=['SPJA','SPJC']
    date_performed=datetime.datetime(2021, 5, 25, 0, 0)
    root = Tkinter.Tk()
    app = update_window_params(root,cage, selectedanimals, date_performed, database_object=MouseDat)
    root.mainloop()
    get_values=app.values
