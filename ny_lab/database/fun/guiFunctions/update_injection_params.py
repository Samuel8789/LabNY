# -*- coding: utf-8 -*-
"""
Created on Thu May 13 08:47:09 2021

@author: sp3660
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 12 16:01:28 2021

@author: sp3660
"""

import tkinter as Tkinter
import random
from tkinter import *
import tkinter as tk
import pandas as pd
import datetime

class update_injection_params(tk.Toplevel):
    def __init__(self,  gui, exp_IDs, injection_id, date_performed,cage, selected_codes, database_object=None):
        tk.Toplevel.__init__(self) 
        
        self.cage=cage
        self.date_performed =date_performed
        self.selected_codes=selected_codes
        self.database_connection=database_object.database_connection
        self.database_object=database_object
        query_check_injeciton_info="SELECT Code, EarMark, Injections_table.* FROM Injections_table LEFT JOIN ExperimentalAnimals_table ON ExperimentalAnimals_table.Injection1ID=Injections_table.ID WHERE Injections_table.ID=? OR Injections_table.ID=? OR Injections_table.ID=? OR Injections_table.ID=? OR Injections_table.ID=?"
        
        injection_id_list=list(injection_id)
        if len(injection_id_list)<5:
            dif=5-len(injection_id_list)
            for i in range(dif):
                injection_id_list.append(injection_id_list[-1])
        
        mice_exp = pd.read_sql_query(query_check_injeciton_info, self.database_connection,params=tuple(injection_id_list,)).values.tolist()          

 
        virus_comb=[i[5] for i in mice_exp]
        if len(virus_comb)<5:
           dif=5-len(virus_comb)
           for i in range(dif):
               virus_comb.append(virus_comb[-1])
        
        
        query_check_virus_from_virus_comb="SELECT Combination, Virus1_table.VirusCode, Virus2_table.VirusCode, Virus3_table.VirusCode FROM VirusCombinations_table LEFT JOIN Virus_table AS Virus1_table  ON  Virus1_table.ID = VirusCombinations_table.Virus1 LEFT JOIN Virus_table AS Virus2_table  ON  Virus2_table.ID = VirusCombinations_table.Virus2 LEFT JOIN Virus_table AS Virus3_table  ON  Virus3_table.ID = VirusCombinations_table.Virus3 WHERE VirusCombinations_table.ID=? OR  VirusCombinations_table.ID=? OR  VirusCombinations_table.ID=? OR  VirusCombinations_table.ID=? OR  VirusCombinations_table.ID=? "
        viruses = pd.read_sql_query(query_check_virus_from_virus_comb, self.database_connection,params=tuple(virus_comb,)).values.tolist()          


        mixe_exp_extended={}
        self.total_rows=len(exp_IDs)
        for exp in mice_exp:
            injection_params={'Mouse_Code':exp[0],
                              'Label':exp[1],
                              'Date':date_performed,
                              'Virus1':viruses[0][1],
                              'Virus2':viruses[0][2],
                              'Virus3':viruses[0][3],
                              'Dilution1':exp[6],
                              'Dilution2':exp[7],
                              'Dilution3':exp[8],
                              'CorticalArea':exp[9],
                              'InjectionSites':exp[10],
                              'Injection1Coordinates':exp[11],
                              'Injection1Volume':exp[12],
                              'Injection1Speeds':exp[13],
                              'Injection1PreTimes':exp[14],
                              'Injection1PostTimes':exp[15],
                              'Injection1GoodVolumes':exp[16],
                              'Injection1Bleeding':exp[17],
                              'Injection2Coordinates':exp[18],
                              'Injection2Volume':exp[19],
                              'Injection2Speeds':exp[20],
                              'Injection2PreTimes':exp[21],
                              'Injection2PostTimes':exp[22],
                              'Injection2GoodVolumes':exp[23],
                              'Injection2Bleeding':exp[24],
                              'Notes':exp[27]
                              }
            corrected_injectio_params={}
            for key,value in injection_params.items():
                if value==None:
                    corrected_injectio_params[key]=''
                else:
                    corrected_injectio_params[key]=injection_params[key]

            mixe_exp_extended[str(exp[3])]=corrected_injectio_params
    
            
            
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
                if j in [2]:
                    self.b[i].append(Entry(self, text="", width=10)) # b[i][j]
                elif j in [17,24,25]:
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
            self.database_object.Experimental_class.update_performed_injections(self.values, self.cage, self.selected_codes, self.date_performed )

            
            
        enter_button = Button(self, text="Enter", command=retrieve_input)
        enter_button.grid(row=1,column=26)

if __name__ == "__main__":
    
    tests=[(243, 244), (153, 154)]
    date_performed='20210420'
    root = Tkinter.Tk()
    app = update_injection_params(root, tests[0], tests[1], date_performed, database_object=MouseDat)
    root.mainloop()
    get_values=app.values
