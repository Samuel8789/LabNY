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

import tkinter as tk
from tkinter import ttk
import pandas as pd
from tkinter import  Label, RAISED,Button, Entry, WORD, Text, END

class update_injection_params(tk.Toplevel):
    def __init__(self,  gui, exp_IDs, injection_id, date_performed,cage,selected_codes, database_object=None):
        tk.Toplevel.__init__(self) 
        self.gui=gui
        self.cage=cage
        self.selected_codes=selected_codes
        self.date_performed =date_performed
        self.database_connection=database_object.database_connection
        self.database_object=database_object
        query_check_injeciton_info="""SELECT Code,
                                        EarMark,
                                        Injections_table.* 
                                        FROM Injections_table 
                                        LEFT JOIN ExperimentalAnimals_table ON ExperimentalAnimals_table.Injection1ID=Injections_table.ID
                                        WHERE Injections_table.ID IN(%s)""" % ','.join('?' for i in injection_id)       
        params=injection_id
        mice_exp = pd.read_sql_query(query_check_injeciton_info, self.database_connection,params=params).values.tolist()          
        virus_comb=[i[5] for i in mice_exp]
        query_check_virus_from_virus_comb="""SELECT
                                                Combination, 
                                                Virus1_table.VirusCode, 
                                                Virus2_table.VirusCode,
                                                Virus3_table.VirusCode 
                                                FROM VirusCombinations_table 
                                                LEFT JOIN Virus_table AS Virus1_table  ON  Virus1_table.ID = VirusCombinations_table.Virus1 
                                                LEFT JOIN Virus_table AS Virus2_table  ON  Virus2_table.ID = VirusCombinations_table.Virus2 
                                                LEFT JOIN Virus_table AS Virus3_table  ON  Virus3_table.ID = VirusCombinations_table.Virus3 
                                                WHERE VirusCombinations_table.ID IN(%s)""" % ','.join('?' for i in virus_comb)
        viruses = pd.read_sql_query(query_check_virus_from_virus_comb, self.database_connection,params=tuple(virus_comb,)).values.tolist()          
   
        query_brain_areas_info="""SELECT * FROM Brain_Areas_table"""    
        query_coordinates_info="""SELECT * FROM Sterocoordinates_table"""  
        query_labels_info="""SELECT * FROM Labels_table"""  

        params=()
        self.brain_areas=database_object.arbitrary_query_to_df(query_brain_areas_info, params).set_index('ID').T.to_dict('list')
        self.coordinates=database_object.arbitrary_query_to_df(query_coordinates_info, params).set_index('ID').T.to_dict('list')
        self.good_coordinates=[str(element)[1:-1] for element in self.coordinates.values() ]
        self.good_brain_areas=[item for sublist in  list(self.brain_areas.values()) for item in sublist]
        self.labels=database_object.arbitrary_query_to_df(query_labels_info, params).set_index('ID').T.to_dict('list')
        self.good_labels=[item for sublist in  list(self.labels.values()) for item in sublist]
        
        
        mixe_exp_extended={}
        self.total_rows=len(exp_IDs)
        for exp in mice_exp: 
           
            injection_params={'Mouse_Code':exp[0],
                              'Label':'',
                              'Date':date_performed,
                              'Virus1':viruses[0][1],
                              'Virus2':viruses[0][2],
                              'Virus3':viruses[0][3],
                              'Dilution1':exp[6],
                              'Dilution2':exp[7],
                              'Dilution3':exp[8],
                              'CorticalArea':'',
                              'InjectionSites':exp[10],
                              'Injection1Coordinates':'',
                              'Injection1Volume':exp[12],
                              'Injection1Speeds':exp[13],
                              'Injection1PreTimes':exp[14],
                              'Injection1PostTimes':exp[15],
                              'Injection1GoodVolumes':exp[16],
                              'Injection1Bleeding':exp[17],
                              'Injection2Coordinates':'',
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
        col_labels.append('MovedRoom')
        self.col_labels=col_labels
        for j in range(len(col_labels)): #Columns
           self.b[0].append( Label( self, text =col_labels[j], relief=RAISED)) # b[i][j]
           self.b[0][j].grid(row=j, column=1, sticky = 'e')

           self.values[0].append(col_labels[j])
        for i in range(1,self.total_rows+1,1): #Rows
            self.b.append(list())  
            self.values.append(list())
            row_defaults=list(mixe_exp_extended[list(mixe_exp_extended.keys())[i-1]].values())
            for j in range(len(col_labels)): #Columns
                if j in [2]:
                    self.b[i].append(Entry(self, text="", width=10)) # b[i][j]
                elif j in [17,24,25]:
                    # self.b[i].append(Entry(self, text="", width=50)) # b[i][j]                   
                    self.b[i].append(Text(self, height=3, width=25, wrap=WORD)) # b[i][j]
                    
                elif j==1:
                    self.b[i].append(ttk.Combobox( self, values= self.good_labels, width=15) ) # b[i][j]
                    self.b[i][j].current(mice_exp[i-1][1]-1)
                elif j==9:
                    self.b[i].append(ttk.Combobox( self, values= self.good_brain_areas, width=20) ) # b[i][j]
                    self.b[i][j].current(mice_exp[i-1][9]-1)

                elif j in [11,18]:
                    self.b[i].append(ttk.Combobox( self, values=self.good_coordinates,  width=25) ) # b[i][j]
                    if j==11:
                        self.b[i][j].current(mice_exp[i-1][11]-1)
                    elif j==18:
                        self.b[i][j].current(mice_exp[i-1][18]-1)
                        
                elif j==26:
                    self.b[i].append(ttk.Combobox( self, values=[0,1]) ) # b[i][j]
                    self.b[i][j].current(0)

                else:
                    self.b[i].append(Entry(self, text="", width=5)) # b[i][j]
                    
                self.b[i][j].grid(row=j, column=i+1,sticky = 'w')

                if j not in [1,9,11,18,17,24,25,26]:
                    self.b[i][j].insert(0, row_defaults[j]) 
                
                if j not in [17,24,25]    :
                    self.values[i].append(self.b[i][j].get())  
                else:
                    self.values[i].append(self.b[i][j].get("1.0",END))

        enter_button = Button(self, text="Enter", command=self.retrieve_input)
        enter_button.grid(row=1,column=26)



    def retrieve_input(self):
        for i in range(1,self.total_rows+1): #Rows
            for j in range(len(self.col_labels)): #Columns            
                if j not in [17,24,25]:
                    self.values[i][j]=self.b[i][j].get()  
                else:
                    self.values[i][j]=self.b[i][j].get("1.0",END)
                
        for i in range(1,len( self.values)):
            self.values[i][9]= self.good_brain_areas.index(self.values[i][9])+1
            self.values[i][11]= self.good_coordinates.index(self.values[i][11])+1
            self.values[i][18]= self.good_coordinates.index(self.values[i][18])+1


        self.destroy()
        self.update()
        self.database_object.Experimental_class.update_performed_injections(self.values, self.cage, self.selected_codes, self.date_performed )


if __name__ == "__main__":
    
    tests=[(243, 244), (153, 154)]
    date_performed='20210420'
    #%%
    root = tk.Tk()

    app = update_injection_params(root, tests[0], tests[1], date_performed, 305,['SPGL'],  database_object=MouseDat)
    root.mainloop()
    #%%
    get_values=app.values
