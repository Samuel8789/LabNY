# -*- coding: utf-8 -*-
"""
Created on Wed May 12 16:01:28 2021

@author: sp3660
"""

import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter import END, Label, RAISED, Text, WORD, StringVar, Button, Entry


class modify_injection_params(tk.Toplevel):
    def __init__(self, gui, mice_codes, injection_sites, virus_dilutions, cage):
        tk.Toplevel.__init__(self) #instead of super
        

        
        self.selected_codes=mice_codes
        self.virus_dilutions=virus_dilutions
        self.injection_sites=injection_sites
        self.cage=cage
        len(virus_dilutions)
        self.gui=gui
        self.total_rows=len(mice_codes)

        query_brain_areas_info="""SELECT * FROM Brain_Areas_table"""    
        query_coordinates_info="""SELECT * FROM Sterocoordinates_table"""  
        params=()
        self.brain_areas=self.gui.MouseDat.arbitrary_query_to_df(query_brain_areas_info, params).set_index('ID').T.to_dict('list')
        self.coordinates=self.gui.MouseDat.arbitrary_query_to_df(query_coordinates_info, params).set_index('ID').T.to_dict('list')
        self.good_coordinates=[str(element)[1:-1] for element in self.coordinates.values() ]
        self.good_brain_areas=[item for sublist in  list(self.brain_areas.values()) for item in sublist]

        injection_params={'Mouse_Code':'',
                          'Date':'TODO',
                          'Virus1':'',
                          'Virus2':'',
                          'Virus3':'',
                          'Dilution1':'',
                          'Dilution2':'',
                          'Dilution3':'',
                          'CorticalArea':1,
                          'InjectionSites':injection_sites,
                          'Injection1Coordinates':4,
                          'Injection1Volume':500,
                          'Injection1Speeds':50,
                          'Injection1PreTimes':10,
                          'Injection1PostTimes':5,
                          'Injection1GoodVolumes':'Yes',
                          'Injection1Bleeding':'No',
                          'Injection2Coordinates':5,
                          'Injection2Volume':500,
                          'Injection2Speeds':50,
                          'Injection2PreTimes':10,
                          'Injection2PostTimes':5,
                          'Injection2GoodVolumes':'Yes',
                          'Injection2Bleeding':'No',
                          'Notes':''
                          }
        injection_params['Virus1']=virus_dilutions[0][0]
        injection_params['Dilution1']=virus_dilutions[0][1]
        res = list(zip(*virus_dilutions))
        if any(x in res[0] for x in ['B', 'C1V1', 'I']):
            injection_params['Virus3']=virus_dilutions[-1][0]
            injection_params['Dilution3']=virus_dilutions[-1][1]
            if len(virus_dilutions)==3:
                injection_params['Virus2']=virus_dilutions[1][0]
                injection_params['Dilution2']=virus_dilutions[1][1]
        elif len(virus_dilutions)==2:
            injection_params['Virus2']=virus_dilutions[1][0]
            injection_params['Dilution2']=virus_dilutions[1][1]
        
        
        self.values=list()
        self.b = list()
        self.b.append(list())  
        self.values.append(list())
        col_labels=list(injection_params.keys())
        row_defaults=list(injection_params.values())
        self.col_labels=col_labels  
        for j in range(len(col_labels)): #Columns
           self.b[0].append( Label( self, text =col_labels[j], relief=RAISED)) # b[i][j]
           self.b[0][j].grid(row=j, column=1, sticky='e')
           self.values[0].append(col_labels[j])
  
        for i in range(1,self.total_rows+1,1): #Rows
            self.b.append(list())  
            self.values.append(list())
            for j in range(len(col_labels)): #Columns
                if j==1:
                    self.b[i].append(Entry(self, text="", width=10)) # b[i][j]
                    
                elif j==8:     
                    self.b[i].append(ttk.Combobox( self, values=self.good_brain_areas, width=15) ) # b[i][j]
                    self.b[i][j].current(0)
                elif j in [10,17]:
                    self.b[i].append(ttk.Combobox( self, values=self.good_coordinates,  width=30) ) # b[i][j]
                    if j==10:
                        self.b[i][j].current(3)
                    else:
                        self.b[i][j].current(4)
                elif j in [16,23,24]:        
                    self.b[i].append(Text(self, height=3, width=25, wrap=WORD))
                else:
                    self.b[i].append(Entry(self, text="", width=5)) # b[i][j]
                    
                    
                    
                self.b[i][j].grid(row=j, column=i+1, sticky='w')
                if j==0:
                    self.b[i][j].insert(0, mice_codes[i-1])
                    
                elif j>0 and j not in [8,10,17,16,23,24]:
                    self.b[i][j].insert(0, row_defaults[j])
                    
                    
                if j in [16,23,24]:
                    self.values[i].append(self.b[i][j].get("1.0",END))
                else :             
                    self.values[i].append(self.b[i][j].get())   

        enter_button = Button(self, text="Enter", command=self.retrieve_input)
        enter_button.grid(row=1,column=25)


    def retrieve_input(self):
        for i in range(1,self.total_rows+1): #Rows
            for j in range(len(self.col_labels)+1): #Columns            
                if j in [16,23,24]:
                    self.values[i][j]=self.b[i][j].get("1.0",END)
                else :             
                    self.values[i][j]=self.b[i][j].get()      
                
        for i in range(1,len( self.values)):
            self.values[i][8]= self.good_brain_areas.index(self.values[i][8])+1
            self.values[i][10]= self.good_coordinates.index(self.values[i][10])+1
            self.values[i][17]= self.good_coordinates.index(self.values[i][17])+1

        self.destroy()
        self.update()
        self.gui.MouseDat.Experimental_class.plan_new_injection(self.values, self.cage, self.selected_codes, self.injection_sites, self.virus_dilutions)
if __name__ == "__main__":
    root = tk.Tk()
    # app = modify_injection_params(root, mice_codes, injection_sites, virus_dilutions)
    root.mainloop()
    # get_values=app.values
