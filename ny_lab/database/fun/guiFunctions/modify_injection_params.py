# -*- coding: utf-8 -*-
"""
Created on Wed May 12 16:01:28 2021

@author: sp3660
"""

import tkinter as Tkinter
import tkinter as tk

import random
from tkinter import *
import pandas as pd
import datetime

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
                
        for j in range(len(col_labels)): #Columns
           self.b[0].append( Label( self, text =col_labels[j], relief=RAISED)) # b[i][j]
           self.b[0][j].grid(row=1, column=j)
           self.values[0].append(col_labels[j])
        pair_counter=0   
        for i in range(1,self.total_rows+1,1): #Rows
            self.b.append(list())  
            self.values.append(list())
            for j in range(len(col_labels)): #Columns
                if j==1:
                    self.b[i].append(Entry(self, text="", width=10)) # b[i][j]
                else:
                    self.b[i].append(Entry(self, text="", width=5)) # b[i][j]
                    
                self.b[i][j].grid(row=i+1, column=j)
                if j==0:
                    self.b[i][j].insert(0, mice_codes[i-1])
                elif j>0:
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
            for i in range(1,self.total_rows,1): #Rows
                for j in range(4): #Columns            
                    self.values[i][j]=self.b[i][j].get()
                    
                    
                    
                    
            self.destroy()
            self.update()
            self.gui.MouseDat.Experimental_class.plan_new_injection(self.values, self.cage, self.selected_codes, self.injection_sites, self.virus_dilutions)
            
            
        enter_button = Button(self, text="Enter", command=retrieve_input)
        enter_button.grid(row=1,column=25)

if __name__ == "__main__":
    root = Tkinter.Tk()
    app = modify_injection_params(root, mice_codes, injection_sites, virus_dilutions)
    root.mainloop()
    get_values=app.values
