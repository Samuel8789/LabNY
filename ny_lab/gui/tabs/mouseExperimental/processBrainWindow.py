# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 10:44:26 2021

@author: sp3660
"""

import tkinter as Tkinter
# import random
from tkinter import *
from tkinter import Label, RAISED, Entry, Button

# import pandas as pd
# import datetime
import tkinter as tk

class ProcessBrain(tk.Toplevel):
    def __init__(self,  gui, mice_codes, date_performed):
        tk.Toplevel.__init__(self, gui)
        self.gui=gui
        self.selected_codes=mice_codes
        self.date_performed=date_performed
        self.total_rows=len(mice_codes)

        brain_extraction_parameters={'Mouse_Code':'',
                                     'Date_performed':'',
                                      'Perfusion_solution':'PFA 4%',
                                      'Postfixation_solution':'PFA 4%',
                                      'Postfixation_time':'O/N',
                                      'Postfixation_temperature':'4ºC',
                                      'Prehistology_storage_solution':'PBS Azide 0.05%',
                                      'Prehistology_storage_location':4,
                                      'Comments':''
                                      
                          }
        

        self.values=list()
        self.b = list()
        self.b.append(list())  
        self.values.append(list())
        self.col_labels=list(brain_extraction_parameters.keys())
        row_defaults=list(brain_extraction_parameters.values())
    
        for j in range(len( self.col_labels)): #Columns
           self.b[0].append( Label( self, text = self.col_labels[j], relief=RAISED)) # b[i][j]
           self.b[0][j].grid(row=1, column=j)
           self.values[0].append( self.col_labels[j])
        for i in range(1,self.total_rows+1,1): #Rows
            self.b.append(list())  
            self.values.append(list())
            for j in range(len( self.col_labels)): #Columns

                self.b[i].append(Entry(self, text="", width=20)) # b[i][j]
                self.b[i][j].grid(row=i+1, column=j)
                if j==0:
                    self.b[i][j].insert(0, mice_codes[i-1])
                elif j==1:
                    self.b[i][j].insert(0, date_performed)
 
                elif j>1:
                    self.b[i][j].insert(0, row_defaults[j])
 
                self.values[i].append(self.b[i][j].get())   
                
        enter_button = Button(self, text="Process Brains", command=self.process_brains)
        enter_button.grid(row=1,column=26)

    def process_brains(self):
        for i in range(1,self.total_rows+1,1): #Rows
            for j in range(len(self.col_labels)): #Columns            
                self.values[i][j]=self.b[i][j].get()

        self.destroy()
        self.update()
        self.gui.MouseDat.Experimental_class.brain_fixation(self.values, self.selected_codes, self.date_performed)

        

if __name__ == "__main__":
    
    mice_codes=['SPJT', 'SPJU']
    root = Tkinter.Tk()
    app = ProcessBrain(root, mice_codes)
    root.mainloop()
    get_values=app.values
