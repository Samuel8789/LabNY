# -*- coding: utf-8 -*-
"""
Created on Thu May 13 14:45:12 2021

@author: sp3660
"""

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
import pandas as pd
import datetime
import tkinter as tk

class plan_window_parameters(tk.Toplevel):
    def __init__(self,  gui, mice_codes):
        tk.Toplevel.__init__(self, gui)

        self.selected_codes=mice_codes

        self.total_rows=len(mice_codes)

        injection_params={'Mouse_Code':'',
                          'Date':'TODO',
                          'CorticalArea':1,
                          'HeadPlateCoordinates':6,
                          'WindowType':3,
                          'CranioSize':3,
                          'CoverType':2,
                          'CoverSize':3,
                          'Durotomy':1,
                          }
        
        
        
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
            for i in range(1,self.total_rows+1,1): #Rows
                for j in range(len(col_labels)): #Columns            
                    self.values[i][j]=self.b[i][j].get()
                    

            self.destroy()
            self.update()
            self.database_object.Experimental_class.plan_new_window(self.values, codes_selected=self.selected_codes)

            
            
        enter_button = Button(self, text="Enter", command=retrieve_input)
        enter_button.grid(row=1,column=26)

if __name__ == "__main__":
    
    mice_info_list=[]
    root = Tkinter.Tk()
    app = plan_window_parameters(root, mice_info_list)
    root.mainloop()
    get_values=app.values
