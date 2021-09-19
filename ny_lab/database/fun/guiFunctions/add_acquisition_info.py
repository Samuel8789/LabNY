# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 08:47:23 2021

@author: sp3660
"""

import tkinter as Tkinter
import random
from tkinter import *
import pandas as pd
import datetime

class add_acquisition_info:
    def __init__(self,  master, acquisiton_path):
        

        master.geometry("+2555+0")

        
        row_labels=['Comments']
        self.total_rows=len(row_labels)+1
        self.values=list()
        self.b = list()
        self.b.append(list())  
        self.values.append(list())      
        col_labels=['Info',acquisiton_path]
    
        for j in range(len(col_labels)): #Columns
           self.b[0].append( Label( master, text =col_labels[j], relief=RAISED)) # b[i][j]
           self.b[0][j].grid(row=1, column=j)
           self.values[0].append(col_labels[j])
        for i in range(1,self.total_rows,1): #Rows
            self.b.append(list())  
            self.values.append(list())
            for j in range(len(col_labels)): #Columns
                if j==0:
                    texts=row_labels[i-1]
                    self.b[i].append(Label(master, text=texts, width=30)) # b[i][j]
                    self.b[i][j].grid(row=i+1, column=j)               
                    self.values[i].append(texts)  
                    
                elif j>0:
                    self.b[i].append(Text(master, height=5, width=150, wrap=WORD)) # b[i][j]
                    self.b[i][j].grid(row=i+1, column=j)                              
                    self.values[i].append(self.b[i][j].get("1.0",END))   
                        
                   
                        
                        
    

        def retrieve_input():
            for i in range(1,self.total_rows,1): #Rows
                    self.values[i][1]=self.b[i][j].get("1.0",END)

            master.destroy()
            
            
            
        enter_button = Button(master, text="Enter", command=retrieve_input)
        enter_button.grid(row=1,column=26)

if __name__ == "__main__":
    
    root = Tkinter.Tk()
    app = add_acquisition_info(root, 'Test')
    root.geometry("+2555+0")
    root.mainloop()
    get_values=app.values