# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 09:15:39 2021

@author: sp3660
"""

import tkinter as Tkinter
import random
from tkinter import *
import pandas as pd
import datetime

class add_visstim_info:
    def __init__(self,  master, vistim_path):
        

        master.geometry("+2555+0")
       
        
        row_labels=['RedFilter','GreenFilter','DichroicBeamsplitter','IsBlockingDichroic','ExcitationWavelength','CoherentPower', 'Comments']
        self.total_rows=len(row_labels)+1
        self.values=list()
        self.b = list()
        self.b.append(list())  
        self.values.append(list())      
        col_labels=['Info',vistim_path]
    
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
                    if i not in [1,2,3,4]:
                        self.b[i].append(Text(master, height=5, width=150, wrap=WORD)) # b[i][j]
                        self.b[i][j].grid(row=i+1, column=j)                              
                        self.values[i].append(self.b[i][j].get("1.0",END))   
                        
                    elif i==1:
                        values=['HQ525/70 M-2P 243899','BrightLine Fluorescent Filter 500/24','S510/20 XP 269755','D510/40 M 5633'] 
                        var = StringVar()
                        self.b[i].append(Tkinter.ttk.Combobox(master, values=values,textvariable=var, width=30)) # b[i][j]
                        self.b[i][j].grid(row=i+1, column=j)   
                        self.b[i][j].current(0)     
                        self.values[i].append(self.b[i][j].get())   
                        
                    elif i==2:
                        values=['HQ607/45 M-2P 243869','BA570-620HQ F3F0024'] 
                        var = StringVar()
                        self.b[i].append(Tkinter.ttk.Combobox(master, values=values,textvariable=var, width=30)) # b[i][j]
                        self.b[i][j].grid(row=i+1, column=j)   
                        self.b[i][j].current(0)     
                        self.values[i].append(self.b[i][j].get())   
                    elif i==3:
                        values=['Standard',] 
                        var = StringVar()
                        self.b[i].append(Tkinter.ttk.Combobox(master, values=values,textvariable=var, width=30)) # b[i][j]
                        self.b[i][j].grid(row=i+1, column=j)   
                        self.b[i][j].current(0)     
                        self.values[i].append(self.b[i][j].get())   
                    elif i==4:
                        values=['Yes','No'] 
                        var = StringVar()
                        self.b[i].append(Tkinter.ttk.Combobox(master, values=values,textvariable=var, width=30)) # b[i][j]
                        self.b[i][j].grid(row=i+1, column=j)   
                        self.b[i][j].current(0)     
                        self.values[i].append(self.b[i][j].get())  
                        
                        
    

        def retrieve_input():
            for i in range(1,self.total_rows,1): #Rows
                if i not in [1,2,3,4]:
                    self.values[i][1]=self.b[i][j].get("1.0",END)
                elif i in [1,2,3,4]:
                    self.values[i][1]=self.b[i][j].get()
            master.destroy()
            
            
            
        enter_button = Button(master, text="Enter", command=retrieve_input)
        enter_button.grid(row=1,column=26)

if __name__ == "__main__":
    
    root = Tkinter.Tk()
    app = add_visstim_info(root, 'Test')
    root.mainloop()
    get_values=app.values
	
	
