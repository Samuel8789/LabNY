# -*- coding: utf-8 -*-
"""
Created on Fri May 28 09:37:45 2021

@author: sp3660
"""


import tkinter as Tkinter
import random
from tkinter import *
import pandas as pd
import datetime

class add_imagedmouse_info:
    def __init__(self,  master, session_date, mouse_code,):
        
        
     
        master.geometry("+2555+0")
        
        
        
        colname=session_date+'\\'+mouse_code
        row_labels=['TimeSetOnWheel',
                    'EyesComments',
                    'BehaviourComments',
                    'FurComments',
                    'LessionComments',
                    'DexaInjection',
                    'BehaviourProtocol',
                    'BehaviourProtocolComments',
                    'OptogeneticsProtocol',
                    'OptogeneticsProtocolComments',
                    'Objectives',
                    'EndOfSessionSummary',
                    'IssuesDuringImaging',]
        self.total_rows=len(row_labels)+1
        self.values=list()
        self.b = list()
        self.b.append(list())  
        self.values.append(list())      
        col_labels=['Info',colname]
    
        for j in range(len(col_labels)): #Columns
           self.b[0].append( Label( master, text =col_labels[j], relief=RAISED)) # b[i][j]
           self.b[0][j].grid(row=1, column=j)
           self.values[0].append(col_labels[j])
        # pair_counter=0   
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
                    if i in [1,2,3,4,5,8,10,11,12,13]:
                        self.b[i].append(Text(master,height=5, width=150, wrap=WORD)) # b[i][j]
                        self.b[i][j].grid(row=i+1, column=j)                              
                        self.values[i].append(self.b[i][j].get("1.0",END))   
                        
                    elif i in [6,7,9]:
                        if i==6:
                            values=['No','Yes']
                        elif i==7:
                            values=[ 'SpontaneousOnly',
                                'SimpleVisStim',
                                'Habituation',
                                'Mistmatch',
                                'FullAllen',
                                'None']
                        elif i==9:

                            values=['None']
                            
                        var = StringVar()
                        self.b[i].append(Tkinter.ttk.Combobox(master, values=values,textvariable=var)) # b[i][j]
                        self.b[i][j].grid(row=i+1, column=j)   
                        self.b[i][j].current(0)     
                        self.values[i].append(self.b[i][j].get())   

  

        def retrieve_input():
            for i in range(1,self.total_rows,1): #Rows      
                if i in [1,2,3,4,5,8,10,11,12,13]:
                    self.values[i][1]=self.b[i][j].get("1.0",END)
                elif i in [6,7,9]:
                    self.values[i][1]=self.b[i][j].get()
                    
            master.destroy()
            
            
            
        enter_button = Button(master, text="Enter", command=retrieve_input)
        enter_button.grid(row=1,column=26)

if __name__ == "__main__":
    
    root = Tkinter.Tk()
    app = add_imagedmouse_info(root, '20210523', 'SPJO', 29, MouseDat)
    root.mainloop()
    get_values=app.values
