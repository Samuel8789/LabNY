# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 08:56:28 2021

@author: sp3660
"""

import tkinter as Tkinter
import random
from tkinter import *
import pandas as pd
import datetime

class add_facecamera_info:
    def __init__(self,  master, face_camera_path):
        master.geometry("+2555+0")
     
        
        row_labels=['IsIRlight','IRLightPosition','CameraPosition','SideImaged','SynchronizeMethods', 'Comments']
        self.total_rows=len(row_labels)+1
        self.values=list()
        self.b = list()
        self.b.append(list())  
        self.values.append(list())      
        col_labels=['Info',face_camera_path]
    
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
                    if i not in [1,2,3,4,5]:
                        self.b[i].append(Text(master, height=5, width=150, wrap=WORD)) # b[i][j]
                        self.b[i][j].grid(row=i+1, column=j)                              
                        self.values[i].append(self.b[i][j].get("1.0",END))   
                        
                    elif i==2:
                        values=['Left lateral over lamp'] 
                        var = StringVar()
                        self.b[i].append(Tkinter.ttk.Combobox(master, values=values,textvariable=var, width=30)) # b[i][j]
                        self.b[i][j].grid(row=i+1, column=j)   
                        self.b[i][j].current(0)     
                        self.values[i].append(self.b[i][j].get())   
                        
                    elif i==3:
                        values=['Front left Lateral ','Front left Lateral. Looking Down'] 
                        var = StringVar()
                        self.b[i].append(Tkinter.ttk.Combobox(master, values=values,textvariable=var, width=30)) # b[i][j]
                        self.b[i][j].grid(row=i+1, column=j)   
                        self.b[i][j].current(0)     
                        self.values[i].append(self.b[i][j].get())   
                    elif i==4:
                        values=['Left','Right'] 
                        var = StringVar()
                        self.b[i].append(Tkinter.ttk.Combobox(master, values=values,textvariable=var, width=30)) # b[i][j]
                        self.b[i][j].grid(row=i+1, column=j)   
                        self.b[i][j].current(0)     
                        self.values[i].append(self.b[i][j].get())   
                    elif i==1:
                        values=['Yes','No'] 
                        var = StringVar()
                        self.b[i].append(Tkinter.ttk.Combobox(master, values=values,textvariable=var, width=30)) # b[i][j]
                        self.b[i][j].grid(row=i+1, column=j)   
                        self.b[i][j].current(0)     
                        self.values[i].append(self.b[i][j].get())  
                    elif i==5:
                        values=['None. Cant synch with locomotion','LaserLight','Led flash'] 
                        var = StringVar()
                        self.b[i].append(Tkinter.ttk.Combobox(master, values=values,textvariable=var, width=30)) # b[i][j]
                        self.b[i][j].grid(row=i+1, column=j)   
                        self.b[i][j].current(0)     
                        self.values[i].append(self.b[i][j].get())  
                           
                        
    

        def retrieve_input():
            for i in range(1,self.total_rows,1): #Rows
                if i not in [1,2,3,4,5]:
                    self.values[i][1]=self.b[i][j].get("1.0",END)
                elif i in [1,2,3,4,5]:
                    self.values[i][1]=self.b[i][j].get()
            master.destroy()
            
            
            
        enter_button = Button(master, text="Enter", command=retrieve_input)
        enter_button.grid(row=1,column=26)

if __name__ == "__main__":
    
    root = Tkinter.Tk()
    app = add_facecamera_info(root, 'Test')

    root.mainloop()
    get_values=app.values