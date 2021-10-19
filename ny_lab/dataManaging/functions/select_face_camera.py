# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 10:09:57 2021

@author: sp3660
"""

import tkinter as tk
import random
from tkinter import *
from tkinter import ttk
import pandas as pd
import datetime
import os

class select_face_camera(tk.Toplevel):
    def __init__(self,  gui, aquisition_path, UnprocessedFaceCameraspaths, UnprocessedVisStimpaths):
        tk.Toplevel.__init__(self, gui) #inst     
      
        row_labels=['FaceCamera', 'VisStim']
        self.total_rows=len(row_labels)+1
        self.values=list()
        self.b = list()
        self.b.append(list())  
        self.values.append(list())      
        col_labels=['Extra Data',os.path.split(aquisition_path)[1]]
    
        for j in range(len(col_labels)): #Columns
           self.b[0].append( Label( self, text =col_labels[j], relief=RAISED)) # b[i][j]
           self.b[0][j].grid(row=1, column=j)
           self.values[0].append(col_labels[j])
        for i in range(1,self.total_rows,1): #Rows
            self.b.append(list())  
            self.values.append(list())
            for j in range(len(col_labels)): #Columns
                if j==0:
                    texts=row_labels[i-1]
                    self.b[i].append(Label(self, text=texts, width=70)) # b[i][j]
                    self.b[i][j].grid(row=i+1, column=j)               
                    self.values[i].append(texts)  
                    
                elif j>0:
                    if i==1:
                        values=UnprocessedFaceCameraspaths
                        var = StringVar()
                        self.b[i].append(ttk.Combobox(self, values=values,textvariable=var, width=70)) # b[i][j]
                        self.b[i][j].grid(row=i+1, column=j)   
                        self.b[i][j].current(0)     
                        self.values[i].append(self.b[i][j].get())   
                    if i==2:
                        values=UnprocessedVisStimpaths
                        var = StringVar()
                        self.b[i].append(ttk.Combobox(self, values=values,textvariable=var, width=70)) # b[i][j]
                        self.b[i][j].grid(row=i+1, column=j)   
                        self.b[i][j].current(0)     
                        self.values[i].append(self.b[i][j].get())   
 
 
        enter_button = Button(self, text="Enter", command=self.retrieve_input)
        enter_button.grid(row=1,column=26)
        
        
    def retrieve_input(self):
        for i in range(1,self.total_rows,1): #Rows  
               self.values[i][1]=self.b[i][1].get()
        self.destroy()
        self.update()         
         

if __name__ == "__main__":
    
    root = tk.Tk()
    app = select_face_camera(root, os.path.split(aq_path)[1],UnprocessedFaceCameraspaths,UnprocessedVisStimpaths )
    root.mainloop()
    get_values=app.values