# -*- coding: utf-8 -*-
"""
Created on Tue May 11 11:17:26 2021

@author: sp3660
"""
import tkinter as Tkinter
import random
from tkinter import *
import pandas as pd
import tkinter as tk
import numpy as np
from tkinter import ttk


class UpdateLitterInput(tk.Toplevel):
    def __init__(self, app, mousedat=False, litter_number=False):       
        tk.Toplevel.__init__(self, app)
        self.gui=app
        self.mousedat=mousedat
        self.breeding_cages=self.mousedat.breedings['Cage'].values.tolist()
        self.breeding_IDs=self.mousedat.breedings['ID'].values.tolist()
        

        litter_df=pd.DataFrame()
        if not litter_number:
            rows= len(self.mousedat.current_litters.index) 
            litter_df=self.mousedat.current_litters
            self.breeding_cages=litter_df['Cage'].values.tolist()
        else:
            rows=litter_number
            
        self.rows=rows
        
        
        if litter_df.empty:
            rsliced=litter_df
            self.col_labels=['Cage','BreedingID','Alive','Dead','Age']

        else:
            rsliced=litter_df[['ID','Cage','NumberAlive','NumberDead','DaysOld']].reset_index(drop=True)
            self.col_labels=['Cage','LiterID','Alive','Dead','Age']

        self.rsliced=rsliced
        
        self.values=list()
        self.b = list()
        self.b.append(list())  
        self.values.append(list())
        for j in range(len(self.col_labels)): #Columns
           self.b[0].append( Label( self, text =self.col_labels[j], relief=RAISED)) # b[i][j]
           self.b[0][j].grid(row=1, column=j)
           self.values[0].append(self.col_labels[j])
           
        for i in range(1,rows+1): #Rows
            self.b.append(list())  
            self.values.append(list())
            for j in range(len(self.col_labels)): #Columns
                if j==0:

                    if rsliced.empty:
                        self.b[i].append(ttk.Combobox(self, values=self.breeding_cages, width=5))       
                        self.b[i][j].var = tk.StringVar()
                        self.b[i][j]['textvariable'] = self.b[i][j].var
                        self.b[i][j].grid(row=i+1, column=j)
                        self.b[i][j].var.trace_add('write', self.update) ## Add
                        self.b[i][j].current(0)  
                        self.values[i].append(self.b[i][j].get())

                    else:
                        self.b[i].append(ttk.Label(self))
                        self.b[i][j].var = tk.StringVar()           
                        self.b[i][j]['textvariable'] = self.b[i][j].var
                        self.b[i][j].grid(row=i+1, column=j)
                        
                        self.b[i][j].var.set(str(rsliced['Cage'][i-1]))
                        self.values[i].append(self.b[i][j].cget('text'))

                        
                        

                elif j==1:
                    self.b[i].append(ttk.Label(self))
                    self.b[i][j].var = tk.StringVar()           
                    self.b[i][j]['textvariable'] = self.b[i][j].var
                    self.b[i][j].grid(row=i+1, column=j)
                    
                    if rsliced.empty:
                        self.b[i][j].var.set('')
                    else:
                       self.b[i][j].var.set(str(rsliced.iloc[i-1,j-1]))
                       
                    self.values[i].append(self.b[i][j].cget('text'))

                else:
                    self.b[i].append(Entry(self, text="", width=5)) # b[i][j]
                    self.b[i][j].grid(row=i+1, column=j)

                    if rsliced.empty:
                        self.b[i][j].insert(0, 0)
                    else:
                        self.b[i][j].insert(0, rsliced.iloc[i-1,j])
                    
                    self.values[i].append(self.b[i][j].get())
        
        enter_button = Button(self, text="Enter", command=self.retrieve_input)
        enter_button.grid(row=1,column=5)


    def update(self, *a):
        
        for i in range(1,self.rows+1):
            index=self.breeding_cages.index(int(self.b[i][0].get()))
            self.breeding_IDs[index]
            self.b[i][1].var.set(str(self.breeding_IDs[index]))    
            

    def retrieve_input(self):
        for i in range(1,self.rows+1): #Rows
            for j in range(len(self.col_labels)): #Columns  
                if not self.rsliced.empty:
                    if j in [0,1]:
                        self.values[i][j]=int(float(self.b[i][j].cget("text")))
                    else:
                        self.values[i][j]=int(float(self.b[i][j].get()))
                else:
                    if j==1:
                        self.values[i][j]=int(float(self.b[i][j].cget("text")))
                    else:
                        self.values[i][j]=int(float(self.b[i][j].get()))
                    
        self.mousedat.update_old_litters(self.values)
        self.destroy()

if __name__ == "__main__":
    root = Tkinter.Tk()
    #%%
    app = UpdateLitterInput(root, MouseDat)
    root.mainloop()
    get_values=app.values
