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

class UpdateLitterInput(tk.Toplevel):
    def __init__(self, app, mousedat):
        
        tk.Toplevel.__init__(self, gui)
        self.gui=app
        self.mousedat=mousedat     
        rows= len(self.mousedat.current_litters.index)
        litter_df= self.mousedat.current_litters
        
        if litter_df.empty:
            rsliced=litter_df
        else:
            rsliced=litter_df[['ID','Cage','NumberAlive','NumberDead','DaysOld']].reset_index(drop=True)
        
        
        self.values=list()
        self.b = list()
        self.b.append(list())  
        self.values.append(list())
        col_labels=['BreedingID','Cage','Alive','Dead','Age']
        for j in range(len(col_labels)): #Columns
           self.b[0].append( Label( self, text =col_labels[j], relief=RAISED)) # b[i][j]
           self.b[0][j].grid(row=1, column=j)
           self.values[0].append(col_labels[j])
           
        for i in range(1,rows+1,1): #Rows
            self.b.append(list())  
            self.values.append(list())
            for j in range(len(col_labels)): #Columns
                self.b[i].append(Entry(self, text="", width=5)) # b[i][j]
                self.b[i][j].grid(row=i+1, column=j)
                if rsliced.empty:
                    self.b[i][j].insert(0, 0)
                else:
                    self.b[i][j].insert(0, rsliced.iloc[i-1,j])
                    
                self.values[i].append(self.b[i][j].get())


        def retrieve_input():
            for i in range(1,rows+1,1): #Rows
                for j in range(len(col_labels)): #Columns            
                    self.values[i][j]=self.b[i][j].get()
                    
            self.mousedat.update_old_litters(self.values)
            self.destroy()
            
            
            
        enter_button = Button(self, text="Enter", command=retrieve_input)
        enter_button.grid(row=1,column=5)

if __name__ == "__main__":
    root = Tkinter.Tk()
    app = App(root, len(currentlitter.index), currentlitter)
    root.mainloop()
    get_values=app.values
