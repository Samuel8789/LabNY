# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 11:50:50 2021

@author: sp3660
"""
import tkinter as tk
from tkinter import ttk
from tkinter import *
import tkinter as Tkinter

class new_window_add_separations:
    def __init__(self, app,  breedings_cages_list, number_of_separations):
        self.gui=app
        self.breedings_cages_list=breedings_cages_list
        self.tableWindow = tk.Toplevel(self.gui)
        self.number_of_separations=number_of_separations    
        self.sep_boxes=[]
        self.sep_entries=[]
        for i in range(number_of_separations):
            var = StringVar()
            self.sep_boxes.append(ttk.Combobox(self.tableWindow, values=breedings_cages_list, textvariable=var, width=5))
            self.sep_boxes[i].grid(row=i, column=0)   
    
            self.sep_entries.append(ttk.Entry(self.tableWindow , text='', width=5))
            self.sep_entries[i].grid(row=i, column=1)   
    
            
        self.accept_button=ttk.Button(self.tableWindow, text='Accept', command=self.button_accept_separations)
        self.accept_button.grid(row=0, column=3)    
            

    def button_accept_separations(self):
        for i in range(self.number_of_separations): 
            self.gui.MouseDat.Separate_male(int(self.sep_boxes[i].get()),int(self.sep_entries[i].get()),commit=True)
        
        self.tableWindow.destroy()  
        
        
if __name__ == "__main__":
    
    breedings_cages_list=['133',
 '134',
 '136',
 '139',
 '140',
 '142',
 '145',
 '146',
 '190',
 '191',
 '192',
 '252']
    number_of_separations=2
    root = Tkinter.Tk()
    app = new_window_add_separations(root,  breedings_cages_list, number_of_separations)
    root.mainloop()
    get_values=app.separation_pairs
    print(app.separation_pairs)
 