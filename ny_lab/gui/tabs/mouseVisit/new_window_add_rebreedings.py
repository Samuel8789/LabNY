# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 15:49:27 2021

@author: sp3660
"""

import tkinter as tk
from tkinter import ttk
from tkinter import *
import tkinter as Tkinter

class new_window_add_rebreedings:
    def __init__(self, app,  male_cages, number_of_rebreedings):
        self.gui=app
        self.separations=male_cages
        self.male_cages=list(male_cages.keys())
        self.tableWindow = tk.Toplevel(self.gui)
        self.number_of_rebreedings=number_of_rebreedings    
        self.sep_boxes=[]
        self.sep_entries=[]
        for i in range(number_of_rebreedings):
            var = StringVar()
            self.sep_boxes.append(ttk.Combobox(self.tableWindow, values=self.male_cages, textvariable=var, width=5))       
            self.sep_boxes[i].var = tk.StringVar()
            self.sep_boxes[i]['textvariable'] = self.sep_boxes[i].var
            self.sep_boxes[i].var.set(self.male_cages[0])
            self.sep_boxes[i].var.trace_add('write', self.update) ## Add
            self.sep_boxes[i].grid(row=i, column=0) 

            self.sep_entries.append(ttk.Label(self.tableWindow, width=20))
            self.sep_entries[i].var = tk.StringVar()           
            self.sep_entries[i]['textvariable'] = self.sep_entries[i].var
            self.sep_entries[i].grid(row=i, column=1)   
    
            
        self.accept_button=ttk.Button(self.tableWindow, text='Accept', command=self.button_accept_rebreedings)
        self.accept_button.grid(row=0, column=3)    
            

    def button_accept_rebreedings(self):
        for i in range(self.number_of_rebreedings): 
            self.gui.MouseDat.readd_male(int(self.sep_entries[i].var.get()), int(self.sep_boxes[i].get()))
        self.destroy() 
        
    def update(self, *a):
        kyes_to_use=list(range(self.number_of_rebreedings))
        for i in range(self.number_of_rebreedings):
            kyes_to_use[i]=self.sep_boxes[i].get()
        for i in range(self.number_of_rebreedings):    
            self.sep_entries[i].var.set(self.separations[kyes_to_use[i]])    
        
if __name__ == "__main__":
    
    breedings_cages_list={'278': 132, '276': 135, '279': 137, '286': 141, '280': 143, '287': 253}
    number_of_rebreedings=2
    root = Tkinter.Tk()
    app = new_window_add_rebreedings(root,  breedings_cages_list, number_of_rebreedings)
    root.mainloop()
    get_values=app.separation_pairs
    print(app.separation_pairs)