# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 12:55:41 2021

@author: sp3660
"""

import tkinter as tk
from tkinter import ttk
from tkinter import *
import tkinter as Tkinter
from functools import partial

class new_window_add_weanings:
    def __init__(self, app,  weanings_info, number_of_weanings):
        
  
        self.gui=app
        self.weanings_info=weanings_info
        self.weanings_ids_list=list(weanings_info.keys())
        self.tableWindow = tk.Toplevel(self.gui)
        self.number_of_weanings=number_of_weanings    
        self.sep_boxes=[]
        self.sep_cage_labels=[]
        self.sep_male_number_entries=[]
        self.sep_female_number_entries=[]
        self.sep_male_cage_entries=[]
        self.sep_female_cage_entries=[]
        self.saced=[]
        self.cage=[]
        self.selected_ids=[]
        self.male_number_label=[]
        self.female_number_label=[]
        self.male_cage_label=[]  
        self.female_cage_label=[]
        self.saced_label=[]
        
        for i in range(self.number_of_weanings):
 
            self.cage.append(StringVar(name=str(i)))
            
            
            self.sep_boxes.append(ttk.Combobox(self.tableWindow, values=self.weanings_ids_list, width=5))
            self.sep_boxes[i].var = tk.StringVar()
            self.sep_boxes[i]['textvariable'] = self.sep_boxes[i].var
            self.sep_boxes[i].var.set(self.weanings_ids_list[0])
            self.sep_boxes[i].var.trace_add('write', self.update) ## Add

            
            self.sep_cage_labels.append(ttk.Label(self.tableWindow,  width=5))
            self.sep_cage_labels[i].var=tk.StringVar()
            self.sep_cage_labels[i]['textvariable'] = self.sep_cage_labels[i].var
            
                
            self.male_number_label.append(ttk.Label(self.tableWindow, text='Male Number', width=15))
            self.female_number_label.append(ttk.Label(self.tableWindow, text='Female Number', width=15))
            self.male_cage_label.append(ttk.Label(self.tableWindow, text='Male Cage', width=15))
            self.female_cage_label.append(ttk.Label(self.tableWindow,text='Female Cage', width=15))
            self.saced_label.append(ttk.Label(self.tableWindow, text='Saced Number', width=15))

    
            self.sep_male_number_entries.append(ttk.Entry(self.tableWindow , text='', width=5))
            self.sep_female_number_entries.append(ttk.Entry(self.tableWindow , text='', width=5))
            self.sep_male_cage_entries.append(ttk.Entry(self.tableWindow , text='', width=5))
            self.sep_female_cage_entries.append(ttk.Entry(self.tableWindow , text='', width=5))
            self.saced.append(ttk.Entry(self.tableWindow , text='', width=5))
            
            
            self.sep_boxes[i].grid(row=2*i, column=0)  
            self.sep_cage_labels[i].grid(row=2*i, column=1)   
            self.sep_male_number_entries[i].grid(row=2*i, column=3)   
            self.sep_female_number_entries[i].grid(row=2*i, column=5)  
            self.sep_male_cage_entries[i].grid(row=2*i+1, column=3)  
            self.sep_female_cage_entries[i].grid(row=2*i+1, column=5)  
            self.saced[i].grid(row=2*i+1, column=1)  
            
            self.male_number_label[i].grid(row=2*i, column=2)  
            self.female_number_label[i].grid(row=2*i, column=4)  
            self.male_cage_label[i].grid(row=2*i+1, column=2)  
            self.female_cage_label[i].grid(row=2*i+1, column=4)  
            self.saced_label[i].grid(row=2*i+1, column=0)  
            
        self.accept_button=ttk.Button(self.tableWindow, text='Accept', command=self.button_accept_weanings)
        self.accept_button.grid(row=0, column=6)    
            
        
        
    def update(self, *a):
        kyes_to_use=list(range(self.number_of_weanings))
        for i in range(self.number_of_weanings):
            kyes_to_use[i]=self.sep_boxes[i].get()
        for i in range(self.number_of_weanings):    
            self.sep_cage_labels[i].var.set(self.weanings_info[kyes_to_use[i]])
        
        # self.cage[i].set(self.weanings_info[a[0]][1])
    
    def button_accept_weanings(self):
        for i in range(self.number_of_weanings):    
            self.gui.MouseDat.Add_Weaning(self.gui, int(self.sep_boxes[i].get()),  
                                      int(self.sep_male_number_entries[i].get()), 
                                      int(self.sep_female_number_entries[i].get()), 
                                      int(self.sep_male_cage_entries[i].get()),   
                                      int(self.sep_female_cage_entries[i].get()),
                                      int(self.saced[i].get())) 
        
        self.tableWindow.destroy()  
        
        
if __name__ == "__main__":
    
    weanings_info={'145': 132, '149': 135, '143': 137, '150': 141, '146': 143, '151': 253}
    number_of_weanings=2
    root = Tkinter.Tk()
    app = new_window_add_weanings(root,   weanings_info, number_of_weanings)
    root.mainloop()
    # get_values=app.separation_pairs
    # print(app.separation_pairs)