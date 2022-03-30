# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 16:06:20 2022

@author: sp3660
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 11:41:13 2021

@author: sp3660
"""

import tkinter as tk
from tkinter import ttk
from tkinter import StringVar, Listbox, DoubleVar, MULTIPLE


class new_window_move_to_exp_room(tk.Toplevel):
    def __init__(self, app, mousdatabase=False):
        tk.Toplevel.__init__(self) #instead of super

        self.gui=app
        if mousdatabase:
            self.gui.MouseDat=mousdatabase

        self.injection_project=tk.StringVar()   
        self.injection_project.trace_add('write', self.update_cages_by_project) #
        self.project_label=ttk.Label(self, text='Project', width=20)
        self.project_label.grid(column=0, row=0)
        self.project_selection=ttk.Combobox(self, values=self.gui.MouseDat.allprojects.values.tolist(), textvariable=self.injection_project, width=30)   
        self.project_selection.grid(column=1, row=0)
        
        self.project_cages=[]
        self.selected_cage=tk.IntVar() 
        self.cage_label=ttk.Label(self, text='Cage', width=20)
        self.cage_label.grid(column=0, row=1)
        self.cage_selection=ttk.Combobox(self, values=self.project_cages, textvariable=self.selected_cage, width=30)   
        self.cage_selection.grid(column=1, row=1)
        

        self.move_to_exp_button= ttk.Button(self , text='Move To Exp', command=self.move_to_exp_button_fun)
        self.move_to_exp_button.grid(column=1, row=100)
        
 

    def move_to_exp_button_fun(self):

        self.gui.MouseDat.Experimental_class.add_new_planned_experimental_cage(self.selected_cage.get())
        self.destroy()
        

    def update_cages_by_project(self, *a):
       
        project= int(self.injection_project.get()[0])
        if project in (4,5):
             stocks=self.gui.MouseDat.stock_mice[self.gui.MouseDat.stock_mice['Line_Short'].isin(['G2C','G2C::Ai14','G2C::Ai75'])]['Cage'].tolist()
           
             cage_selec=stocks
             cage_selec.sort()
             self.cage_selection['values']=cage_selec
           
             
        if project in (2,3):
            stocks=  self.gui.MouseDat.stock_mice[self.gui.MouseDat.stock_mice['Line_Short'].isin(['VRC::PVF::Ai65','VRC::SLF::Ai65','VRC::PVF','VRC::SLF'])]['Cage'].tolist()  


            cage_selec=stocks
            cage_selec.sort()
            self.cage_selection['values']=cage_selec

        if project ==7:
            stocks=  self.gui.MouseDat.stock_mice[self.gui.MouseDat.stock_mice['Line_Short'].isin(['VGC::Ai162','VGC::Ai148'])]['Cage'].tolist()
            cage_selec=stocks
            cage_selec.sort()
            self.cage_selection['values']=cage_selec

        if project ==8:
            stocks=  self.gui.MouseDat.stock_mice[self.gui.MouseDat.stock_mice['Line_Short'].isin(['Other Colony'])]['Cage'].tolist()
            cage_selec=stocks
            cage_selec.sort()
            self.cage_selection['values']=cage_selec

if __name__ == "__main__":
    
#%%
    root = tk.Tk()
    app = new_window_move_to_exp_room(root ,MouseDat)
    root.mainloop()
