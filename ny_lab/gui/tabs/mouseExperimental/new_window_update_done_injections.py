# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 14:03:21 2021

@author: sp3660
"""


import tkinter as tk
from tkinter import ttk
from tkinter import *
import tkinter as Tkinter
import datetime
from ....database.fun.guiFunctions.update_injection_params import update_injection_params
from ....database.fun.guiFunctions.plan_window_parameters import plan_window_parameters



class new_window_update_done_injections(tk.Toplevel):
    def __init__(self, app):
        tk.Toplevel.__init__(self) #instead of super
        
        self.gui=app
        self.selected_mice = list()
        self.selected_dead_mice=[]

        self.injection_cage=tk.IntVar()   
        self.injection_cage.trace_add('write', self.update) #
        self.cage_label=ttk.Label(self, text='Cage', width=20)
        self.cage_selection=ttk.Combobox(self, values= list(self.gui.MouseDat.Experimental_class.all_mouse_to_do_injection['Cage'].unique()), textvariable=self.injection_cage, width=30)   
      
        self.mice_in_cage=StringVar()       
        self.mice_label=ttk.Label(self, text='Mice', width=20)
        self.mice_lstbox =Listbox(self, listvariable=self.mice_in_cage, selectmode=MULTIPLE, width=20, height=10, exportselection=0)
     
        self.injection_date=tk.StringVar()          
        self.injection_date_entry=  ttk.Entry(self , textvariable=self.injection_date, width=9)
        self.injection_date.set( datetime.date.today().strftime("%Y%m%d"))   
        self.injection_date_label=ttk.Label(self, text='Injection Date', width=20)

        self.update_injection_button= ttk.Button(self , text='Update Injections', command=self.update_injection_button)
        
   
        self.dead_mice_label=ttk.Label(self, text='Dead Mice', width=20)
        self.dead_mice_lstbox =Listbox(self, listvariable=self.mice_in_cage, selectmode=MULTIPLE, width=20, height=10, exportselection=0)
        
        
        
        self.injection_date_label.grid(column=0, row=0)
        self.injection_date_entry.grid(column=1, row=0)
        
        self.cage_label.grid(column=0, row=1)
        self.cage_selection.grid(column=1, row=1)

        self.mice_label.grid(column=0, row=2)
        self.mice_lstbox.grid(column=1, row=2, columnspan=2)
        
        self.update_injection_button.grid(column=0, row=12)
              
        self.dead_mice_label.grid(column=3, row=2)
        self.dead_mice_lstbox.grid(column=4, row=2, columnspan=2)

    def update_injection_button(self):
            self.select()
            
       
            query_mice_exp_info="SELECT ExperimentalAnimals_table.ID, MICE_table.ID, Lab_number,Code,Label,Room,ExperimentalAnimals_table.Experimental_status,MICE_table.Experimental_status, Experiment,Injection1ID FROM ExperimentalAnimals_table LEFT JOIN MICE_table ON MICE_table.ID = ExperimentalAnimals_table.Mouse_ID WHERE Cage=?"
            params=(self.injection_cage.get(),)
            mice_exp=self.gui.MouseDat.arbitrary_query_to_df(query_mice_exp_info,params).values.tolist()  
            good_mice_exp=[i for i in mice_exp if i[3] in self.selected_mice]
            res = list(zip(*good_mice_exp))  
            exp_and_injection_ids=[res[0], res[-1]] 
            self.destroy()

            self.update_injection_parameter_window=update_injection_params(self.gui, exp_and_injection_ids[0], exp_and_injection_ids[1], self.injection_date.get(), self.injection_cage.get(),
                                    self.selected_mice, database_object=self.gui.MouseDat)
            
            self.update_injection_parameter_window.wait_window()
            
            print('injections updated')
            if self.selected_dead_mice:
                print('ss')
                for code in self.selected_dead_mice:
                  self.gui.MouseDat.Experimental_class.mouse_dead_during_surgery(code)
                  
            self.plan_window_parameters_window=plan_window_parameters(self.gui, mice_codes=self.selected_mice)
            self.plan_window_parameters_window.wait_window()
            print('windows planned')
            
           


        
    def update(self, *a):
        cage=self.injection_cage.get()
        mice=self.gui.MouseDat.Experimental_class.all_mouse_to_do_injection[self.gui.MouseDat.Experimental_class.all_mouse_to_do_injection['Cage']==cage]['Code'].values.tolist()
        self.mice_in_cage.set(mice)

    def select(self):
       
        selection = self.mice_lstbox.curselection()
        for i in selection:
            mouse = self.mice_lstbox.get(i)
            self.selected_mice.append(mouse)
            
        self.selected_dead_mice = list()
        selection_dead = self.dead_mice_lstbox.curselection()
        for i in selection_dead:
            mouse_dead = self.dead_mice_lstbox.get(i)
            self.selected_dead_mice.append(mouse_dead)     
            
            

    
if __name__ == "__main__":
    

    root = Tkinter.Tk()
    app = new_window_update_done_injections(root,)
    root.mainloop()
