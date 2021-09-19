# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 16:08:08 2021

@author: sp3660
"""

import tkinter as tk
from tkinter import ttk
from tkinter import *
import tkinter as Tkinter
import datetime
import distutils.util

class new_window_post_op_injections:
    def __init__(self, app):
        
        self.gui=app
        self.tableWindow = tk.Toplevel(self.gui)
        

        self.post_op_date=tk.StringVar()  
        # self.post_op_date.trace_add('write', self.update2) #
        self.tableWindow.post_op_date_entry=  ttk.Entry(self.tableWindow , textvariable=self.post_op_date, width=9)     
        self.tableWindow.post_op_date_entry.bind('<Return>', self.update2)
        self.post_op_date.set(datetime.date.today().strftime("%Y%m%d"))   
        self.tableWindow.post_op_date_label=ttk.Label(self.tableWindow, text='Post Op Date', width=20)

      


        self.post_op_cage=tk.IntVar()   
        self.post_op_cage.trace_add('write', self.update) #
        self.tableWindow.cage_label=ttk.Label(self.tableWindow, text='Cage', width=20)
        self.tableWindow.cage_selection=ttk.Combobox(self.tableWindow, values= list(self.gui.gui_ref.MouseDat.Experimental_class.all_mouse_to_do_postop_injection['Cage'].unique()), textvariable=self.post_op_cage, width=30)   
        
        self.post_op_skipped=tk.StringVar()   
        self.tableWindow.skipped_label=ttk.Label(self.tableWindow, text='Skipped?', width=20)
        self.tableWindow.skipped_selection=ttk.Combobox(self.tableWindow, values= [True, False], textvariable=self.post_op_skipped, width=30)   
        
      
        self.mice_in_cage=StringVar()       
        self.tableWindow.mice_label=ttk.Label(self.tableWindow, text='Mice', width=20)
        self.tableWindow.mice_lstbox =Listbox(self.tableWindow, listvariable=self.mice_in_cage, selectmode=MULTIPLE, width=20, height=10, exportselection=0)

        self.tableWindow.update_post_op_button= ttk.Button(self.tableWindow , text='Update Post Ops', command=self.update_post_op_button)
          
        self.tableWindow.dead_mice_label=ttk.Label(self.tableWindow, text='Dead Mice', width=20)
        self.tableWindow.dead_mice_lstbox =Listbox(self.tableWindow, listvariable=self.mice_in_cage, selectmode=MULTIPLE, width=20, height=10, exportselection=0)
        
        
        
        
        
        
        self.tableWindow.post_op_date_label.grid(column=0, row=0)
        self.tableWindow.post_op_date_entry.grid(column=1, row=0)
        
        self.tableWindow.cage_label.grid(column=0, row=1)
        self.tableWindow.cage_selection.grid(column=1, row=1)

        self.tableWindow.mice_label.grid(column=0, row=2)
        self.tableWindow.mice_lstbox.grid(column=1, row=2, columnspan=2)
        
        self.tableWindow.update_post_op_button.grid(column=0, row=12)
              
        self.tableWindow.dead_mice_label.grid(column=3, row=2)
        self.tableWindow.dead_mice_lstbox.grid(column=4, row=2, columnspan=2)
        
        self.tableWindow.skipped_label.grid(column=0, row=3)
        self.tableWindow.skipped_selection.grid(column=1, row=3)
        
        
        
        
        
        
        
        
        
               
#         cage=[212]
# mice=['SPKI','SPKL']
# date_inject='20210819'
# #%%
# #%%
# MouseDat.Experimental_class.update_postop_injection(cage, mice )
# #%%
# MouseDat.Experimental_class.update_postop_injection(cage, mice, date_performed=date_inject)
# #%%
# MouseDat.Experimental_class.update_postop_injection(cage, mice, date_performed=date_inject,ignored=True)


# #%% dead postop
# ExperimentalDat.mouse_dead_postop('SPJL')
        
        
        
        

    def update_post_op_button(self):
            self.selected_dead_mice=[]
            self.select()
            self.gui.gui_ref.MouseDat.Experimental_class.update_postop_injection([self.post_op_cage.get()], 
                                                                         self.selected_mice, date_performed=self.post_op_date.get(),
                                                                         ignored=bool(distutils.util.strtobool(self.post_op_skipped.get())))
            if self.selected_dead_mice:
                print('ss')
                for code in self.selected_dead_mice:
                  self.gui.gui_ref.MouseDat.Experimental_class.mouse_dead_postop(code)
        
    def update(self, *a):
        cage=self.post_op_cage.get()
        mice=self.gui.gui_ref.MouseDat.Experimental_class.all_mouse_to_do_postop_injection[self.gui.gui_ref.MouseDat.Experimental_class.all_mouse_to_do_postop_injection['Cage']==cage]['Code'].values.tolist()
        self.mice_in_cage.set(mice)
        
    def update2(self, *a):
        date=self.post_op_date.get()
        # date='20210902'
        self.gui.gui_ref.MouseDat.Experimental_class.to_do_postop_injection(date)
        self.gui.button_update_database()
        self.tableWindow.cage_selection['values']=list(self.gui.gui_ref.MouseDat.Experimental_class.all_mouse_to_do_postop_injection['Cage'].unique())

    def select(self):
        self.selected_mice = list()
        selection = self.tableWindow.mice_lstbox.curselection()
        for i in selection:
            mouse = self.tableWindow.mice_lstbox.get(i)
            self.selected_mice.append(mouse)
            
        self.selected_dead_mice = list()
        selection_dead = self.tableWindow.dead_mice_lstbox.curselection()
        for i in selection_dead:
            mouse_dead = self.tableWindow.dead_mice_lstbox.get(i)
            self.selected_dead_mice.append(mouse_dead)     
            
            

    
if __name__ == "__main__":
    

    root = Tkinter.Tk()
    app = new_window_post_op_injections(root,)
    root.mainloop()