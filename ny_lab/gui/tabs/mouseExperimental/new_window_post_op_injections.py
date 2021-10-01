# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 16:08:08 2021

@author: sp3660
"""

import tkinter as tk
from tkinter import ttk
from tkinter import StringVar, Listbox, MULTIPLE
import datetime
import distutils.util
from ...utils import button_update_database

class new_window_post_op_injections(tk.Toplevel):
    def __init__(self, app):
        tk.Toplevel.__init__(self) #instead of super  
        self.gui=app
        
        self.post_op_date=tk.StringVar()  
        self.post_op_date_entry=  ttk.Entry(self , textvariable=self.post_op_date, width=9)     
        self.post_op_date_entry.bind('<Return>', self.update_cages_by_date)
        self.post_op_date.set(datetime.date.today().strftime("%Y%m%d"))   
        self.post_op_date_label=ttk.Label(self, text='Post Op Date', width=20)

        self.post_op_cage=tk.IntVar()   
        self.post_op_cage.trace_add('write', self.update_mice_in_cage) #
        self.cage_label=ttk.Label(self, text='Cage', width=20)
        self.cage_selection=ttk.Combobox(self, values= list(self.gui.gui_ref.MouseDat.Experimental_class.all_mouse_to_do_postop_injection['Cage'].unique()), textvariable=self.post_op_cage, width=30)   
        
        self.post_op_skipped=tk.StringVar()   
        self.skipped_label=ttk.Label(self, text='Skipped?', width=20)
        self.skipped_selection=ttk.Combobox(self, values= [True, False], textvariable=self.post_op_skipped, width=30)   
        
      
        self.mice_in_cage=StringVar()       
        self.mice_label=ttk.Label(self, text='Mice', width=20)
        self.mice_lstbox =Listbox(self, listvariable=self.mice_in_cage, selectmode=MULTIPLE, width=20, height=10, exportselection=0)

        self.update_post_op_button= ttk.Button(self , text='Update Post Ops', command=self.update_post_op_button)
          
        self.dead_mice_label=ttk.Label(self, text='Dead Mice', width=20)
        self.dead_mice_lstbox =Listbox(self, listvariable=self.mice_in_cage, selectmode=MULTIPLE, width=20, height=10, exportselection=0)
        
        self.post_op_date_label.grid(column=0, row=0)
        self.post_op_date_entry.grid(column=1, row=0)
        
        self.cage_label.grid(column=0, row=1)
        self.cage_selection.grid(column=1, row=1)

        self.mice_label.grid(column=0, row=2)
        self.mice_lstbox.grid(column=1, row=2, columnspan=2)
        
        self.update_post_op_button.grid(column=0, row=12)
              
        self.dead_mice_label.grid(column=3, row=2)
        self.dead_mice_lstbox.grid(column=4, row=2, columnspan=2)
        
        self.skipped_label.grid(column=0, row=3)
        self.skipped_selection.grid(column=1, row=3)
        

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
            print('post op updated')
            
    def update_mice_in_cage(self, *a):
        cage=self.post_op_cage.get()
        mice=self.gui.gui_ref.MouseDat.Experimental_class.all_mouse_to_do_postop_injection[self.gui.gui_ref.MouseDat.Experimental_class.all_mouse_to_do_postop_injection['Cage']==cage]['Code'].values.tolist()
        self.mice_in_cage.set(mice)
        
    def update_cages_by_date(self, *a):
        date=self.post_op_date.get()
        # date='20210902'
        self.gui.gui_ref.MouseDat.Experimental_class.to_do_postop_injection(date)
        button_update_database(self.gui.gui_ref)
        self.cage_selection['values']=list(self.gui.gui_ref.MouseDat.Experimental_class.all_mouse_to_do_postop_injection['Cage'].unique())
        print('post op updated')
    def select(self):
        self.selected_mice = list()
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
    

    root = tk.Tk()
    app = new_window_post_op_injections(root,)
    root.mainloop()