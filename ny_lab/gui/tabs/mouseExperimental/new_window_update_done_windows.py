# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 15:04:49 2021

@author: sp3660
"""

import tkinter as tk
from tkinter import ttk
from tkinter import *
import tkinter as Tkinter
import datetime

from ....database.fun.guiFunctions.update_window_params import update_window_params


class new_window_update_done_windows(tk.Toplevel):
    def __init__(self, app):
        tk.Toplevel.__init__(self) #instead of super

        self.gui=app
        

        self.window_cage=tk.IntVar()   
        self.window_cage.trace_add('write', self.update) #
        self.cage_label=ttk.Label(self, text='Cage', width=20)
        self.cage_selection=ttk.Combobox(self, values= list(self.gui.MouseDat.Experimental_class.all_mouse_to_do_window['Cage'].unique()), textvariable=self.window_cage, width=30)   
      
        self.mice_in_cage=StringVar()       
        self.mice_label=ttk.Label(self, text='Mice', width=20)
        self.mice_lstbox =Listbox(self, listvariable=self.mice_in_cage, selectmode=MULTIPLE, width=20, height=10, exportselection=0)
     
        self.window_date=tk.StringVar()          
        self.window_date_entry=  ttk.Entry(self , textvariable=self.window_date, width=9)
        self.window_date.set( datetime.date.today().strftime("%Y%m%d"))   
        self.window_date_label=ttk.Label(self, text='window Date', width=20)

        self.update_window_button= ttk.Button(self , text='Update windows', command=self.update_window_button)
        
   
        self.dead_mice_label=ttk.Label(self, text='Dead Mice', width=20)
        self.dead_mice_lstbox =Listbox(self, listvariable=self.mice_in_cage, selectmode=MULTIPLE, width=20, height=10, exportselection=0)
        
        
        
        self.window_date_label.grid(column=0, row=0)
        self.window_date_entry.grid(column=1, row=0)
        
        self.cage_label.grid(column=0, row=1)
        self.cage_selection.grid(column=1, row=1)

        self.mice_label.grid(column=0, row=2)
        self.mice_lstbox.grid(column=1, row=2, columnspan=2)
        
        self.update_window_button.grid(column=0, row=12)
              
        self.dead_mice_label.grid(column=3, row=2)
        self.dead_mice_lstbox.grid(column=4, row=2, columnspan=2)

    def update_window_button(self):
            self.selected_dead_mice=[]
            self.select()
            
            self.destroy()
            updte_window=update_window_params(self.gui, self.window_cage.get(), self.selected_mice, self.window_date.get(), database_object=self.gui.MouseDat)
            updte_window.wait_window()
            print('windows updated')

            if self.selected_dead_mice:
                print('ss')
                for code in self.selected_dead_mice:
                    #here add the surgery date
                  self.gui.MouseDat.Experimental_class.mouse_dead_during_surgery(code, updte_window.date_performed)
        
    def update(self, *a):
        cage=self.window_cage.get()
        mice=self.gui.MouseDat.Experimental_class.all_mouse_to_do_window[self.gui.MouseDat.Experimental_class.all_mouse_to_do_window['Cage']==cage]['Code'].values.tolist()
        self.mice_in_cage.set(mice)

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
    

    root = Tkinter.Tk()
    app = new_window_update_done_windows(root,)
    root.mainloop()