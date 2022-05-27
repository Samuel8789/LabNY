# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 11:41:20 2021

@author: sp3660
"""


import tkinter as tk
from tkinter import ttk
import tkinter as Tkinter
from tkinter import StringVar, Listbox, IntVar, MULTIPLE
import ast

class WidgetSelectStockCageMice(tk.Frame):
        def __init__(self, MouseDat, frame, cage_list, exp_code=False):
            tk.Frame.__init__(self, frame)
            self.exp_code=exp_code
            self.cage_list=cage_list
            self.parent=frame
            self.parent.selected_mice=None

            self.dat_ref=MouseDat
            
            self.cage=IntVar()   
            self.cage.trace_add('write', self.update_mice) #

            self.cage_label=ttk.Label(self, text='Cage', width=20)
            self.cage_label.grid(column=0, row=0)
        
            self.cage_selection=ttk.Combobox(self, values= self.cage_list,  textvariable=self.cage, width=6, postcommand = self.update_lists)
            self.cage_selection.grid(column=0, row=1)

            self.mice_in_cage=StringVar()

            self.mouse_label=ttk.Label(self, text='Stock Mouse', width=20)
            self.mouse_label.grid(column=0, row=2)
                    
            self.mouse_selection=Listbox( self, listvariable=self.mice_in_cage, selectmode=MULTIPLE, width=6, height=8, exportselection=0)
            self.mouse_selection.grid(column=0, row=3)
            
            self.select_button= ttk.Button(self , text='Select Mice', command=self.select_mice_button)
            self.select_button.grid(column=0, row=4)
            
        def select_mice_button(self):
            self.select()
            self.parent.selected_mice=self.selected_mice
            self.parent.selected_cage=self.cage.get()
            print('mice selected')
            
        def update_mice(self, *a):
       
            cage=self.cage.get()
            if not self.exp_code:
                mice=self.dat_ref.allMICE[self.dat_ref.allMICE['Cage']==cage]['Lab_Number'].values.to_numpy().tolist()
            else:
                mice=self.dat_ref.Experimental_class.all_experimental_all_info[self.dat_ref.Experimental_class.all_experimental_all_info['Cage']==cage]['Code'].values.tolist()
            self.mice_in_cage.set(mice)
            
        def select(self):
                self.selected_mice = list()
                selection = self.mouse_selection.curselection()
                if selection:
                    for i in selection:
                        mouse = self.mouse_selection.get(i)
                        self.selected_mice.append(mouse)
                else:
                     self.selected_mice=list(ast.literal_eval( self.mice_in_cage.get()))
                     
        def update_lists(self, updated_cage_list=None):
            if updated_cage_list:
                self.cage_selection['values'] = updated_cage_list 
           
     


                    
if __name__ == "__main__":
   
    from sys import platform
    import socket
    from project_manager.ProjectManager import ProjectManager
    
    house_PC='DESKTOP-V1MT0U5'
    lab_PC='DESKTOP-OKLQSQS'
    # small_laptop_ubuntu='samuel-XPS-13-9560'
    # small_laptop_kali='samuel-XPS-13-9560'
    big_laptop_ubuntu='samuel-XPS-15-9560'
    big_laptop_arch='samuel-XPS-15-9560'
    
    
    
    if platform == "win32":
        if socket.gethostname()==house_PC:
            githubtoken_path=r'C:\Users\Samuel\Documents\Github\GitHubToken.txt'
            computer=house_PC
        elif socket.gethostname()==lab_PC:
            githubtoken_path=r'C:\Users\sp3660\Documents\Github\GitHubToken.txt'
            computer=lab_PC 
    
    
    ProjectManager=ProjectManager(githubtoken_path, computer, platform)
    gui=0
    lab=ProjectManager.initialize_a_project('LabNY', gui)   

    MouseDat=lab.database
    datamanaging=lab.datamanaging
    
    
    root = Tkinter.Tk()
    app = WidgetSelectStockCageMice(MouseDat, root )
    root.mainloop()
    print(root.selected_mice)
    # get_values=app.separation_pairs
    # print(app.separation_pairs)