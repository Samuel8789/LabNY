# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 15:50:58 2021

@author: sp3660
"""



import tkinter as tk
from tkinter import ttk
import tkinter as Tkinter
from tkinter import Listbox, MULTIPLE, END
import ast

class WidgetSelectMultiCage(tk.Frame):
        def __init__(self, MouseDat, frame, cage_list):
            tk.Frame.__init__(self, frame)
            
            self.cage_list=cage_list
            self.parent=frame
            self.dat_ref=MouseDat
               
            self.cages=tk.StringVar()
            self.cage_selection=Listbox(self, listvariable=self.cages, selectmode=MULTIPLE, width=6, height=8, exportselection=0)
            self.cage_selection.grid(column=0, row=3)
            self.cages.set(self.cage_list)
        
       
            self.select_button= ttk.Button(self , text='Select Cages', command=self.select_cage_button)
            self.select_button.grid(column=0, row=4)
            
        def select_cage_button(self):
            self.select()
            self.parent.selected_cages=self.selected_cages
            print('mice selected')
      
        def select(self):
                self.selected_cages = list()
                selection = self.cage_selection.curselection()
                if selection:
                    for i in selection:
                        cage = self.cage_selection.get(i)
                        self.selected_cages.append(cage)
                else:
                     self.selected_cages=list(ast.literal_eval( self.cages.get()))
                     
        def update_lists(self, updated_cage_list=None):
            if updated_cage_list:
                self.cage_selection.delete(0, END)
                for cage in updated_cage_list: #populate listbox again
                    self.cage_selection.insert(END, cage)
                


                    
if __name__ == "__main__":
   
    from sys import platform
    import socket
    from project_manager.ProjectManager import ProjectManager
    
    house_PC='DESKTOP-V1MT0U5'
    lab_PC='DESKTOP-NBGKRCG'
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
    app = WidgetSelectMultiCage(MouseDat, root )
    root.mainloop()
    print(root.selected_mice)
    # get_values=app.separation_pairs
    # print(app.separation_pairs)