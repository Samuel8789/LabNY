# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 11:41:13 2021

@author: sp3660
"""

import tkinter as tk
from tkinter import ttk
from tkinter import StringVar, Listbox, DoubleVar, MULTIPLE
from ....database.fun.guiFunctions.modify_injection_params import modify_injection_params


class new_window_plan_injections(tk.Toplevel):
    def __init__(self, app):
        tk.Toplevel.__init__(self) #instead of super

        self.gui=app
        
        self.injection_project=tk.StringVar()   
        self.injection_project.trace_add('write', self.update_cages_by_project) #
        self.project_label=ttk.Label(self, text='Project', width=20)
        self.project_label.grid(column=0, row=0)
        self.project_selection=ttk.Combobox(self, values=self.gui.MouseDat.allprojects.values.tolist(), textvariable=self.injection_project, width=30)   
        self.project_selection.grid(column=1, row=0)
        
  
        self.project_cages=[]
        self.selected_cage=tk.IntVar()
        self.selected_cage.trace_add('write', self.update_mice_in_cage) ## Add        
        self.cage_label=ttk.Label(self, text='Cage', width=20)
        self.cage_label.grid(column=0, row=1)
        self.cage_selection=ttk.Combobox(self, values=self.project_cages, textvariable=self.selected_cage, width=30)   
        self.cage_selection.grid(column=1, row=1)
        


        self.mice_in_cage=StringVar()       
        self.mice_label=ttk.Label(self, text='Mice', width=20)
        self.mice_label.grid(column=0, row=2)
        self.mice_lstbox =Listbox(self, listvariable=self.mice_in_cage, selectmode=MULTIPLE, width=20, height=10, exportselection=0)
        self.mice_lstbox.grid(column=1, row=2, columnspan=1)

        
        self.number_of_injections=tk.IntVar()          
        self.number_of_injections_entry=  ttk.Entry(self , textvariable=self.number_of_injections, width=5)
        self.number_of_injections_entry.grid(column=0, row=4)
        self.number_of_injections.set(2)   
        self.number_of_injections_label=ttk.Label(self, text='Number of Injections', width=20)
        self.number_of_injections_label.grid(column=0, row=3)
        
       
        
        full=list(zip(self.gui.MouseDat.allVirusstock['VirusCode'].tolist(),  self.gui.MouseDat.allVirusstock['VirusName'].tolist()))
        
        
        self.gui.MouseDat.allVirusstock[self.gui.MouseDat.allVirusstock['Recombinase']==3][['VirusCode', 'VirusName']].values.tolist()

        self.virus1_names=[ virus_name[0]+': '+virus_name[1] for  virus_name in self.gui.MouseDat.allVirusstock[['VirusCode', 'VirusName']].values.tolist()]
        self.virus2_names=[ virus_name[0]+': '+virus_name[1] for  virus_name in self.gui.MouseDat.allVirusstock[self.gui.MouseDat.allVirusstock['Recombinase']==3][['VirusCode', 'VirusName']].values.tolist()]

        self.virus3_names=[virus_name[0]+': '+virus_name[1] for virus_name in self.gui.MouseDat.allVirusstock[(self.gui.MouseDat.allVirusstock['ID']==2) | 
                                                                                                              (self.gui.MouseDat.allVirusstock['ID']==4) | 
                                                                                                              (self.gui.MouseDat.allVirusstock['ID']==10)| 
                                                                                                              (self.gui.MouseDat.allVirusstock['ID']==21)|
                                                                                                              (self.gui.MouseDat.allVirusstock['ID']==27)][['VirusCode', 'VirusName']].values.tolist()]
        
        
        
        self.virus1_label=ttk.Label(self, text='Virus 1', width=20)
        self.virus1_label.grid(column=0, row=5)
        self.virus2_label=ttk.Label(self, text='Virus 2', width=20)
        self.virus2_label.grid(column=0, row=6)
        self.virus3_label=ttk.Label(self, text='Virus 3', width=20)
        self.virus3_label.grid(column=0, row=7)
        self.vir1=tk.StringVar()     
        self.vir2=tk.StringVar() 
        self.vir3=tk.StringVar() 
        self.virus1_label_selection=ttk.Combobox(self, values=self.virus1_names, textvariable=self.vir1, width=50)   
        self.virus1_label_selection.grid(column=1, row=5)
        self.virus1_label_selection=ttk.Combobox(self, values=self.virus2_names, textvariable=self.vir2, width=50)   
        self.virus1_label_selection.grid(column=1, row=6)
        self.virus1_label_selection=ttk.Combobox(self, values=self.virus3_names, textvariable=self.vir3, width=50)   
        self.virus1_label_selection.grid(column=1, row=7)
        
        self.virus1_dilution_label=ttk.Label(self, text='Dilution Virus 1', width=20)
        self.virus1_dilution_label.grid(column=2, row=5)
        self.virus2_dilution_label=ttk.Label(self, text='Dilution Virus 2', width=20)
        self.virus2_dilution_label.grid(column=2, row=6)
        self.virus3_dilution_label=ttk.Label(self, text='Dilution Virus 3', width=20)
        self.virus3_dilution_label.grid(column=2, row=7)
        self.virus1_dilution=DoubleVar()
        self.virus2_dilution=DoubleVar()
        self.virus3_dilution=DoubleVar()
        self.virus1_dilution_entry=ttk.Entry(self , textvariable=self.virus1_dilution, width=5)
        self.virus1_dilution_entry.grid(column=3, row=5)
        self.virus2_dilution_entry=ttk.Entry(self , textvariable=self.virus2_dilution, width=5)
        self.virus2_dilution_entry.grid(column=3, row=6)
        self.virus3_dilution_entry=ttk.Entry(self , textvariable=self.virus3_dilution, width=5)
        self.virus3_dilution_entry.grid(column=3, row=7)
        self.virus1_dilution.set(0.5)
        self.virus2_dilution.set(0.5)
        self.virus3_dilution.set(0.2)
        
        
        self.plan_injection_button_but= ttk.Button(self , text='Plan Injections', command=self.plan_injection_button)
        self.plan_injection_button_but.grid(column=1, row=100)
        
 

    def plan_injection_button(self):
            self.vir1code=self.gui.MouseDat.allVirusstock[self.gui.MouseDat.allVirusstock['VirusName']==self.vir1.get()[3:]]['VirusCode'].tolist()
            if len(self.vir1code)>1:
                self.vir1code= [self.vir1code[-1]]
            self.vir2code=self.gui.MouseDat.allVirusstock[self.gui.MouseDat.allVirusstock['VirusName']==self.vir2.get()[3:]]['VirusCode'].tolist()
            self.vir3code=self.gui.MouseDat.allVirusstock[self.gui.MouseDat.allVirusstock['VirusName']==self.vir3.get()[3:]]['VirusCode'].tolist()
            self.vir1dil=self.virus1_dilution.get()
            self.vir2dil=self.virus2_dilution.get()
            self.vir3dil=self.virus3_dilution.get()
            self.select()
            self.VirusCombination=[self.vir1code,self.vir2code,self.vir3code]
            self.Dilutions=[self.vir1dil,self.vir2dil,self.vir3dil]
            self.virus_dilutions=[(vir[0], self.Dilutions[idx])  for idx, vir in  enumerate(self.VirusCombination)  if vir]
            

            self.destroy()

            self.modify_injection_window=modify_injection_params(self.gui, self.selected_mice, self.number_of_injections.get(),  self.virus_dilutions, self.selected_cage.get())
            self.modify_injection_window.wait_window()

        
    def update_mice_in_cage(self, *a):
        cage=self.selected_cage.get()
        if cage not in  self.gui.main_tabs['Mouse Experimental'].experimental_recoveryanimalsbetter['Cage'].values.to_numpy().tolist():
            mice=self.gui.MouseDat.allMICE[self.gui.MouseDat.allMICE['Cage']==cage]['Lab_Number'].values.to_numpy().tolist()
        else:
            mice=  self.gui.main_tabs['Mouse Experimental'].experimental_recoveryanimalsbetter[self.gui.main_tabs['Mouse Experimental'].experimental_recoveryanimalsbetter['Cage']==cage][self.gui.main_tabs['Mouse Experimental'].experimental_recoveryanimalsbetter['Experiment']=='Planned']['Code'].tolist()
        self.mice_in_cage.set(mice)
        
    def update_cages_by_project(self, *a):
       
        project= int(self.injection_project.get()[0])
        if project in (4,5):
             stocks=self.gui.MouseDat.stock_mice[self.gui.MouseDat.stock_mice['Line_Short'].isin(['G2C','G2C::Ai14','G2C::Ai75'])]['Cage'].tolist()
             exp_siblings=self.gui.main_tabs['Mouse Experimental'].experimental_recoveryanimalsbetter[self.gui.main_tabs['Mouse Experimental'].experimental_recoveryanimalsbetter['Line_Short'].isin(
                 ['G2C','G2C::Ai14','G2C::Ai75']
                 )][self.gui.main_tabs['Mouse Experimental'].experimental_recoveryanimalsbetter['Experiment']=='Planned']['Cage'].tolist()
             cage_selec=stocks+list(set(exp_siblings))
             cage_selec.sort()
             self.cage_selection['values']=cage_selec
           
             
        if project in (2,3):
            stocks=  self.gui.MouseDat.stock_mice[self.gui.MouseDat.stock_mice['Line_Short'].isin(['VRC::PVF::Ai65','VRC::SLF::Ai65','VRC::PVF','VRC::SLF'])]['Cage'].tolist()  

            exp_siblings=  self.gui.main_tabs['Mouse Experimental'].experimental_recoveryanimalsbetter[self.gui.main_tabs['Mouse Experimental'].experimental_recoveryanimalsbetter['Line_Short'].isin(['VRC::PVF::Ai65  ','VRC::SLF::Ai65','VRC::PVF','VRC::SLF'])][self.gui.main_tabs['Mouse Experimental'].experimental_recoveryanimalsbetter['Experiment']=='Planned']['Cage'].tolist()

            cage_selec=stocks+list(set(exp_siblings))
            cage_selec.sort()
            self.cage_selection['values']=cage_selec

        if project ==7:
            stocks=  self.gui.MouseDat.stock_mice[self.gui.MouseDat.stock_mice['Line_Short'].isin(['VGC::Ai162','VGC::Ai148'])]['Cage'].tolist()

            exp_siblings=  self.gui.main_tabs['Mouse Experimental'].experimental_recoveryanimalsbetter[self.gui.main_tabs['Mouse Experimental'].experimental_recoveryanimalsbetter['Line_Short'].isin(['VGC::Ai162','VGC::Ai148'])][self.gui.main_tabs['Mouse Experimental'].experimental_recoveryanimalsbetter['Experiment']=='Planned']['Cage'].tolist()

            cage_selec=stocks+list(set(exp_siblings))
            cage_selec.sort()
            self.cage_selection['values']=cage_selec

        if project ==8:
            stocks=  self.gui.MouseDat.stock_mice[self.gui.MouseDat.stock_mice['Line_Short'].isin(['Other Colony'])]['Cage'].tolist()
    
            exp_siblings=  self.gui.main_tabs['Mouse Experimental'].experimental_recoveryanimalsbetter[self.gui.main_tabs['Mouse Experimental'].experimental_recoveryanimalsbetter['Line_Short'].isin(['Other Colony'])][self.gui.main_tabs['Mouse Experimental'].experimental_recoveryanimalsbetter['Experiment']=='Planned']['Cage'].tolist()
            cage_selec=stocks+list(set(exp_siblings))
            cage_selec.sort()
            self.cage_selection['values']=cage_selec


        
    def select(self):
        self.selected_mice = list()
        selection = self.mice_lstbox.curselection()
        for i in selection:
            mouse = self.mice_lstbox.get(i)
            self.selected_mice.append(mouse)

    
if __name__ == "__main__":
    

    root = tk.Tk()
    app = new_window_plan_injections(root,)
    root.mainloop()
