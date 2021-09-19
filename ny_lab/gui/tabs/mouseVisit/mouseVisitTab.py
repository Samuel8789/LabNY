# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 09:20:56 2021

@author: sp3660
"""

import tkinter as tk
from tkinter import ttk

from .new_window_add_separations import new_window_add_separations
from .new_window_add_weanings import new_window_add_weanings
from .new_window_add_rebreedings import new_window_add_rebreedings
from ...utils import button_update_database  
from ....database.fun.guiFunctions.updateLitterInput import UpdateLitterInput

 

class MouseVisitTab(tk.Frame):
    def __init__(self, gui_object, gui_tab_control):
        super().__init__(gui_tab_control)
        self.gui_ref=gui_object

        
 #%%TAB 2 'Mouse Visit' 
   
        self.frames_names=['Main Frame',
                           'Litters Frame', 
                           'Male Separations Frame' 
                           ]
        self.frames={}
        for i in range(len(self.frames_names)):
              self.frames[self.frames_names[i]]=ttk.Frame(self, borderwidth = 4)
              
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.frames[self.frames_names[0]].grid(row=0, column=0, sticky="nswe")         
        self.frames[self.frames_names[1]].grid(row=0, column=1, sticky="nswe")        
        self.frames[self.frames_names[2]].grid(row=0, column=2, sticky="nswe") 
        

#%%TAB 2 'Mouse Visit' 'Main Frame' 

        self.frame1= self.frames[self.frames_names[0]]     
        self.frame1.buttons={}
        self.frame1.buttons_names=['Mouse Previsit',
                                   'Mouse Post Visit', 
                                   'New Mouse Previsit', 
                                   'New Mouse Postvisit', 
                                   'Accept_death',
                                   'Sac Cage'
                                   ]
        self.frame1.buttons_commands=[self.mouse_previsit_button,
                                      self.mouse_postvisit_button, 
                                      self.mouse_new_previsit_button,
                                      self.mouse_new_postvisit_button,
                                      self.female_breeder_death_button,
                                      self.sac_cage_button
                                      ]
        for i in range(len(self.frame1.buttons_names)):
              self.frame1.buttons[self.frame1.buttons_names[i]]= ttk.Button(self.frame1 , text=self.frame1.buttons_names[i], command=self.frame1.buttons_commands[i])
             
        self.frame1.entries={}
        self.frame1.entries_names=[ 'Mouse Post Visit Date', 
                                   'Dead female Code', 
                                   'Dead female Comment',
                                   'Cage to sac'
                                   ]
        for i in range(len(self.frame1.entries_names)):
            self.frame1.entries[self.frame1.entries_names[i]]=ttk.Entry(self.frame1 , text='', width=9)
            
            
        self.frame1.labels={}
        self.frame1.labels_names=['Dead female Code', 
                                  'Dead female Comment', 
                                  'Cage to sac'
                                  ]
        for i in range(len(self.frame1.labels_names)):
            self.frame1.labels[self.frame1.labels_names[i]]=ttk.Label(self.frame1, text=self.frame1.labels_names[i], width=25)    
            
        self.frame1.buttons[self.frame1.buttons_names[0]].grid(column=0, row=0)    
        self.frame1.buttons[self.frame1.buttons_names[1]].grid(column=0, row=1)      
        self.frame1.buttons[self.frame1.buttons_names[2]].grid(column=0, row=3)        
        self.frame1.buttons[self.frame1.buttons_names[3]].grid(column=0, row=4)
        self.frame1.entries[self.frame1.entries_names[0]].grid(column=1, row=1)
        self.frame1.entries[self.frame1.entries_names[0]].insert(0, self.gui_ref.todays_date)

        self.frame1.buttons[self.frame1.buttons_names[4]].grid(column=0, row=7)
        self.frame1.entries[self.frame1.entries_names[1]].grid(column=0, row=5)
        self.frame1.labels[self.frame1.labels_names[0]].grid(column=1, row=5)        
        self.frame1.entries[self.frame1.entries_names[2]].grid(column=0, row=6)
        self.frame1.labels[self.frame1.labels_names[1]].grid(column=1, row=6)
        
        self.frame1.entries[self.frame1.entries_names[2]].insert(0, 'Dead female Breeder')
        self.frame1.entries[self.frame1.entries_names[2]]['width']=20
        
        self.frame1.buttons[self.frame1.buttons_names[5]].grid(column=0, row=9)
        self.frame1.entries[self.frame1.entries_names[3]].grid(column=0, row=8)
        self.frame1.labels[self.frame1.labels_names[2]].grid(column=1, row=8)


        
        
        
          
#%% TAB 2 'Mouse Visit' 'Litters Frame'

        self.litters_info_cages={str(pair[0]):pair[1] for pair in  self.gui_ref.MouseDat.get_litters()[['ID', 'Cage']].apply(list, axis=1).tolist()}
        
        self.frame2= self.frames[self.frames_names[1]]  
        self.frame2.buttons={}
        self.frame2.buttons_names=['Add weanings', 
                                   'Update Litters', 
                                   'Add litters'
                                   ]
        self.frame2.buttons_commands=[self.add_weaning_button,
                                      self.update_old_litters, 
                                      self.button_add_litters]
        for i in range(len(self.frame2.buttons_names)):
              self.frame2.buttons[self.frame2.buttons_names[i]]= ttk.Button(self.frame2 , text=self.frame2.buttons_names[i], command=self.frame2.buttons_commands[i])

        self.frame2.entries={}
        self.frame2.entries_names=['Number of Weanings', 
                                   'Number of Litters'
                                   ]
        for i in range(len(self.frame2.entries_names)):
            self.frame2.entries[self.frame2.entries_names[i]]=ttk.Entry(self.frame2 , text='', width=5)
       
        self.frame2.labels={}
        self.frame2.labels_names=['Number of Weanings', 
                                  'Number of Litters'
                                  ]
        for i in range(len(self.frame2.labels_names)):
            self.frame2.labels[self.frame2.labels_names[i]]=ttk.Label(self.frame2, text=self.frame2.labels_names[i], width=20)

        self.frame2.buttons[self.frame2.buttons_names[0]].grid(column=2, row=2)
        self.frame2.buttons[self.frame2.buttons_names[1]].grid(column=0, row=2)       
        self.frame2.buttons[self.frame2.buttons_names[2]].grid(column=1, row=2)
        
        self.frame2.labels[self.frame2.labels_names[0]].grid(column=2, row=1)
        self.frame2.labels[self.frame2.labels_names[1]].grid(column=1, row=1)
        
        self.frame2.entries[self.frame2.entries_names[0]].grid(column=2, row=0)
        self.frame2.entries[self.frame2.entries_names[1]].grid(column=1, row=0)
  
#%% TAB 2 'Mouse Visit' 'Male Separations Frame' 
        self.breeding_cages=[str(x) for x in  self.gui_ref.MouseDat.breedings[ self.gui_ref.MouseDat.breedings['Male_Cage'].isna()]['Cage'].tolist()]
        self.male_cages={str(pair[0]):pair[1] for pair in  self.gui_ref.MouseDat.breedings[ self.gui_ref.MouseDat.breedings['Male_Cage']>=1][['Male_Cage','Cage']].apply(list, axis=1).tolist()}
        
        self.frame3= self.frames[self.frames_names[2]]  
        self.frame3.buttons={}
        self.frame3.buttons_names=['Add Separations',
                                   'Add Rebreedings', 
                                   ]
        self.frame3.buttons_commands=[self.button_add_separations, 
                                      self.button_add_rebreedings
                                      ]
        for i in range(len(self.frame3.buttons_names)):
              self.frame3.buttons[self.frame3.buttons_names[i]]= ttk.Button(self.frame3 , text=self.frame3.buttons_names[i], command=self.frame3.buttons_commands[i])

        self.frame3.entries={}
        self.frame3.entries_names=['Number of Separations',
                                   'Number of Rebreedings'
                                   ]
        for i in range(len(self.frame3.entries_names)):
            self.frame3.entries[self.frame3.entries_names[i]]=ttk.Entry(self.frame3 , text='', width=5)
       
        self.frame3.labels={}
        self.frame3.labels_names=['Number of Separations',
                                  'Number of Rebreedings'
                                  ]
        for i in range(len(self.frame3.labels_names)):
            self.frame3.labels[self.frame3.labels_names[i]]=ttk.Label(self.frame3, text=self.frame3.labels_names[i], width=20)
        
        self.frame3.entries[self.frame3.entries_names[0]].grid(column=3, row=0)
        self.frame3.entries[self.frame3.entries_names[1]].grid(column=4, row=0)
        
        self.frame3.labels[self.frame3.labels_names[0]].grid(column=3, row=1)
        self.frame3.labels[self.frame3.labels_names[1]].grid(column=4, row=1)
        
        self.frame3.buttons[self.frame3.buttons_names[0]].grid(column=3, row=2)
        self.frame3.buttons[self.frame3.buttons_names[1]].grid(column=4, row=2)
#%% TAB 2 Functions to add    

# #%% TAB 3 MOUSE VISIT FRAME MAIN Cage Sacings
# cages_saced=[195,214,215,218,219,224,225,234,250,251,256,265]
# MouseDat.cage_sacing(cages_saced)
# colony_allmice=MouseDat.all_colony_mice
# colony_actions=MouseDat.actions

# #%% TAB 3 MOUSE VISIT FRAME MAIN Mouse Sacings
# mice_saced_lab_number=[4313,4314,4315]
# for mouse in mice_saced_lab_number:
#     MouseDat.mouse_sacing(mouse, commit=True)
# colony_allmice=MouseDat.all_colony_mice
# colony_actions=MouseDat.actions
# #%% TAB 3 MOUSE VISIT FRAME MAIN Mouse Dead
# mousedaead=[4391, 4392]
# comment='Transfer to Alejandro'
# for mouse in mousedaead:
#     MouseDat.mouse_dead(mouse, comment=comment, commit=True)
    

  
#  #%% TAB 3 MOUSE VISIT FRAME MAIN Labeling
#  cages_labelled=[195,196,208,209,231,232,240,241,242,243]
# MouseDat.labelling(cages_labelled)
# colony_allmice=MouseDat.all_colony_mice
# colony_actions=MouseDat.actions


# #%% TAB 3 MOUSE VISIT FRAME MAIN Genotyping
# cages_genotyped=[195,196,208,209,231,232,240,241,242,243]
# MouseDat.genotyping(cages_genotyped)
# colony_allmice=MouseDat.all_colony_mice
# colony_actions=MouseDat.actions

# #%% TAB 3 MOUSE VISIT FRAME MAIN Breeding Stop
# cages_genotyped=[195,196,208,209,231,232,240,241,242,243]
# MouseDat.genotyping(cages_genotyped)
# colony_allmice=MouseDat.all_colony_mice
# colony_actions=MouseDat.actions




    
# #%% TAB 3 MOUSE VISIT FRAME MAIN New Breeding
    
#   cage=252
# male=4192
# females=[4193]

    def mouse_previsit_button(self) :
         self.gui_ref.MouseDat.mouse_previsit()
         print('previsit printed')

    def mouse_postvisit_button(self) :
        date_performed= self.frame1.entries[self.frame1.entries_names[0]].get()
        self.gui_ref.MouseDat.mouse_postvisit(date_performed=date_performed)
        print('postvisit printed')

       
    def mouse_new_previsit_button(self):
         self.gui_ref.MouseDat.mouse_previsit(new_visit=True)
         print('new previsit printed')

    def mouse_new_postvisit_button(self) :
         self.gui_ref.MouseDat.mouse_postvisit(new_visit=True)
         print('new postvisit printed')

    def button_add_separations(self):
        number_of_separations=int(self.frame3.entries[self.frame3.entries_names[0]].get())     
        new_window_add_separations( self.gui_ref,  self.breeding_cages, number_of_separations)
        button_update_database(self.gui_ref)
        print('separations added')

        
    def add_weaning_button(self):
        number_of_weanings=int( self.frame2.entries[self.frame2.entries_names[0]].get())  
        new_window_add_weanings( self.gui_ref,  self.litters_info_cages, number_of_weanings)
        button_update_database(self.gui_ref)
        print('weanings added')

        
    def update_old_litters(self):
        
    
        UpdateLitterInput(self.gui_ref, self.gui_ref.MouseDat)
        button_update_database(self.gui_ref)
        print('litters updated')


    def button_add_litters(self):
        new_litter_number=int( self.frame2.entries[self.frame2.entries_names[1]].get()) 
        self.gui_ref.MouseDat.add_new_litters(new_litter_number)
        button_update_database(self.gui_ref)
        print('litters added')

        
    def button_add_rebreedings(self):
        number_of_rebreedings=int(self.frame3.entries[self.frame3.entries_names[1]].get())     
        new_window_add_rebreedings( self.gui_ref,  self.male_cages, number_of_rebreedings)
        button_update_database(self.gui_ref)
        print('rebreedings updated')

        
    def female_breeder_death_button(self) :
    
         self.gui_ref.MouseDat.dead_mother(int(self.frame1.entries[self.frame1.entries_names[1]].get()), 
                              comment=self.frame1.entries[self.frame1.entries_names[2]].get(), 
                              commit=True)
         print('female death updated')

    def sac_cage_button (self):
        cage_saced=[int(self.frame1.entries[self.frame1.entries_names[3]].get())]
        self.gui_ref.MouseDat.cage_sacing(cage_saced)
        button_update_database(self.gui_ref)
        print('Cage saced')

