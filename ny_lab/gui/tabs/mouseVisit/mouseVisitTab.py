# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 09:20:56 2021

@author: sp3660
"""

import tkinter as tk
from tkinter import ttk
from tkcalendar import Calendar
import datetime
import os
from .new_window_add_separations import new_window_add_separations
from .new_window_add_weanings import new_window_add_weanings
from .new_window_add_rebreedings import new_window_add_rebreedings
from ...utils import button_update_database  
from ....database.fun.guiFunctions.updateLitterInput import UpdateLitterInput
from .widgetSelectStockCageMice import WidgetSelectStockCageMice
from .widgetSelectMultiCage import WidgetSelectMultiCage
import subprocess

class MouseVisitTab(tk.Frame):
    def __init__(self, gui_object, gui_tab_control):
        super().__init__(gui_tab_control)
        self.gui_ref=gui_object

        
 #%%TAB 2 'Mouse Visit' 
   
        self.frames_names=['Main Frame',
                           'Litters Frame', 
                           'Male Separations Frame' ,
                           'Stock Sacings and Removals',
                           'Labbeling and genotyping',
                           'Breeding managing'
                           
                           ]
        self.frames={}
        for i in range(len(self.frames_names)):
              self.frames[self.frames_names[i]]=ttk.Frame(self, borderwidth = 4, relief='groove')
              
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        
        self.frames[self.frames_names[0]].grid(row=0, column=0, sticky="nswe")         
        self.frames[self.frames_names[1]].grid(row=0, column=1, sticky="nswe")        
        self.frames[self.frames_names[2]].grid(row=0, column=2, sticky="nswe") 
        self.frames[self.frames_names[3]].grid(row=1, column=0, sticky="nswe") 
        self.frames[self.frames_names[4]].grid(row=1, column=1, sticky="nswe") 
        self.frames[self.frames_names[5]].grid(row=1, column=2, sticky="nswe") 


        

#%%TAB 2 'Mouse Visit' 'Main Frame' 

        self.frame1= self.frames[self.frames_names[0]]     
        self.frame1.buttons={}
        self.frame1.buttons_names=['Mouse Previsit',
                                   'Mouse Post Visit', 
                                   'New Mouse Previsit', 
                                   'New Mouse Postvisit', 
                                   'Accept_death',
                                   'Open Mouse Visit Dir'
                                   ]
        self.frame1.buttons_commands=[self.mouse_previsit_button,
                                      self.mouse_postvisit_button, 
                                      self.mouse_new_previsit_button,
                                      self.mouse_new_postvisit_button,
                                      self.female_breeder_death_button,
                                      self.open_mouse_visits_button

                                      ]
        for i in range(len(self.frame1.buttons_names)):
              self.frame1.buttons[self.frame1.buttons_names[i]]= ttk.Button(self.frame1 , text=self.frame1.buttons_names[i], command=self.frame1.buttons_commands[i])
             
        self.frame1.entries={}
        self.frame1.entries_names=[ 'Mouse Post Visit Date', 
                                   'Dead female Code', 
                                   'Dead female Comment',
                                   ]
        for i in range(len(self.frame1.entries_names)):
            self.frame1.entries[self.frame1.entries_names[i]]=ttk.Entry(self.frame1 , text='', width=9)
            
            
        self.frame1.labels={}
        self.frame1.labels_names=['Dead female Code', 
                                  'Dead female Comment', 
                                  ]
        for i in range(len(self.frame1.labels_names)):
            self.frame1.labels[self.frame1.labels_names[i]]=ttk.Label(self.frame1, text=self.frame1.labels_names[i], width=25)    
            
        self.frame1.buttons[self.frame1.buttons_names[0]].grid(column=0, row=0)    
        self.frame1.buttons[self.frame1.buttons_names[1]].grid(column=0, row=1)      
        self.frame1.buttons[self.frame1.buttons_names[2]].grid(column=0, row=3)        
        self.frame1.buttons[self.frame1.buttons_names[3]].grid(column=0, row=4)
        self.frame1.entries[self.frame1.entries_names[0]].grid(column=1, row=1)
        self.frame1.entries[self.frame1.entries_names[0]].insert(0, self.gui_ref.todays_date)
        self.frame1.buttons[self.frame1.buttons_names[5]].grid(column=1, row=0)


        self.frame1.buttons[self.frame1.buttons_names[4]].grid(column=0, row=7)
        self.frame1.entries[self.frame1.entries_names[1]].grid(column=0, row=5)
        self.frame1.labels[self.frame1.labels_names[0]].grid(column=1, row=5)        
        self.frame1.entries[self.frame1.entries_names[2]].grid(column=0, row=6)
        self.frame1.labels[self.frame1.labels_names[1]].grid(column=1, row=6)
        
        self.frame1.entries[self.frame1.entries_names[2]].insert(0, 'Dead female Breeder')
        self.frame1.entries[self.frame1.entries_names[2]]['width']=20
        

        
          
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
#%% TAB 2 'Mouse Visit' Sacings and removals  

        self.sac_animals=self.frames[self.frames_names[3]]
        self.sac_animals.selection_frame=WidgetSelectStockCageMice(self.gui_ref.MouseDat, self.sac_animals, sorted(self.gui_ref.MouseDat.stock_mice['Cage'].values.tolist()))      
        self.sac_animals.sac_animals_button= ttk.Button( self.sac_animals , text='SAC Animals', command=self.sac_animals_button_action)
        
        self.sac_animals.other_remove_animals_label=ttk.Label( self.sac_animals, text='Removal reason', width=25)   
        self.sac_animals.other_remove_comment_entry=ttk.Entry(self.sac_animals , text='', width=25)
        self.sac_animals.other_remove_animals_button= ttk.Button( self.sac_animals , text='Remove Animal', command=self.remove_animals_button_action)


        self.sac_animals.sac_cage_button= ttk.Button( self.sac_animals , text='SAC cage', command=self.sac_cage_button)

 
        self.sac_animals.selection_frame.grid(column=0, row=0)
        
        self.sac_animals.sac_cage_button.grid(column=0, row=1)

        
        self.sac_animals.sac_animals_button.grid(column=0, row=2)
        

        self.sac_animals.other_remove_animals_label.grid(column=0, row=3)
        self.sac_animals.other_remove_comment_entry.grid(column=0, row=4)
        self.sac_animals.other_remove_animals_button.grid(column=0, row=5)
        
#%% TAB 2 'Mouse Visit' Genotypings and labellings 
        self.label_genot=self.frames[self.frames_names[4]]
        self.label_genot.selection_frame=WidgetSelectStockCageMice(self.gui_ref.MouseDat, self.label_genot, sorted(self.gui_ref.MouseDat.mice_to_genotype['Cage'].values.tolist()))      


        self.label_genot.selection_frame.grid(column=0, row=0)
        
        self.label_genot.label_animals_button= ttk.Button( self.label_genot , text='Label Animals', command=self.label_animals_button_action)
        self.label_genot.genotype_animals_button= ttk.Button( self.label_genot , text='Genotype Animals', command=self.genotype_animals_button_action)

        self.label_genot.label_animals_button.grid(column=0, row=1)
        self.label_genot.genotype_animals_button.grid(column=0, row=2)



#%% TAB 2 'Mouse Visit'Breedinsg start end
        today_date=datetime.date.today()
        self.breedings_manager=self.frames[self.frames_names[5]]
        
        self.breedings_manager.stop_selection_frame= WidgetSelectMultiCage(self.gui_ref.MouseDat, self.breedings_manager, sorted(self.gui_ref.MouseDat.breedings['Cage'].values.tolist())) 
        self.breedings_manager.stop_breedings_button= ttk.Button( self.breedings_manager , text='Stop Breeding', command=self.stop_breedings_button_action)
        
        self.breedings_manager.sac_breeders_subframe=ttk.Frame(self.breedings_manager, borderwidth = 4, relief='groove')
        self.breedings_manager.sac_breeders_subframe.sac_breeder_selection_frame= WidgetSelectMultiCage(self.gui_ref.MouseDat,
                                                                                                        self.breedings_manager.sac_breeders_subframe, 
                                                                                                        sorted(self.gui_ref.MouseDat.all_colony_mice[self.gui_ref.MouseDat.all_colony_mice['Breeders_types']=='Ended']['Cage'].unique().tolist())) 
       
        
        self.breedings_manager.sac_breeders_subframe.sac_breedings_button= ttk.Button(   self.breedings_manager.sac_breeders_subframe , text='Sac Breeding', command=self.sac_breeder_cage_button)
        
        self.breedings_manager.male_selection_label=ttk.Label( self.breedings_manager, text='Select Male', width=15)   
        self.breedings_manager.start_male_selection_frame= WidgetSelectStockCageMice(self.gui_ref.MouseDat, 
                                                                                     self.breedings_manager, 
                                                                                     sorted(self.gui_ref.MouseDat.stock_mice[self.gui_ref.MouseDat.stock_mice['Sex']=='Male']['Cage'].values.tolist()))   
        
        
        self.breedings_manager.female_selection_label=ttk.Label( self.breedings_manager, text='Select Females', width=15)   
        self.breedings_manager.breedings_manager_female_subframe=ttk.Frame(self.breedings_manager, borderwidth = 4, relief='groove')
        self.breedings_manager.breedings_manager_female_subframe.start_female_selection_frame= WidgetSelectStockCageMice(self.gui_ref.MouseDat, 
                                                                                                                         self.breedings_manager.breedings_manager_female_subframe, 
                                                                                                                         sorted(self.gui_ref.MouseDat.stock_mice[self.gui_ref.MouseDat.stock_mice['Sex']=='Female']['Cage'].values.tolist()))  
        
        
        
        
        self.breedings_manager.female_selection_label2=ttk.Label( self.breedings_manager, text='Select Females2', width=15)   
        self.breedings_manager.breedings_manager_female_subframe2=ttk.Frame(self.breedings_manager, borderwidth = 4, relief='groove')
        self.breedings_manager.breedings_manager_female_subframe2.start_female_selection_frame= WidgetSelectStockCageMice(self.gui_ref.MouseDat, 
                                                                                                                         self.breedings_manager.breedings_manager_female_subframe2, 
                                                                                                                         sorted(self.gui_ref.MouseDat.stock_mice[self.gui_ref.MouseDat.stock_mice['Sex']=='Female']['Cage'].values.tolist()))   
        
     
        self.breeding_cage=tk.IntVar()   

        self.breedings_manager.start_breeding_date_selection = Calendar(self.breedings_manager, selectmode = 'day', year = today_date.year, month = today_date.month, day = today_date.day,  date_pattern ='ymmdd')
        self.breedings_manager.start_breeding_button= ttk.Button( self.breedings_manager , text='Start New Breeding', command=self.add_new_breeding_button)
        self.breedings_manager.new_breeding_cage_label=ttk.Label( self.breedings_manager, text='Breeding Cage', width=16)   
        self.breedings_manager.new_breeding_cage_entry=ttk.Entry(self.breedings_manager , text='', textvariable=self.breeding_cage,width=8)
        
        
        self.breedings_manager.stop_selection_frame.grid(column=0, row=1)
        self.breedings_manager.stop_breedings_button.grid(column=0, row=2)
        self.breedings_manager.sac_breeders_subframe.grid(column=0, row=3)
        self.breedings_manager.sac_breeders_subframe.sac_breeder_selection_frame.grid(column=0, row=0)
        self.breedings_manager.sac_breeders_subframe.sac_breedings_button.grid(column=0, row=4)
        
        self.breedings_manager.male_selection_label.grid(column=1, row=0)
        self.breedings_manager.female_selection_label.grid(column=2, row=0)
        
       
        self.breedings_manager.start_male_selection_frame.grid(column=1, row=1)
        self.breedings_manager.breedings_manager_female_subframe.grid(column=2, row=1)
        self.breedings_manager.breedings_manager_female_subframe2.grid(column=3, row=1)
        self.breedings_manager.breedings_manager_female_subframe.start_female_selection_frame.grid(column=0, row=0)
        self.breedings_manager.breedings_manager_female_subframe2.start_female_selection_frame.grid(column=1, row=0)

        
        self.breedings_manager.start_breeding_date_selection.grid(column=4, row=1)
        
        self.breedings_manager.start_breeding_button.grid(column=2, row=2)
        self.breedings_manager.sac_breeders_subframe.sac_breedings_button.grid(column=0, row=3)
        
        self.breedings_manager.new_breeding_cage_label.grid(column=3, row=2) 
        self.breedings_manager.new_breeding_cage_entry.grid(column=3, row=3)
        
   
    
   
        
#%% button funcrions
    def open_mouse_visits_button(self):
        
        visitpath=os.path.join(os.path.expanduser('~'),r'Documents/Projects/LabNY/4. Mouse Managing/MouseVisits')
        
        subprocess.Popen(['xdg-open',visitpath])

        
        
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
        self.gui_ref.MouseDat.add_new_litters(self.gui_ref, new_litter_number)
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
        cage_saced= [self.sac_animals.selected_cage]
        self.gui_ref.MouseDat.cage_sacing(cage_saced)
        button_update_database(self.gui_ref)
        self.sac_animals.selection_frame.update_lists(updated_cage_list=sorted(self.gui_ref.MouseDat.stock_mice['Cage'].values.tolist()))
        print('Cage saced')
        
    def sac_breeder_cage_button(self):
        breeding_cages= self.breedings_manager.sac_breeders_subframe.selected_cages
        self.gui_ref.MouseDat.cage_sacing(breeding_cages)
        button_update_database(self.gui_ref)
        self.breedings_manager.sac_breeders_subframe.sac_breeder_selection_frame.update_lists(updated_cage_list= sorted(self.gui_ref.MouseDat.all_colony_mice[self.gui_ref.MouseDat.all_colony_mice['Breeders_types']=='Ended']['Cage'].unique().tolist()))

        print('Cage saced')
    def sac_animals_button_action(self):
        
        
        for mouse in  self.sac_animals.selected_mice:
           self.gui_ref.MouseDat.mouse_sacing(mouse, commit=True)

        button_update_database(self.gui_ref)
        self.sac_animals.selection_frame.update_lists(updated_cage_list=sorted(self.gui_ref.MouseDat.stock_mice['Cage'].values.tolist()))

        print('Animals saced')
               
    def remove_animals_button_action(self):

        for mouse in  self.sac_animals.selected_mice:
           self.gui_ref.MouseDat.mouse_dead(mouse, comment= self.sac_animals.other_remove_comment_entry.get(), commit=True)

        button_update_database(self.gui_ref)
        self.sac_animals.selection_frame.update_lists(updated_cage_list=sorted(self.gui_ref.MouseDat.stock_mice['Cage'].values.tolist()))

        print('Animals removed')  
        
    def label_animals_button_action(self):
        self.label_genot.selection_frame
        cages_labelled=[self.label_genot.selected_cage]
        self.gui_ref.MouseDat.labelling(cages_labelled)
        button_update_database(self.gui_ref)
        self.label_genot.selection_frame.update_lists(updated_cage_list=sorted(self.gui_ref.MouseDat.mice_to_genotype['Cage'].values.tolist()))
        
        print('Cage Labelled')
    
        
    def genotype_animals_button_action(self):
        cages_genotyped=[self.label_genot.selected_cage]
        self.gui_ref.MouseDat.genotyping(self.gui_ref, cages_genotyped)
        button_update_database(self.gui_ref)
        self.label_genot.selection_frame.update_lists(updated_cage_list=sorted(self.gui_ref.MouseDat.mice_to_genotype['Cage'].values.tolist()))

        print('Cage genotyped')
        
    def stop_breedings_button_action(self):
        # add a 1 cage selection conditional
        breeding_cages= self.breedings_manager.selected_cages
        self.gui_ref.MouseDat.breeding_stop(breeding_cages)
        button_update_database(self.gui_ref)
        self.breedings_manager.stop_selection_frame.update_lists(updated_cage_list=sorted(self.gui_ref.MouseDat.breedings['Cage'].values.tolist()))
        self.breedings_manager.sac_breeders_subframe.sac_breeder_selection_frame.update_lists( updated_cage_list=sorted(self.gui_ref.MouseDat.all_colony_mice[self.gui_ref.MouseDat.all_colony_mice['Breeders_types']=='Ended']['Cage'].unique().tolist()))

        print('Breeders stopped')      
        
        
    def add_new_breeding_button(self):
        cage=self.breeding_cage.get()
        male_select=self.breedings_manager.selected_mice[0]
        females_select1=self.breedings_manager.breedings_manager_female_subframe.selected_mice
        females_select2=[]
        if self.breedings_manager.breedings_manager_female_subframe2.selected_mice:
            females_select2=self.breedings_manager.breedings_manager_female_subframe2.selected_mice
        females_select=females_select1+females_select2



        date_entry=self.breedings_manager.start_breeding_date_selection.get_date()
       
        
        self.gui_ref.MouseDat.add_new_breeding( cage, male_select, females_select , date_performed=date_entry)
        button_update_database(self.gui_ref)
        self.breedings_manager.start_male_selection_frame.update_lists(updated_cage_list=sorted(self.gui_ref.MouseDat.stock_mice[self.gui_ref.MouseDat.stock_mice['Sex']=='Male']['Cage'].values.tolist()))
        self.breedings_manager.breedings_manager_female_subframe.start_female_selection_frame.update_lists(updated_cage_list=sorted(self.gui_ref.MouseDat.stock_mice[self.gui_ref.MouseDat.stock_mice['Sex']=='Female']['Cage'].values.tolist()))
        self.breedings_manager.breedings_manager_female_subframe2.start_female_selection_frame.update_lists(updated_cage_list=sorted(self.gui_ref.MouseDat.stock_mice[self.gui_ref.MouseDat.stock_mice['Sex']=='Female']['Cage'].values.tolist()))
        self.breedings_manager.breedings_manager_female_subframe2.selected_mice=None
        females_select2=[]

        print('New Breeding Set Up')      
        
        