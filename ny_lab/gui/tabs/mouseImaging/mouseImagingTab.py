# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 09:21:56 2021

@author: sp3660
"""

import tkinter as tk
from tkinter import ttk
from tkinter import StringVar, Listbox
from ...utils import open_multiple_df_in_new_tkinter_window, open_multiple_df_in_tkinter_frame, button_update_database
# from ...utils import button_update_database  

class MouseImagingTab(tk.Frame):
    def __init__(self, gui_object, gui_tab_control):
        super().__init__(gui_tab_control)
        self.gui_ref=gui_object        
        #%%TAB 4 'Mouse Imaging'
   
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.frames_names=['Actions', 
                           'Imaging Tables'
                           ]
        self.frames={}
        for i in range(len(self.frames_names)):
              self.frames[self.frames_names[i]]=ttk.Frame(self, borderwidth = 4)
              
        self.frames[self.frames_names[0]].grid(row=0, column=0, sticky="nswe") 
        self.frames[self.frames_names[1]].grid(row=1, column=0, sticky="nswe") 

#%%TAB 4 'Mouse Imaging'  'Actions'     
        self.experimental_recoveryanimalsbetter=self.gui_ref.MouseDat.remove_unnecesary_virusinfo_from_df(self.gui_ref.MouseDat.remove_unnecesary_genes_from_df(self.gui_ref.MouseDat.Experimental_class.all_exp_animals_recovery))

        self.frame1= self.frames[self.frames_names[0]]  
        self.frame1.buttons={}
        self.frame1.buttons_names=['Add New Imaging Session',
                                   'All MouseImaging Info', 
                                   'Print Cage Info For Imaging'
                                   ]
        self.frame1.buttons_commands=[self.add_new_imaging_session_button, 
                                      self.check_all_mouse_imaging_info_button, 
                                      self.print_cage_info_for_imaging
                                      ]
        for i in range(len(self.frame1.buttons_names)):
              self.frame1.buttons[self.frame1.buttons_names[i]]= ttk.Button(self.frame1 , text=self.frame1.buttons_names[i], command=self.frame1.buttons_commands[i])
 
        self.frame1.entries={}
        self.frame1.entries_names=['Session Path',
                                   'Mouse Code'
                                   ]
        for i in range(len(self.frame1.entries_names)):
            self.frame1.entries[self.frame1.entries_names[i]]=ttk.Entry(self.frame1 , text='', width=45)
 
        self.frame1.labels={}
        self.frame1.labels_names=['Session Path',
                                  'Mouse Code',
                                  'Cage List'
                                  ]
        for i in range(len(self.frame1.labels_names)):
            self.frame1.labels[self.frame1.labels_names[i]]=ttk.Label(self.frame1, text=self.frame1.labels_names[i], width=20)
             
        self.recovery_cages=StringVar()    
        self.frame1.cage_lstbox =Listbox(self.frame1, listvariable=self.recovery_cages, selectmode='extended', width=10, height=10)
        self.recovery_cages.set(list(self.experimental_recoveryanimalsbetter['Cage'].unique()))
 
 
 
        self.frame1.entries[self.frame1.entries_names[0]].grid(column=1, row=0)
        self.frame1.entries[self.frame1.entries_names[0]].insert(0, r'F:\Projects\LabNY\Imaging\2021\2021MMDD')
        self.frame1.labels[self.frame1.labels_names[0]].grid(column=0, row=0)
        self.frame1.buttons[self.frame1.buttons_names[0]].grid(column=2, row=0)
         
         
        self.frame1.buttons[self.frame1.buttons_names[1]].grid(column=2, row=2)
        self.frame1.entries[self.frame1.entries_names[1]].grid(column=1, row=2)
        self.frame1.entries[self.frame1.entries_names[1]].insert(0, 'SPJA')
        self.frame1.entries[self.frame1.entries_names[1]]['width']=6
        self.frame1.labels[self.frame1.labels_names[1]].grid(column=0, row=2)
         
         
         
        self.frame1.buttons[self.frame1.buttons_names[2]].grid(column=2, row=3)
        self.frame1.labels[self.frame1.labels_names[2]].grid(column=0, row=3)
        self.frame1.cage_lstbox.grid(column=1, row=3, columnspan=1)
 
 
    
    
    #%%TAB 4 'Mouse Imaging'  'Imaging Tables'  
        self.frame2=self.frames[self.frames_names[1]]  
            
        df_dictionary={       
        'All_sessions': self.gui_ref.MouseDat.ImagingDatabase_class.all_imaging_sessions,
        'All_mice': self.gui_ref.MouseDat.ImagingDatabase_class.all_imaged_mice,
        'Interneuron_imaging': self.gui_ref.MouseDat.ImagingDatabase_class.interneuron_imaging_imaged_mice,
        'Chandelier_imaging': self.gui_ref.MouseDat.ImagingDatabase_class.chandelier_imaging_imaged_mice,
        'Interneuron_optogenetics': self.gui_ref.MouseDat.ImagingDatabase_class.interneuron_opto_imaged_mice,
        'Chandelier_optogenetics': self.gui_ref.MouseDat.ImagingDatabase_class.chandelier_opto_imaged_mice,
        'Tigre_controls': self.gui_ref.MouseDat.ImagingDatabase_class.tigres_imaged_mice,
        'All_recordings': self.gui_ref.MouseDat.ImagingDatabase_class.all_acquisitions,
        'allINFO': self.gui_ref.MouseDat.ImagingDatabase_class.all_acquisitons_full_info

        }
        
        open_multiple_df_in_tkinter_frame(self, self.frame2, df_dictionary)
        
    def add_new_imaging_session_button(self):
        sessions=['\\\?\\' +self.frame1.entries[self.frame1.entries_names[0]].get()]
        self.gui_ref.MouseDat.ImagingDatabase_class.add_new_session_to_database(sessions)
        button_update_database(self.gui_ref)

        
        
    def check_all_mouse_imaging_info_button(self):  
        imaging_filer=[('Code',self.frame1.entries[self.frame1.entries_names[1]].get())]        
        mouse_df=self.gui_ref.MouseDat.remove_unnecesary_genes_from_df(self.gui_ref.MouseDat.remove_unnecesary_virusinfo_from_df(self.gui_ref.MouseDat.ImagingDatabase_class.get_filtered_df(self.gui_ref.MouseDat.ImagingDatabase_class.all_acquisitons_full_info, imaging_filer)))
        window_title="Mouse Imaging Info"       
        df_dictionary={self.frame1.entries[self.frame1.entries_names[1]].get(): mouse_df }    
        open_multiple_df_in_new_tkinter_window(self, window_title, df_dictionary)
        
        
    def print_cage_info_for_imaging(self):

        selected_cages = list()
        selection =self.frame1.cage_lstbox.curselection()
        for i in selection:
            cage =self.frame1.cage_lstbox.get(i)
            selected_cages.append(int(cage))
        self.gui_ref.MouseDat.ImagingDatabase_class.get_all_cage_info_for_imaging(selected_cages)