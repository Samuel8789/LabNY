# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 09:21:26 2021

@author: sp3660
"""
import os
import datetime
import tkinter as tk
from tkinter import ttk
from tkinter import StringVar, Listbox, MULTIPLE
from tkcalendar import Calendar

from ...utils import open_multiple_df_in_tkinter_frame, button_update_database
from .new_window_plan_injections import  new_window_plan_injections
from .new_window_update_done_injections import  new_window_update_done_injections
from .new_window_update_done_windows import  new_window_update_done_windows
from .new_window_post_op_injections import  new_window_post_op_injections

from ..mouseVisit.widgetSelectStockCageMice import WidgetSelectStockCageMice
from .processBrainWindow import ProcessBrain

class MouseExperimentalTab(tk.Frame):
    def __init__(self, gui_object, gui_tab_control):
        super().__init__(gui_tab_control)
        self.gui_ref=gui_object        
#%%TAB 3 'Mouse Experimental' 
   
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
               
        self.frames_names=['Injections',
                           'Windows' , 
                           'Post-op',
                           'Experimental Tables',
                           'Brain_processing']
        self.frames={}
        for i in range(len(self.frames_names)):
              self.frames[self.frames_names[i]]=ttk.Frame(self, borderwidth = 4,  relief='groove')
           
        self.frames[self.frames_names[0]].grid(row=0, column=0, sticky="nswe") 
        self.frames[self.frames_names[1]].grid(row=0, column=1, sticky="nswe") 
        self.frames[self.frames_names[2]].grid(row=0, column=2, sticky="nswe") 
        self.frames[self.frames_names[4]].grid(row=1, column=0, sticky="nswe") 

        self.frames[self.frames_names[3]].grid(row=2, column=0, columnspan=3, sticky="nswe") 

    

#%%TAB 3 'Mouse Experimental'  'Injections'
        self.frame1= self.frames[self.frames_names[0]]  
        self.frame1.buttons={}
        self.frame1.buttons_names=['Plan Injections', 
                                   'Prepare Injection Templates',
                                   'Update Done Injections'
                                        ]
        self.frame1.buttons_commands=[self.plan_injection_button, 
                                      self.prepare_injection_templates_button,
                                      self.update_done_injection_button
                                           ]
        for i in range(len(self.frame1.buttons_names)):
              self.frame1.buttons[self.frame1.buttons_names[i]]= ttk.Button(self.frame1 , text=self.frame1.buttons_names[i], command=self.frame1.buttons_commands[i])
              
        self.frame1.buttons[self.frame1.buttons_names[0]].grid(column=0, row=0)

        self.to_do_injections_codes=StringVar()    
        self.frame1.codes_lstbox =Listbox(self.frame1, listvariable=self.to_do_injections_codes, selectmode=MULTIPLE, width=10, height=10)
        self.to_do_injections_codes.set(self.gui_ref.MouseDat.Experimental_class.all_mouse_to_do_injection['Code'].tolist())
        
        self.frame1.buttons[self.frame1.buttons_names[1]].grid(column=1, row=0)
        self.frame1.codes_lstbox.grid(column=1, row=1)

        self.frame1.buttons[self.frame1.buttons_names[2]].grid(column=0, row=11)


#%%TAB 3 'Mouse Experimental'  'Windows'
#%% TO DO PLAN WINDOW
        self.frame2= self.frames[self.frames_names[1]]  
        self.frame2.buttons={}
        self.frame2.buttons_names=['Plan Windows No Injection', 
                                   'Prepare Window Templates',
                                   'Update Done Windows'
                                   ]
        self.frame2.buttons_commands=[self.plan_window_button, 
                                      self.prepare_window_templates_button,
                                      self.update_done_window_button                                      
                                      ]
        for i in range(len(self.frame2.buttons_names)):
              self.frame2.buttons[self.frame2.buttons_names[i]]= ttk.Button(self.frame2 , text=self.frame2.buttons_names[i], command=self.frame2.buttons_commands[i])

        self.frame2.buttons[self.frame2.buttons_names[0]].grid(column=0, row=0)
        
        self.to_do_windows_codes=StringVar()    
        self.frame2.codes_lstbox =Listbox(self.frame2, listvariable=self.to_do_windows_codes, selectmode=MULTIPLE, width=10, height=10)
        self.to_do_windows_codes.set(self.gui_ref.MouseDat.Experimental_class.all_mouse_to_do_window['Code'].tolist())
        
        self.frame2.buttons[self.frame2.buttons_names[1]].grid(column=1, row=0)
        self.frame2.codes_lstbox.grid(column=1, row=1)

        self.frame2.buttons[self.frame2.buttons_names[2]].grid(column=0, row=11)


#%%TAB 3 'Mouse Experimental' 'Post-op'
        self.frame3= self.frames[self.frames_names[2]]  
        today_date=datetime.date.today()
        self.frame3.buttons={}
        self.frame3.buttons_names=['Update Post-op',
                                   'Saced Exp Animal'
                                   ]
        self.frame3.buttons_commands=[self.update_post_op_button,
                                      self.saced_exp_animal_button
                                      ]
        for i in range(len(self.frame3.buttons_names)):
              self.frame3.buttons[self.frame3.buttons_names[i]]= ttk.Button(self.frame3 , text=self.frame3.buttons_names[i], command=self.frame3.buttons_commands[i])

        self.frame3.saced_date_selection = Calendar(self.frame3, selectmode = 'day', year = today_date.year, month = today_date.month, day = today_date.day,  date_pattern ='ymmdd')
        self.frame3.saced_date_selection.grid(column=3, row=1)
        self.saced_exp_codes=StringVar()
        self.frame3.codes_lstbox =Listbox(self.frame3, listvariable=self.saced_exp_codes, selectmode=MULTIPLE, width=10, height=10)
        self.saced_exp_codes.set(self.gui_ref.MouseDat.Experimental_class.all_exp_animals_recovery['Code'].tolist())
        # self.saced_exp_codes.set(self.gui_ref.MouseDat.Experimental_class.all_mouse_planned_in_colony['Code'].tolist())

        self.frame3.codes_lstbox.grid(column=1, row=1)
        self.frame3.buttons[self.frame3.buttons_names[0]].grid(column=0, row=0)
        self.frame3.buttons[self.frame3.buttons_names[1]].grid(column=1, row=2)
#%%TAB 3 'Mouse Experimental'  'Brain_processing'
        self.frame_brain_processing= self.frames[self.frames_names[4]]  
        today_date=datetime.date.today()
        self.frame_brain_processing.brain_processing_selection_frame= WidgetSelectStockCageMice(self.gui_ref.MouseDat, 
                                                                                     self.frame_brain_processing, 
                                                                                     sorted(self.gui_ref.MouseDat.Experimental_class.all_experimental_all_info['Cage'].dropna().unique().to_numpy().tolist()), 
                                                                                     exp_code=True)   
        
        self.frame_brain_processing.brain_processing_date_selection = Calendar(self.frame_brain_processing, selectmode = 'day', year = today_date.year, month = today_date.month, day = today_date.day,  date_pattern ='ymmdd')

        self.frame_brain_processing.buttons={}
        self.frame_brain_processing.buttons_names=['Process_Brains',                                                                     
                                                   ]
        self.frame_brain_processing.buttons_commands=[self.process_brain_button,
                                     ]
        for i in range(len(self.frame_brain_processing.buttons_names)):
              self.frame_brain_processing.buttons[self.frame_brain_processing.buttons_names[i]]= ttk.Button(self.frame_brain_processing ,
                                                                                                            text=self.frame_brain_processing.buttons_names[i], 
                                                                                                            command=self.frame_brain_processing.buttons_commands[i])

        self.frame_brain_processing.labels={}
        self.frame_brain_processing.labels_names=['Process Brains' 
                                   ]
        for i in range(len(self.frame_brain_processing.labels_names)):
            self.frame_brain_processing.labels[self.frame_brain_processing.labels_names[i]]=ttk.Label(self.frame_brain_processing, text=self.frame_brain_processing.labels_names[i], width=20)
 
 
        self.frame_brain_processing.buttons[self.frame_brain_processing.buttons_names[0]].grid(column=1, row=0)   
        self.frame_brain_processing.labels[self.frame_brain_processing.labels_names[0]].grid(column=0, row=0)        
        self.frame_brain_processing.brain_processing_selection_frame.grid(column=0, row=1)
        self.frame_brain_processing.brain_processing_date_selection .grid(column=1, row=1)




#%%TAB 3 'Mouse Experimental'  'Experimental Tables'
        self.frame4=self.frames[self.frames_names[3]]  
        self.experimental_recoveryanimalsbetter=self.gui_ref.MouseDat.remove_unnecesary_virusinfo_from_df(self.gui_ref.MouseDat.remove_unnecesary_genes_from_df(self.gui_ref.MouseDat.Experimental_class.all_exp_animals_recovery))
        self.frame4.df_dictionary={'All Experimental':self.gui_ref.MouseDat.Experimental_class.all_experimental_all_info,
                       'Planned in colony':self.gui_ref.MouseDat.Experimental_class.all_mouse_planned_in_colony,
                       'All Exp In Recovery':self.gui_ref.MouseDat.Experimental_class.all_exp_animals_recovery, 
                       'Virus Combinations':self.gui_ref.MouseDat.Experimental_class.virusinfo,
                       'To Do Injections':self.gui_ref.MouseDat.Experimental_class.all_mouse_to_do_injection,
                       'Mouse To Image': self.gui_ref.MouseDat.Experimental_class.all_mouse_with_window,
                       'To Do Windows':self.gui_ref.MouseDat.Experimental_class.all_mouse_to_do_window,
                       'To Do Carprofen':self.gui_ref.MouseDat.Experimental_class.all_mouse_to_do_postop_injection,
                       'Better All Exp In Recovery':self.experimental_recoveryanimalsbetter,
                       'Interneuron_Imaging':self.gui_ref.MouseDat.remove_unnecesary_virusinfo_from_df(self.gui_ref.MouseDat.remove_unnecesary_genes_from_df(self.experimental_recoveryanimalsbetter[self.experimental_recoveryanimalsbetter['Projects']=='Interneuron_Imaging'])),
                       'Interneuron_Optogenetics':self.gui_ref.MouseDat.remove_unnecesary_virusinfo_from_df(self.gui_ref.MouseDat.remove_unnecesary_genes_from_df(self.experimental_recoveryanimalsbetter[self.experimental_recoveryanimalsbetter['Projects']=='Interneuron_Optogenetics'])),
                       'Chandelier_Imaging':self.gui_ref.MouseDat.remove_unnecesary_virusinfo_from_df(self.gui_ref.MouseDat.remove_unnecesary_genes_from_df(self.experimental_recoveryanimalsbetter[self.experimental_recoveryanimalsbetter['Projects']=='Chandelier_Imaging'])),
                       'Chandelier_Optogenetics':self.gui_ref.MouseDat.remove_unnecesary_virusinfo_from_df(self.gui_ref.MouseDat.remove_unnecesary_genes_from_df(self.experimental_recoveryanimalsbetter[self.experimental_recoveryanimalsbetter['Projects']=='Chandelier_Optogenetics'])),
                       'Tigre_Controls':self.gui_ref.MouseDat.remove_unnecesary_virusinfo_from_df(self.gui_ref.MouseDat.remove_unnecesary_genes_from_df(self.experimental_recoveryanimalsbetter[self.experimental_recoveryanimalsbetter['Projects']=='Tigre_Controls']))                      
                      }  
        open_multiple_df_in_tkinter_frame(self,  self.frame4, self.frame4.df_dictionary)
        

    def update_dataframes(self):
        self.experimental_recoveryanimalsbetter=self.gui_ref.MouseDat.remove_unnecesary_virusinfo_from_df(self.gui_ref.MouseDat.remove_unnecesary_genes_from_df(self.gui_ref.MouseDat.Experimental_class.all_exp_animals_recovery))
        self.frame4.df_dictionary={'All Experimental':self.gui_ref.MouseDat.Experimental_class.all_experimental_all_info,
                       'Planned in colony':self.gui_ref.MouseDat.Experimental_class.all_mouse_planned_in_colony,
                       'All Exp In Recovery':self.gui_ref.MouseDat.Experimental_class.all_exp_animals_recovery, 
                       'Virus Combinations':self.gui_ref.MouseDat.Experimental_class.virusinfo,
                       'To Do Injections':self.gui_ref.MouseDat.Experimental_class.all_mouse_to_do_injection,
                       'Mouse To Image': self.gui_ref.MouseDat.Experimental_class.all_mouse_with_window,
                       'To Do Windows':self.gui_ref.MouseDat.Experimental_class.all_mouse_to_do_window,
                       'To Do Carprofen':self.gui_ref.MouseDat.Experimental_class.all_mouse_to_do_postop_injection,
                       'Better All Exp In Recovery':self.experimental_recoveryanimalsbetter,
                       'Interneuron_Imaging':self.gui_ref.MouseDat.remove_unnecesary_virusinfo_from_df(self.gui_ref.MouseDat.remove_unnecesary_genes_from_df(self.experimental_recoveryanimalsbetter[self.experimental_recoveryanimalsbetter['Projects']=='Interneuron_Imaging'])),
                       'Interneuron_Optogenetics':self.gui_ref.MouseDat.remove_unnecesary_virusinfo_from_df(self.gui_ref.MouseDat.remove_unnecesary_genes_from_df(self.experimental_recoveryanimalsbetter[self.experimental_recoveryanimalsbetter['Projects']=='Interneuron_Optogenetics'])),
                       'Chandelier_Imaging':self.gui_ref.MouseDat.remove_unnecesary_virusinfo_from_df(self.gui_ref.MouseDat.remove_unnecesary_genes_from_df(self.experimental_recoveryanimalsbetter[self.experimental_recoveryanimalsbetter['Projects']=='Chandelier_Imaging'])),
                       'Chandelier_Optogenetics':self.gui_ref.MouseDat.remove_unnecesary_virusinfo_from_df(self.gui_ref.MouseDat.remove_unnecesary_genes_from_df(self.experimental_recoveryanimalsbetter[self.experimental_recoveryanimalsbetter['Projects']=='Chandelier_Optogenetics'])),
                       'Tigre_Controls':self.gui_ref.MouseDat.remove_unnecesary_virusinfo_from_df(self.gui_ref.MouseDat.remove_unnecesary_genes_from_df(self.experimental_recoveryanimalsbetter[self.experimental_recoveryanimalsbetter['Projects']=='Tigre_Controls']))                      
                      }  
        open_multiple_df_in_tkinter_frame(self, self.frame4, self.frame4.df_dictionary, update=True)
     
        
#%% injections        
    def plan_injection_button(self):
        self.plan_injection_window=new_window_plan_injections(self.gui_ref)
        self.plan_injection_window.wait_window()
        button_update_database(self.gui_ref)  
        print('injection planned')
        
    def prepare_injection_templates_button(self):
      
            selected_codes = list()
            selection = self.frame1.codes_lstbox.curselection()
            for i in selection:
                code = self.frame1.codes_lstbox.get(i)
                selected_codes.append(code)

            selectedinjections=self.gui_ref.MouseDat.Experimental_class.all_mouse_to_do_injection.loc[self.gui_ref.MouseDat.Experimental_class.all_mouse_to_do_injection['Code'].isin(selected_codes)]
            selectedinjectionsinfo=selectedinjections[['Code','Lab_Number','Sex','Cage','Age','Line_Short','Projects','Labels_types']]
            selectedinjectionsinfo.to_excel(os.path.join(r'C:\Users\sp3660\Desktop\Temp_Excel_Files' ,'InfoForInjectionTemplates_{0}.xlsx'.format(datetime.date.today().strftime("%Y%m%d") )))
            
    def update_done_injection_button(self): 
                
            self.update_injection_window=new_window_update_done_injections(self.gui_ref)
            self.update_injection_window.wait_window()
            button_update_database(self.gui_ref)
                            
#%% TO DO PLAN WINDOW            
    def plan_window_button(self):
        # new_window_plan_windows(self)
        button_update_database(self.gui_ref)    
#         cage=113
# selectedanimals=[4158,4182]
# MouseDat.Experimental_class.plan_new_window(cage=cage,lab_number_selected=selectedanimals)

#%%  windows      
    def prepare_window_templates_button(self):
      
        selected_codes = list()
        selection = self.frame2.codes_lstbox.curselection()
        for i in selection:
            code = self.frame2.codes_lstbox.get(i)
            selected_codes.append(code)

        selectedwidnows=self.gui_ref.MouseDat.Experimental_class.all_mouse_to_do_window.loc[self.gui_ref.MouseDat.Experimental_class.all_mouse_to_do_window['Code'].isin(selected_codes)]
        selectedwidnowsinfo=selectedwidnows[['Code','Lab_Number','Sex','Cage','Age','WeeksFromInjection','Line_Short','Projects','Labels_types']]
        selectedwidnowsinfo.to_excel(os.path.join(r'C:\Users\sp3660\Desktop\Temp_Excel_Files' ,'InfoForWindowTemplates_{0}.xlsx'.format(datetime.date.today().strftime("%Y%m%d") )))
        
    
    def update_done_window_button(self):         
        self.update_window_window=new_window_update_done_windows(self.gui_ref)
        self.update_window_window.wait_window()
        button_update_database(self.gui_ref)
#%% post op            
    def update_post_op_button(self):
        self.update_postop_window=new_window_post_op_injections(self)
        self.update_postop_window.wait_window()
        button_update_database(self.gui_ref)

    def saced_exp_animal_button(self):
        date_entry=self.frame3.saced_date_selection.get_date()
        selected_codes = list()
        selection = self.frame3.codes_lstbox.curselection()
        for i in selection:
            code = self.frame3.codes_lstbox.get(i)
            selected_codes.append(code)
        for mouse_code in selected_codes:
            self.gui_ref.MouseDat.Experimental_class.sac_experimental_mouse(mouse_code, date_entry)     
        print('All animlas saced')    
            
    def process_brain_button(self):    
        selected_mice=self.frame_brain_processing.selected_mice
        date_entry= self.frame_brain_processing.brain_processing_date_selection.get_date()
        self.update_process_brain_window=ProcessBrain(self.gui_ref, selected_mice, date_entry)     
        self.update_process_brain_window.wait_window()

        button_update_database(self.gui_ref)
        self.frame_brain_processing.brain_processing_selection_frame.update_lists(updated_cage_list= sorted(self.gui_ref.MouseDat.Experimental_class.all_experimental_all_info['Cage'].dropna().unique().to_numpy().tolist()))
