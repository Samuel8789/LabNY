# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 09:21:26 2021

@author: sp3660
"""

import datetime
import tkinter as tk
from tkinter import ttk
import os
from tkinter import StringVar, Listbox


from ...utils import open_multiple_df_in_tkinter_frame, button_update_database
from .new_window_plan_injections import  new_window_plan_injections
from .new_window_update_done_injections import  new_window_update_done_injections
from .new_window_update_done_windows import  new_window_update_done_windows
from .new_window_post_op_injections import  new_window_post_op_injections

 


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
                           'Experimental Tables']
        self.frames={}
        for i in range(len(self.frames_names)):
              self.frames[self.frames_names[i]]=ttk.Frame(self, borderwidth = 4)
           
        self.frames[self.frames_names[0]].grid(row=0, column=0, sticky="nswe") 
        self.frames[self.frames_names[1]].grid(row=0, column=1, sticky="nswe") 
        self.frames[self.frames_names[2]].grid(row=0, column=2, sticky="nswe") 
        self.frames[self.frames_names[3]].grid(row=1, column=0, columnspan=3, sticky="nswe") 
    

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
        self.frame1.codes_lstbox =Listbox(self.frame1, listvariable=self.to_do_injections_codes, selectmode='MULTIPLE', width=10, height=10)
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
        self.frame2.codes_lstbox =Listbox(self.frame2, listvariable=self.to_do_windows_codes, selectmode='MULTIPLE', width=10, height=10)
        self.to_do_windows_codes.set(self.gui_ref.MouseDat.Experimental_class.all_mouse_to_do_window['Code'].tolist())
        
        self.frame2.buttons[self.frame2.buttons_names[1]].grid(column=1, row=0)
        self.frame2.codes_lstbox.grid(column=1, row=1)

        self.frame2.buttons[self.frame2.buttons_names[2]].grid(column=0, row=11)

        
        
        
#%%TAB 3 'Mouse Experimental' 'Post-op'
        self.frame3= self.frames[self.frames_names[2]]  
        self.frame3.buttons={}
        self.frame3.buttons_names=['Update Post-op',
                                   'Saced Exp Animal'
                                   ]
        self.frame3.buttons_commands=[self.update_post_op_button,
                                      self.saced_exp_animal_button
                                      ]
        for i in range(len(self.frame3.buttons_names)):
              self.frame3.buttons[self.frame3.buttons_names[i]]= ttk.Button(self.frame3 , text=self.frame3.buttons_names[i], command=self.frame3.buttons_commands[i])


        self.saced_exp_codes=StringVar()    
        self.frame3.codes_lstbox =Listbox(self.frame3, listvariable=self.saced_exp_codes, selectmode='MULTIPLE', width=10, height=10)
        self.saced_exp_codes.set(self.gui_ref.MouseDat.Experimental_class.all_exp_animals_recovery['Code'].tolist())



        self.frame3.codes_lstbox.grid(column=0, row=1)
        self.frame3.buttons[self.frame3.buttons_names[0]].grid(column=0, row=0)
        self.frame3.buttons[self.frame3.buttons_names[1]].grid(column=0, row=11)







#%%TAB 3 'Mouse Experimental'  'Experimental Tables'
        self.frame4=self.frames[self.frames_names[3]]  
        self.experimental_recoveryanimalsbetter=self.gui_ref.MouseDat.remove_unnecesary_virusinfo_from_df(self.gui_ref.MouseDat.remove_unnecesary_genes_from_df(self.gui_ref.MouseDat.Experimental_class.all_exp_animals_recovery))
        df_dictionary={'All Experimental':self.gui_ref.MouseDat.Experimental_class.all_experimental_all_info,
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
        open_multiple_df_in_tkinter_frame(self,  self.frame4, df_dictionary)
        
    def plan_injection_button(self):
        new_window_plan_injections(self.gui_ref)
        button_update_database(self.gui_ref)        
        
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
            new_window_update_done_injections(self.gui_ref)
            button_update_database(self.gui_ref)
                            
#%% TO DO PLAN WINDOW            
    def plan_window_button(self):
        # new_window_plan_windows(self)
        button_update_database(self.gui_ref)    
#         cage=113
# selectedanimals=[4158,4182]
# MouseDat.Experimental_class.plan_new_window(cage=cage,lab_number_selected=selectedanimals)

        
    def prepare_window_templates_button(self):
      
        selected_codes = list()
        selection = self.frame1.codes_lstbox.curselection()
        for i in selection:
            code = self.frame1.codes_lstbox.get(i)
            selected_codes.append(code)

        selectedwidnows=self.gui_ref.MouseDat.Experimental_class.all_mouse_to_do_window.loc[self.gui_ref.MouseDat.Experimental_class.all_mouse_to_do_window['Code'].isin(selected_codes)]
        selectedwidnowsinfo=selectedwidnows[['Code','Lab_Number','Sex','Cage','Age','WeeksFromInjection','Line_Short','Projects','Labels_types']]
        selectedwidnowsinfo.to_excel(os.path.join(r'C:\Users\sp3660\Desktop\Temp_Excel_Files' ,'InfoForWindowTemplates_{0}.xlsx'.format(datetime.date.today().strftime("%Y%m%d") )))
        
    
    def update_done_window_button(self):         
        new_window_update_done_windows(self.gui_ref)
        button_update_database(self.gui_ref)
            
    def update_post_op_button(self):
        new_window_post_op_injections(self)
        button_update_database(self.gui_ref)

    def saced_exp_animal_button(self):
        selected_codes = list()
        selection = self.frame3.codes_lstbox.curselection()
        for i in selection:
            code = self.frame3.codes_lstbox.get(i)
            selected_codes.append(code)
        for mouse_code in selected_codes:
            self.gui_ref.MouseDat.Experimental_class.sac_experimental_mouse(mouse_code)     