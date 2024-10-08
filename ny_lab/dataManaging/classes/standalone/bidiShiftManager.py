# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 19:57:36 2021

@author: sp3660
"""
import caiman as cm
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pickle
import glob
import gc
import shutil 
import scipy.signal as sg
import scipy.stats as st

import logging 
from ScanImageTiffReader import ScanImageTiffReader
import matplotlib

# from bidicorrect_image import shiftBiDi, biDiPhaseOffsets
try:
    from .bidicorrect_image import shiftBiDi, biDiPhaseOffsets
except:
    from bidicorrect_image import shiftBiDi, biDiPhaseOffsets


module_logger = logging.getLogger(__name__)

class BidiShiftManager:
    
    def __init__(self, caiman_movie=None, expanded_dataset_name=None, dataset_full_file_mmap_path=None, 
                 dataset_image_sequence_path=None, bidiphase_file_path=None, shifted_movie_path=None,
                 temporary_path=None, dataset_object=None, fullscripttemp=False, 
                 raw_dataset_object=None,
                 custom_start_end=False, force=False):
        
        self.start_end_flag=False
        self.raw_dataset_object=raw_dataset_object
        self.caiman_movie=caiman_movie
        self.dataset_full_file_mmap_path=dataset_full_file_mmap_path
        self.dataset_image_sequence_path=dataset_image_sequence_path
        self.fullscripttemp=fullscripttemp
        self.bidiphase_file_path=bidiphase_file_path
        self.shifted_movie_path=shifted_movie_path
        self.frame_start=0
        self.frame_end=-1
        self.custom_start_end=custom_start_end
        self.temporary_path=temporary_path
        self.shifted_movie=np.array([False])
        self.image_sequence=np.array([False])
        self.meanmov=np.array([False])
        self.m_mean=np.array([False])
        self.bidiphases=[]
        self.expanded_dataset_name=expanded_dataset_name
        self.dataset_object=dataset_object
        self.led_start_end_file=os.path.join(os.path.split(os.path.split(self.temporary_path)[0])[0],'LED_Start_End.txt')

        self.led_corrected_frame_start=None
        self.led_corrected_frame_end=None
        self.led_start_end_flag=None
        

        self.force=force
        
        
        
        if self.dataset_object:
            self.temporary_path=self.dataset_object.selected_dataset_mmap_path
            self.custom_start_end=True
         
            if self.temporary_path:

                self.check_shifted_movie_path()
                self.check_bidiphases_in_directory()
                self.get_proper_filenames() 
                     
                if self.custom_start_end:
                    module_logger.info('checking_custom_start_end')
                    self.read_custom_start_end()
                    module_logger.info('checking_led_start_end')
                    self.load_LED_tips()
                module_logger.info('organizing names')

                self.create_output_names_if_dont_exist()

                    # check if there is already a shifted file
                if self.shifted_movie_full_caiman_path and not force:
                    
                    if self.start_end_flag:
    
                        if self.shifted_movie_full_caiman_path==self.shifted_movie_custom_files_path:
                            module_logger.info('bidishifted movie there, not loaded')
                           
                            if self.bidiphase_file_path:
                                module_logger.info('bidishifs there, not loaded')
                                # self.load_bidiphases_from_file()
                            # self.remove_unclipped_issue_shifted_movies()
    
                        elif self.shifted_movie_full_caiman_path==self.shifted_movie_files_path:
                            module_logger.info('doing new custom bidishifting')
                            self.dataset_full_file_mmap_path=self.shifted_movie_full_caiman_path
                            self.load_dataset_from_mmap()
                            self.correct_bidi_movie() 
                            module_logger.info('saving files')
                            self.save_shifts()
                            self.save_shifted_movie()
                            module_logger.info('saving files')
                            self.save_shifts()
                            self.save_shifted_movie()
                            module_logger.info('rechecking output files')
                            self.check_shifted_movie_path()
                            self.remove_unclipped_issue_shifted_movies()
                            self.unload_shifted_movie()
                            self.unload_bidishifts()
                            


                    else:        
                        module_logger.info('bidishifted movie there, not loaded')
                        try:
                            module_logger.info('bidishifted movie path: '+ self.shifted_movie_full_caiman_path)
                        except:
                            module_logger.exception('bidishifted movie path doesn exist')
                        try:
                            module_logger.info('bidishifs path: '+  self.bidiphase_file_path)
                        except:
                            module_logger.exception('bidishifs path doesn exist')
                
                elif self.dataset_image_sequence_path:  
                    # self.dataset_image_sequence_path=self.selected_dataset_raw_path

                    if os.path.isfile(self.bidiphase_file_path):
                        module_logger.info('loading bidishifts')
                        self.load_bidiphases_from_file()
                        
                    module_logger.info('loading raw image sequence')
                    # self.load_dataset_from_image_sequence()
                    
                    
                    self.load_dataset_from_image_sequence()
                   
                    if self.dataset_object.associated_aquisiton.all_vis_stim_mat_files:
                        self.dataset_object.associated_aquisiton.load_vis_stim_info()
                        self.load_LED_tips()
                        self.manually_detect_led_synchs()
                        self.load_LED_tips()
                        self.border_clip_image_sequence()

                      
        
                    self.correct_bidi_movie() 
              
                        
                    module_logger.info('saving files')
                    self.save_shifts()
                    module_logger.info('saving bidishifte corrected movie')
                    self.save_shifted_movie()
                    module_logger.info('rechecking output files')
                    self.check_shifted_movie_path()
                    self.remove_unclipped_issue_shifted_movies()

                    self.unload_shifted_movie()
                    self.unload_bidishifts()
                
                self.load_mean_mov()
                self.save_mean_raw_movie_array()

        elif self.raw_dataset_object:
            # self.temporary_path=r'C:\Users\sp3660\Desktop\TemporaryProcessing'
            self.temporary_path=os.path.join(os.path.expanduser('~'),r'Desktop//TemporaryProcessing')

            pass

        else:
            
            if self.shifted_movie_path:
                # self.load_shifted_movie_from_mmap()
                self.load_bidiphases_from_file()
                self.shifted_movie_path=self.shifted_movie_path
                module_logger.info('bidishifted movie loaded')
                
            elif self.custom_start_end:
                if self.bidiphase_file_path:
                    self.load_bidiphases_from_file()
                self.load_dataset_from_image_sequence()
                # self.border_clip_image_sequence()
               
                self.correct_bidi_movie()
            else:

                if self.caiman_movie:
                    self.image_sequence=self.caiman_movie
                    
                elif self.dataset_full_file_mmap_path:
                    self.load_dataset_from_mmap()
         
                elif self.dataset_image_sequence_path:
                     self.load_dataset_from_image_sequence()   
                     self.border_clip_image_sequence()
                    
    
                if self.bidiphase_file_path:
                    self.load_bidiphases_from_file()
                  
                self.correct_bidi_movie()

            self.save_shifts()
            # self.plot_bidiphases()
            self.save_shifted_movie()
            self.unload_shifted_movie()
            self.unload_bidishifts()
        
#%% check what ther is in the directory     
    def load_mean_mov(self):
        if  self.mean_movie_path and os.path.isfile(self.mean_movie_path):
            self.meanmov=np.load(self.mean_movie_path)
        else:
            module_logger.info('no mean mov saved')    
            
        if self.meanmov.any():
            if self.led_start_end_flag:
                self.redo_mean_movie()
            
            
    def redo_mean_movie(self): 
        
        if len(self.meanmov)== self.led_corrected_frame_end-self.led_corrected_frame_start:
            os.remove(self.mean_movie_path)
            movie =cm.load(glob.glob(os.path.join( self.dataset_object.associated_aquisiton.database_acq_raw_path, 'Ch2Green','plane1',"**Ch2**.tif")))
            self.save_mean_raw_movie_array(redomov=movie)
        
        
    def save_mean_raw_movie_array(self, redomov=np.array([False])):
        
        if self.m_mean.any() and self.mean_movie_path:
            if not os.path.isfile(self.mean_movie_path):
                np.save(self.mean_movie_path,self.m_mean)
        else:
            
            if not self.led_start_end_flag:
                if self.m_mean.any() and self.mean_movie_path:
                    if not os.path.isfile(self.mean_movie_path):
                        np.save(self.mean_movie_path,self.m_mean)
                # elif self.shifted_movie.any() and self.mean_movie_path:
                #     if not os.path.isfile(self.mean_movie_path):
                #         m_mean = self.shifted_movie.mean(axis=(1, 2))
                #         np.save(self.mean_movie_path,m_mean)
            else:
                if self.image_sequence.any() and self.mean_movie_path:
                    if not os.path.isfile(self.mean_movie_path):
        
                        movie =cm.load(glob.glob(os.path.join( self.dataset_object.associated_aquisiton.database_acq_raw_path, 'Ch2Green','plane1',"**Ch2**.tif")))
                        m_mean = movie.mean(axis=(1, 2))
                        np.save(self.mean_movie_path,m_mean)
                    
                elif redomov.any() and self.mean_movie_path:
                    m_mean = redomov.mean(axis=(1, 2))
                    np.save(self.mean_movie_path,m_mean)
                
  
                

    def read_custom_start_end(self):
        start_end_file=os.path.join(os.path.split(os.path.split(self.temporary_path)[0])[0],'Start_End.txt')
        if os.path.isfile(start_end_file):
            self.start_end_flag=True
            with open(start_end_file) as f:
                lines = f.readlines()
            module_logger.info('changing custom start end')    
            start_end=[int(x) for x in lines]
            self.frame_start= start_end[0]
            self.frame_end=start_end[1]-1  
            # self.frame_end=start_end[1]  
            
    def load_LED_tips(self):
               
        if os.path.isfile(self.led_start_end_file):
            self.led_start_end_flag=True
            with open(self.led_start_end_file) as f:
                lines = f.readlines()
            led_start_end=[int(x) for x in lines]
            self.led_corrected_frame_start= led_start_end[0]
            self.led_corrected_frame_end=led_start_end[1] 
            
        return  self.led_corrected_frame_start, self.led_corrected_frame_end
    
  

        

    # def detect_LED_synchs(self):
        
            
    #     m_mean = self.image_sequence.mean(axis=(1, 2))
    #     scored=st.zscore(m_mean)
    #     # plt.plot(m_mean)
    #     dif=np.diff(scored)
    #     median=sg.medfilt(dif, kernel_size=1)
    #     rounded=np.round(median)
    #     # finsg start transition
    #     # transitions=np.where(abs(dif)>max(abs(dif))/2)[0]
    #     # transitions_median=np.where(abs(median)>max(abs(median))/2)[0]
    #     transitions_medina_rounded=np.where(abs(rounded)>max(abs(rounded))/2)[0]

    #     star_led=transitions_medina_rounded[transitions_medina_rounded<int(len(m_mean)/2)]
    #     end_led=transitions_medina_rounded[transitions_medina_rounded>int(len(m_mean)/2)]


    #     if len(star_led)>0:
    #         led_frame_start_end=star_led[-1]+1
    #         pad1=5
    #     else:
    #         led_frame_start_end=0
    #         pad1=0

    #     if len(end_led)>0:
    #         led_frame_end_start=end_led[0]+1
    #         pad2=5
    #     else:     
    #         led_frame_end_start=len(m_mean)+1
    #         pad2=0
            
    #     if not os.path.isfile(self.led_start_end_file):
    #         with open(self.led_start_end_file, 'w') as f:
    #             f.writelines((str( led_frame_start_end+pad1),'\n', str( led_frame_end_start-pad2)))
      

          
    def manually_detect_led_synchs(self):
        
        if (not self.led_corrected_frame_start)  and (not self.led_corrected_frame_end):

        

            scored=st.zscore(self.m_mean)
            dif=np.diff(scored)
            median=sg.medfilt(dif, kernel_size=1)
            rounded=np.round(median,decimals=2)
    
            
            f,axs=plt.subplots(2)
            axs[0].plot(self.m_mean,'k')
            axs[1].plot(scored,'r')
            axs[1].plot(median,'b')
            axs[1].plot(abs(rounded),'k')
            # mngr = plt.get_current_fig_manager()
            # mngr.window.setGeometry(50,100,2000, 1000)
            plt.show(block = False)
            plt.pause(0.01)
            
    
            
            raw_fluorescence_threshold = int(input('Integer raw florescence threshold\n'))
            # raw_fluorescence_threshold=7000
            # scored_threshold = int(input('Integer scored threshold\n'))
            # transition_up_threshold = int(input('Integer transitions threshold\n'))
            plt.close(f)
            
            no_led_start = int(input('Type 1 if no LED start signal\n'))
            no_led_end = int(input('Type 1 if no LED end signal\n'))
            # no_led_start=0
            # no_led_end=0
            
            led_on_frames=np.where(self.m_mean>raw_fluorescence_threshold)[0]
            movie_midpoint=int(np.floor(len(self.m_mean)/2))
    
    
            if (not no_led_start) or (not no_led_end):
                if not no_led_start:
                    
                    
                    led_on_frames_start=led_on_frames[led_on_frames<movie_midpoint]
                    led_on_frames_start_first=led_on_frames_start[0]
                    led_on_frames_start_last=led_on_frames_start[-1]
                    pad_frames=5
                    
                    prepad=np.arange(led_on_frames_start_first-pad_frames,led_on_frames_start_first)
                    postpad=np.arange(led_on_frames_start_last+1,led_on_frames_start_last+pad_frames+1)
                    
                    
                    extended_LED_frames = np.concatenate((led_on_frames_start,prepad,postpad))
                    extended_LED_frames.sort(kind='mergesort')
                    
                    
                    f,axs=plt.subplots(1)
                    axs.plot(extended_LED_frames,self.m_mean[extended_LED_frames],'k')
                    axs.plot(prepad,self.m_mean[prepad],'r')
                    axs.plot(postpad,self.m_mean[postpad],'y')
                    axs.plot(led_on_frames_start_first,self.m_mean[led_on_frames_start_first],'mo')
                    axs.plot(led_on_frames_start_last,self.m_mean[led_on_frames_start_last],'mo')
                    axs.set_xticks(extended_LED_frames)
                    axs.tick_params(direction='in' ,length=2,width=2)

                    axs.set_xticklabels(axs.get_xticks(), rotation = 90)
                    # mngr = plt.get_current_fig_manager()
                    # mngr.window.setGeometry(50,100,2000, 1000)
                    plt.show(block = False)
                    plt.pause(0.01)
                    
                    new_start_LED_start=int(input('Integer correct led start start\n'))
                    new_start_LED_end=int(input('Integer correct led start end\n'))
                    
                    peri_led_pad=5
                    plt.close(f)
                    
                    
                    movie_start_frame=new_start_LED_end+peri_led_pad
                    
                if not no_led_end:
                    led_on_frames_end=led_on_frames[led_on_frames>movie_midpoint]
                    
                    
                    led_on_frames_end_first=led_on_frames_end[0]
                    led_on_frames_end_last=led_on_frames_end[-1]
                    pad_frames=5
                    
                    prepad=np.arange(led_on_frames_end_first-pad_frames,led_on_frames_end_first)
                    postpad=np.arange(led_on_frames_end_last+1,led_on_frames_end_last+pad_frames+1)
                    
                    
                    extended_LED_frames = np.concatenate((led_on_frames_end,prepad,postpad))
                    extended_LED_frames.sort(kind='mergesort')
                    
                    
                    f,axs=plt.subplots(1)
                    axs.plot(extended_LED_frames,self.m_mean[extended_LED_frames],'k')
                    axs.plot(prepad,self.m_mean[prepad],'r')
                    axs.plot(postpad,self.m_mean[postpad],'y')
                    axs.plot(led_on_frames_end_first,self.m_mean[led_on_frames_end_first],'mo')
                    axs.plot(led_on_frames_end_last,self.m_mean[led_on_frames_end_last],'mo')
                    axs.set_xticks(extended_LED_frames)
                    axs.tick_params(direction='in' ,length=2,width=2)
                    axs.set_xticklabels(axs.get_xticks(), rotation = 90)

                    
                    # mngr = plt.get_current_fig_manager()
                    # mngr.window.setGeometry(50,100,2000, 1000)
                    plt.show(block = False)
                    plt.pause(0.01)
                    
                    
                    
                    new_finish_LED_start=int(input('Integer correct led end start\n'))
                    new_finish_LED_end=int(input('Integer correct led end end\n'))
                    plt.close(f)
                    peri_led_pad=5
                    
                    movie_end_frame=new_finish_LED_start-peri_led_pad
                    
                   
            
            
            if no_led_start:
                movie_start_frame=0
            if no_led_end:
                movie_end_frame=len(self.m_mean)
                
            movie_range=np.arange(movie_start_frame,movie_end_frame)
            f,axs=plt.subplots(1)
            axs.plot(self.m_mean,'k')
            axs.plot(movie_range,self.m_mean[movie_range],'r')
            axs.set_xticklabels(axs.get_xticks(), rotation = 90)
            # mngr = plt.get_current_fig_manager()
            # mngr.window.setGeometry(50,100,2000, 1000)
            plt.show(block = False)
            plt.pause(0.01)
            
            
            
            if not os.path.isfile(self.led_start_end_file):
                with open(self.led_start_end_file, 'w') as f:
                    f.writelines((str( movie_start_frame),'\n', str(movie_end_frame)))
            plt.close('all')
            self.led_corrected_frame_start= movie_start_frame
            self.led_corrected_frame_end=movie_end_frame
            
        else:
            pass                    
                
        
            

    
     
    def check_bidiphases_in_directory(self):
        self.bidiphase_custom_file_paths=[]
        self.bidiphase_full_file_paths=[]
        self.bidiphase_custom_file_path=None
        self.bidiphase_full_file_path=None
        self.bidiphase_file_path=None

        self.all_bidipahse_files=glob.glob(self.temporary_path+os.sep+'**Bidiphases**.pkl')
        # mmap_files=glob.glob(self.mmap_directory+os.sep+'**.mmap')
        self.bidiphase_custom_file_paths=[i for i in self.all_bidipahse_files if 'Bidiphases_custom' in i ]
        self.bidiphase_full_file_paths=[i for i in self.all_bidipahse_files if 'Bidiphases.' in i]
        
        if self.bidiphase_custom_file_paths:
            self.bidiphase_custom_file_path= self.bidiphase_custom_file_paths[0]
            self.bidiphase_file_path= self.bidiphase_custom_file_path    

        if self.bidiphase_full_file_paths:
            self.bidiphase_full_file_path= self.bidiphase_full_file_paths[0]
            self.bidiphase_file_path= self.bidiphase_full_file_path  

        
        
        
    def check_shifted_movie_path(self): 
        
        self.shifted_movie_custom_files_paths=[]
        self.shifted_movie_files_paths=[]
        self.shifted_movie_custom_files_path=None
        self.shifted_movie_files_path=None
        self.shifted_movie_full_caiman_path=None
        self.all_mmap_files=glob.glob(self.temporary_path+os.sep+'**.mmap')
        self.shifted_movie_files_paths=[i for i in self.all_mmap_files if 'Shifted_Movie_d1' in i ]
        self.shifted_movie_custom_files_paths=[i for i in self.all_mmap_files if 'Shifted_Movie_custom_start_end_d1' in i ]


        if self.shifted_movie_custom_files_paths:
            self.shifted_movie_custom_files_path= self.shifted_movie_custom_files_paths[0]
            self.shifted_movie_full_caiman_path= self.shifted_movie_custom_files_path    

        if self.shifted_movie_files_paths:
            self.shifted_movie_files_path= self.shifted_movie_files_paths[0]
            self.shifted_movie_full_caiman_path= self.shifted_movie_files_path    


    
 
            
#%% get proper variables 
    
    def get_proper_filenames(self):
        
        if self.dataset_full_file_mmap_path or self.shifted_movie_full_caiman_path  :       
            self.eliminate_caiman_extra_from_mmap()
            # self.check_bidiphases_in_directory()
        elif self.dataset_full_file_mmap_path: 
            first_frame_filename=os.path.split(glob.glob(self.dataset_image_sequence_path+os.sep+'**.tif')[0])[1]
            self.good_filename=first_frame_filename[0:first_frame_filename.find('_Cycle')]
            # self.check_bidiphases_in_directory()
        elif self.dataset_image_sequence_path:
            first_frame_filename=os.path.split(glob.glob(self.dataset_image_sequence_path+os.sep+'**.tif')[0])[1]
            self.good_filename=first_frame_filename[0:first_frame_filename.find('_Cycle')]+'_'+self.dataset_object.plane
            
        elif self.temporary_path: # this is to solbve a problem when no mmap is inn the dataset folder but we are reading the dataset as i didnt force to finsih the mmap befor, no I do it throiugh the app
            self.good_filename=os.path.split(os.path.split(os.path.split(os.path.split(self.temporary_path)[0])[0])[0])[1]
            pass
      
                        
    def eliminate_caiman_extra_from_mmap(self):
        # here detect if there is a raw mmap(legacy not done anymore, only save the shifted mmap)
        if self.dataset_full_file_mmap_path:
            self.mmap_directory, caiman_filename=os.path.split(self.dataset_full_file_mmap_path)
        # here detect if there is a raw mmap(legacy not done anymore, only save the shifted mmap), I have to aditional remove the shifted movie

        elif self.shifted_movie_full_caiman_path:
            self.mmap_directory, caiman_filename=os.path.split(self.shifted_movie_full_caiman_path)

        self.good_filename=caiman_filename[:caiman_filename.find('Shifted_Movie')-1]   
        # self.caiman_extra=caiman_filename[caiman_filename.find('_d1_'):caiman_filename.find('_mmap')-4]       
            
    def correc_name_duplication(self):
        # self.good_filename Shifted_Movie
        pass
        
    def create_output_names_if_dont_exist(self):
        self.bidiphase_file_path=None
        self.shifted_movie_path=None
        self.mean_movie_path=None

        
        if (not self.shifted_movie_full_caiman_path) or (not self.bidiphase_file_path):
            
            if self.temporary_path and not self.start_end_flag:
                self.bidiphase_file_path='.'.join( ['_'.join([os.path.join(self.temporary_path,self.good_filename),'Bidiphases']),'pkl'])
                self.shifted_movie_path='.'.join( ['_'.join([os.path.join(self.temporary_path,self.good_filename),'Shifted_Movie']),'mmap'])
                self.mean_movie_path='.'.join( ['_'.join([os.path.join(self.temporary_path,self.good_filename[:-7]),'mean_raw_Movie']),'npy'])
    
    
            elif self.start_end_flag:
                self.bidiphase_file_path='.'.join( ['_'.join([os.path.join(self.temporary_path,self.good_filename),'Bidiphases_custom_start_end']),'pkl'])  
                self.shifted_movie_path='.'.join( ['_'.join([os.path.join(self.temporary_path,self.good_filename),'Shifted_Movie_custom_start_end']),'mmap'])
                self.mean_movie_path='.'.join( ['_'.join([os.path.join(self.temporary_path,self.good_filename),'mean_raw_Movie_custom_start_end']),'npy'])
    
    
            elif self.mmap_directory:
                self.bidiphase_file_path='.'.join( ['_'.join([os.path.join(self.mmap_directory,self.good_filename),'Bidiphases']),'pkl']) 
                self.shifted_movie_path='.'.join( ['_'.join([os.path.join(self.mmap_directory,self.good_filename),'Shifted_Movie']),'mmap'])
                self.mean_movie_path='.'.join( ['_'.join([os.path.join(self.mmap_directory,self.good_filename),'mean_raw_Movie']),'npy'])

        elif self.force:
            if self.temporary_path and not self.start_end_flag:
                self.bidiphase_file_path='.'.join( ['_'.join([os.path.join(self.temporary_path,self.good_filename),'Force_Bidiphases']),'pkl'])
                self.shifted_movie_path='.'.join( ['_'.join([os.path.join(self.temporary_path,self.good_filename),'Force_Shifted_Movie']),'mmap'])
                self.mean_movie_path='.'.join( ['_'.join([os.path.join(self.temporary_path,self.good_filename[:-7]),'Force_mean_raw_Movie']),'npy'])
                
        else:
            if self.temporary_path and not self.start_end_flag:
                self.bidiphase_file_path='.'.join( ['_'.join([os.path.join(self.temporary_path,self.good_filename),'Bidiphases']),'pkl'])
                self.shifted_movie_path='.'.join( ['_'.join([os.path.join(self.temporary_path,self.good_filename),'Shifted_Movie']),'mmap'])
                self.mean_movie_path='.'.join( ['_'.join([os.path.join(self.temporary_path,self.good_filename[:-7]),'mean_raw_Movie']),'npy'])
                    

        
#%% load thing if existing files
    

    def load_bidiphases_from_file(self):   
        if os.path.isfile(self.bidiphase_file_path):
            module_logger.info('loading bidishifts')
            with open(self.bidiphase_file_path, "rb") as fp:   # Unpickling
              self.bidiphases = pickle.load(fp)

    def load_dataset_from_image_sequence(self):

        
        image_sequence_paths=sorted(glob.glob(self.dataset_image_sequence_path+os.sep+'**.tif'))
        
        module_logger.info('loading files')
        try:
            self.image_sequence=cm.load(image_sequence_paths[0])
            module_logger.info('Caiman loaded for first metadta file, sometimes it is the only that works for red or when all file ar in same direcotyr')
            
            if 'Ch1Red' in image_sequence_paths[0]:
                if len(image_sequence_paths)!= len(self.image_sequence):
                    try:
                        self.image_sequence=cm.load(image_sequence_paths)
                        module_logger.info('Caiman load sequence properly loaded')
                    except:
                        try:
                            self.image_sequence=cm.load(image_sequence_paths[0])[:,0,:,:]
                            module_logger.info('all files in single file')
                        except:
                            module_logger.exception('Check caiman loading, something wrong')
                else:
                    self.image_sequence= self.image_sequence[:,0,:,:]
            else:
                   
                if len(image_sequence_paths)!= len(self.image_sequence)  or self.image_sequence.shape[1]==2:      
                    module_logger.info('No full sequence from first file')
                    try:
                        self.image_sequence=cm.load(image_sequence_paths)
                        module_logger.info('Caiman load sequence properly loaded')
                        if len( self.image_sequence)==2*len(image_sequence_paths)-1:
                           self.image_sequence=self.image_sequence[:len(image_sequence_paths),:,:]
                  
                    except:
                        try:
                            self.image_sequence=cm.load(image_sequence_paths[0])[:,0,:,:]
                            module_logger.info('all files in single file')
                        except:
                            module_logger.exception('Check caiman loading, something wrong')
                            
                            
            self.m_mean = self.image_sequence.mean(axis=(1, 2))
            self.save_mean_raw_movie_array()
        except:                    
            module_logger.exception('No video loaded')



     
    def border_clip_image_sequence(self):
        
        if not self.start_end_flag:
            self.frame_end= len(self.image_sequence)    
            
        if len(self.image_sequence)>1:
            self.image_sequence= self.image_sequence[self.frame_start:self.frame_end,:,:]
        else:
            pass

       

          
        if not self.led_start_end_flag:
            self.led_corrected_frame_start=0
            self.led_corrected_frame_end= len(self.image_sequence)+1  
            
        if len(self.image_sequence)>1:
            self.image_sequence= self.image_sequence[self.led_corrected_frame_start:self.led_corrected_frame_end,:,:]
        else:
            pass
        

 
    def remove_optoled_frames(self):
        pass
            

    def load_dataset_from_mmap(self): 
        self.image_sequence
        self.image_sequence= self.image_sequence[self.frame_start:self.frame_end,:,:]

        
    def load_shifted_movie_from_mmap(self): 
        if self.shifted_movie_full_caiman_path:
            if os.path.isfile(self.shifted_movie_full_caiman_path):
                self.shifted_movie=cm.load(self.shifted_movie_full_caiman_path)

    

#%% do the processing

 


    def correct_bidi_movie(self): 
        if not self.bidiphases:
            module_logger.info('calculating bidiphases')
            self.calculate_bidi_shifts()  
        else:
            module_logger.info('bidiphases loaded')
        if not self.shifted_movie.any():
            module_logger.info('calculating bidishifted movie')
            self.shifted_movie=cm.movie(self.shift_images())
        else:
            module_logger.info('bidishifted movie loaded')

    def calculate_bidi_shifts(self):
        self.bidiphases=[]
        for i in range(self.image_sequence.shape[0]):
            BiDiPhase=biDiPhaseOffsets(self.image_sequence[i,:,:])
            self.bidiphases.append(BiDiPhase)
    
    def shift_images(self):
        shifted_images=np.zeros(self.image_sequence.shape).astype('float32')
        for i in range(self.image_sequence.shape[0]):
            shifted_images[i,:,:]=shiftBiDi(self.bidiphases[i][0], self.image_sequence[i,:,:])
        return shifted_images
    

    
#%% save the stuff
   
    def save_shifts(self):
        if self.bidiphases and self.bidiphase_file_path:
            if not os.path.isfile(self.bidiphase_file_path):
                with open(self.bidiphase_file_path, "wb") as fp:   #Pickling
                    pickle.dump(self.bidiphases, fp)
                    
                    

               
    def save_shifted_movie(self):
   
        if self.shifted_movie.any() and self.shifted_movie_path:
            if not os.path.isfile(self.shifted_movie_path):
                self.shifted_movie.save(self.shifted_movie_path ,to32=False)  


    def unload_shifted_movie(self):     
        module_logger.info('unloading bidishifted movies')

        if self.shifted_movie.any():
            del self.shifted_movie
            gc.collect()
            sys.stdout.flush()
            self.shifted_movie=np.array([False])
            
    def unload_bidishifts(self):
        if self.bidiphases:
            del self.bidiphases
            gc.collect()
            self.bidiphases=[]
        
       

#%% plotting and others           

    def plot_bidiphases(self):
        plt.figure()
        plt.plot(self.bidiphases)
        plt.figure()
        plt.hist(self.bidiphases, 50, density=True, alpha=0.75)
        
    def copy_results_to_new_directory(self, new_directory):        
        self.shifted_movie_path
        self.bidiphase_file_path
        
        shutil.copyfile(self.path_to_save_shifted_movie,   os.path.join(new_directory, os.path.split(self.path_to_save_shifted_movie)[1]))
        shutil.copyfile(self.bidiphase_file_path,          os.path.join(new_directory, os.path.split(self.bidiphase_file_path)[1]))

    def remove_unclipped_issue_shifted_movies(self):
        module_logger.info('removing unclipped ')

        if self.bidiphase_custom_file_path and  self.bidiphase_full_file_path :
            if os.path.isfile(self.bidiphase_full_file_path):
                os.remove(self.bidiphase_full_file_path)


        if self.shifted_movie_custom_files_path and  self.shifted_movie_files_path :
            if os.path.isfile(self.shifted_movie_files_path):
                os.remove(self.shifted_movie_files_path)
         
                
if __name__ == "__main__":
    
    # dataset_image_sequence_path=r'F:\Projects\LabNY\Imaging\2022\20220428\Mice\SPMT\FOV_1\Aq_1\220428_SPMT_FOV2_AllenA_25x_920_52570_570620_without-000\Ch2Green\plane1'
    # dataset_image_sequence_path=r'F:\Projects\LabNY\Imaging\2022\20220428\Mice\SPMT\FOV_1\SurfaceImage\Aq_1\220428_SPMT_FOV2_Surface_25x_920_52570_570620_without-000\Ch2Green\plane1'
    # dataset_image_sequence_path=r'F:\Projects\LabNY\Imaging\2022\20220428\Mice\SPMT\0CoordinateAcquisiton\Aq_1\220428_SPMT_0Coordinate_25x_940_52570_570620_wit-000\Ch2Green\plane1'
    temporary_path=r'C:\Users\sp3660\Desktop\TemporaryProcessing'
    
    
    dataset_image_sequence_path=r'F:\Projects\LabNY\Imaging\2023\20230307\Mice\SPRA\FOV_2\Aq_2\230307_SPRA_FOV2_19CellOptoScreen_dial1_25x_920_51020_63075-001\Ch1Red\plane1'


    bidihits = BidiShiftManager(dataset_image_sequence_path=dataset_image_sequence_path, temporary_path=temporary_path )

    # dataset_full_file_mmap_path=os.path.join(temporary_path,'210930_SPKI_2mintestvideo_920_50024_narrow_without-000_shifted_movie_d1_256_d2_256_d3_1_order_F_frames_3391_.mmap')
    # bidihits = BidiShiftManager(dataset_full_file_mmap_path=dataset_full_file_mmap_path )

    bidihits.image_sequence.zproject()


    