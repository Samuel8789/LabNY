# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 14:21:42 2021

@author: sp3660
"""

import numpy as np
import os
import glob
import gc
import sys
import shutil
import caiman as cm

# from OnACID import run_on_acid
# from metadata import Metadata
# from motionCorrectedKalman import MotionCorrectedKalman
# from summaryImages import SummaryImages
# import caiman as cm
try:
    from .OnACID import run_on_acid
    from .metadata import Metadata
    from .caimanResults import CaimanResults
except:
    from standalone.OnACID import run_on_acid
    from standalone.metadata import Metadata
    from standalone.caimanResults import CaimanResults
    

# from .motionCorrectedKalman import MotionCorrectedKalman
# from .summaryImages import SummaryImages
import logging 
module_logger = logging.getLogger(__name__)
from caiman.source_extraction import cnmf as cnmf
import paramiko
import pickle
import time
from datetime import datetime

class CaimanExtraction():
    
    def __init__(self, bidishifted_movie_path=None,
                 metadata_file_path=None,
                 temporary_path=None,
                 first_pass_mot_correct=False, 
                 save_mot_correct=False,
                 metadata_object=None,
                 force_run=False,
                 dataset_object=None, 
                 deep=False,
                 new_parameters: dict=False,
                 galois=False):
        
        module_logger.info('Instantiating ' +__name__)
        self.temp_path=r'C:\Users\sp3660\Desktop\CaimanTemp'
        self.galoistempdir='/home/sp3660/Desktop/caiman_temp'
        self.cnm_object=None
        self.dataset_object=dataset_object
        self.deep=deep
        self.metadata_object=metadata_object
        self.temporary_path=temporary_path
        self.bidishifted_movie_path=bidishifted_movie_path
        self.metadata_file_path=metadata_file_path
        self.save_mot_correct=save_mot_correct
        self.force_run=force_run
        self.first_pass_mot_correct=first_pass_mot_correct
        self.check_caiman_files()
        self.check_motion_corrected_on_acid()
        self.eliminate_caiman_extra_from_mmap()

     
  
#%%
        if self.bidishifted_movie_path:
            self.get_info_from_metadata()
            self.set_caiman_parameters()
            
            if new_parameters:
                self.change_set_parameters(**new_parameters)

            if first_pass_mot_correct: 
                self.dataset_caiman_parameters['epochs']=1
                self.dataset_caiman_parameters['motion_correct']=True
                self.save_mot_correct=True
                self.dataset_caiman_parameters['use_cnn']=False
                
            if not galois:
                self.apply_caiman()
                self.check_motion_corrected_on_acid()
                self.check_caiman_files()
                self.check_shifts_files()
                self.remove_unclipped_issue_shifted_movies()
                
            elif galois:
                self.run_galois_caiman()
                self.check_motion_corrected_on_acid()
                self.check_caiman_files()
                self.check_shifts_files()
                self.remove_unclipped_issue_shifted_movies()
            
        elif self.temporary_path:
            self.check_motion_corrected_on_acid()
            self.check_caiman_files()
            self.check_shifts_files()
            # self.load_cnmf_object()
            # self.unload_cnmf_object()
            
            
    def temp_file_to_fast_disk(self):
        if not self.movie_slice.any():
            self.original_path= self.dataset_caiman_parameters['fnames']
            shutil.copyfile(self.dataset_caiman_parameters['fnames'],os.path.join(self.temp_path,os.path.split(self.dataset_caiman_parameters['fnames'])[1]))
            self.dataset_caiman_parameters['fnames']=os.path.join(self.temp_path,os.path.split(self.dataset_caiman_parameters['fnames'])[1])
        else:
            mov=cm.load(self.dataset_caiman_parameters['fnames'])
            self.original_path= self.dataset_caiman_parameters['fnames']
            mov=mov[self.movie_slice,:,:]
            mov.save(os.path.join(self.temp_path,self.good_filename+'.mmap'))
            self.dataset_caiman_parameters['fnames']=glob.glob(os.path.join(self.temp_path,self.good_filename)+'**.mmap')[0]
            
        
    def copy_fast_results_remove_all(self):
        
        caiman_file=glob.glob(self.temp_path+'\**.hdf5')
        motcorrectfile=glob.glob(self.temp_path+'\**MC_OnACID**.mmap')
        shiftsfile=glob.glob(self.temp_path+'\**shifts**.pkl')
        if caiman_file:
            shutil.copyfile(caiman_file[0],os.path.join(os.path.split(self.original_path)[0],os.path.split(caiman_file[0])[1]))
        if motcorrectfile:
            shutil.copyfile(motcorrectfile[0],os.path.join(os.path.split(self.original_path)[0],os.path.split(motcorrectfile[0])[1]))
        if shiftsfile:
            shutil.copyfile(shiftsfile[0],os.path.join(os.path.split(self.original_path)[0],os.path.split(shiftsfile[0])[1]))


        
        filelist = glob.glob(os.path.join(self.temp_path, "*"))
        for f in filelist:
            os.remove(f)

    def eliminate_caiman_extra_from_mmap(self) :   
        caiman_filename=None
        if self.bidishifted_movie_path:
            self.mmap_directory, caiman_filename=os.path.split(self.bidishifted_movie_path) 

        elif self.mc_onacid_path:
            self.mmap_directory, caiman_filename=os.path.split(self.mc_onacid_path) 
         
        if caiman_filename:
            if caiman_filename.find('_d1_')!=-1:
                self.good_filename=caiman_filename[:caiman_filename.find('_d1_')]   
                self.caiman_extra=caiman_filename[caiman_filename.find('_d1_'):caiman_filename.find('_mmap')-4]   
            else:
                self.good_filename=os.path.splitext(caiman_filename)[0]  

    def get_info_from_metadata(self):

        if (not self.metadata_object) and self.metadata_file_path :
            self.metadata=Metadata(aq_metadataPath=self.metadata_file_path)
        elif self.metadata_object:
            self.metadata=self.metadata_object
        elif  self.dataset_object :
            self.metadata=  self.dataset_object.metadata
        try:    
            module_logger.info('checking metadata for caiman ' + self.bidishifted_movie_path )
            self.objective='Defaul Fat25x'
            self.halfsize=5  
            self.volume_period =10
            if self.metadata.imaging_metadata:
                datasetmeta=self.metadata.imaging_metadata
                         
                self.objective=self.metadata.imaging_metadata[0]['Objective']
                self.framePeriod=float(datasetmeta[0]['framePeriod'])
                self.rastersPerFrame=int(datasetmeta[0]['RasterAveraging'])
                self.number_planes=datasetmeta[1]['PlaneNumber']
                if self.number_planes=='Single':
                    self.number_planes=1
                    self.volume_period=self.framePeriod*self.rastersPerFrame
                
                if self.number_planes>1:
                    self.etl_frame_period=float(datasetmeta[2][0][0]['framePeriod'])
                    self.plane_period=float(self.framePeriod*self.rastersPerFrame)
                    self.volume_period=self.etl_frame_period*self.number_planes
                    
            elif self.metadata.translated_imaging_metadata:
                self.objective=self.metadata.translated_imaging_metadata['Objective']
                self.volume_period=self.metadata.translated_imaging_metadata['FinalVolumePeriod']
                
                
            if self.objective=='MBL Olympus 20x':
                self.halfsize=4
            elif '25' in self.objective:
               self.halfsize=5          
               if '20x' in self.bidishifted_movie_path:
                    self.halfsize=4  
        except:
            module_logger.exception('No metdata found ' + self.bidishifted_movie_path )

                # 1/(float(datasetmeta[2][1][0]['absoluteTime'])-float(datasetmeta[2][0][0]['absoluteTime']))
                
    def change_set_parameters(self,**kwargs):
        
        for param,value in kwargs.items():
            if param=='movie_slice':
                self.movie_slice=value
            else:
                self.dataset_caiman_parameters[param]=value

       
                
                
    def set_caiman_parameters(self):
        
        self.movie_slice=np.empty(0)
  
        fr = 1/self.volume_period  # frame rate (Hz) 3pl + 4ms = 15.5455
        decay_time = 0.2# 2 for s 0.5 for f # approximate length of transient event in seconds
        gSig = (self.halfsize,self.halfsize)  # expected half size of neurons
        p = 2  # order of AR indicator dynamics
        min_SNR = 1   # minimum SNR for accepting new components
        ds_factor = 1  # spatial downsampling factor (increases speed but may lose some fine structure)
        gnb = 2  # number of background components
        gSig = tuple(np.ceil(np.array(gSig) / ds_factor).astype('int')) # recompute gSig if downsampling is involved
        mot_corr = False  # flag for online motion correction
        pw_rigid = False  # flag for pw-rigid motion correction (slower but potentially more accurate)
        max_shifts_online = 10  # maximum allowed shift during motion correction
        sniper_mode = True  # use a CNN to detect new neurons (o/w space correlation)
        rval_thr = 0.8  # soace correlation threshold for candidate components
        # set up some additional supporting parameters needed for the algorithm
        # (these are default values but can change depending on dataset properties)
        init_batch = 500 # number of frames for initialization (presumably from the first file)
        K = 1  # initial number of components
        epochs = 3 # number of passes over the data
        show_movie = False # show the movie as the data gets processed
        merge_thr = 0.8
        use_cnn = True  # use the CNN classifier
        min_cnn_thr = 0.90  # if cnn classifier predicts below this value, reject
        cnn_lowest = 0.3  # neurons with cnn probability lowe
        fudge_factor=0.99 #defqault is 0.96


        self.dataset_caiman_parameters = {'fnames': self.bidishifted_movie_path,
                                           'fr': fr,
                                           'decay_time': decay_time,
                                           'gSig': gSig,
                                           'p': p,
                                           'min_SNR': min_SNR,
                                           'rval_thr': rval_thr,
                                           'merge_thr': merge_thr,
                                           'ds_factor': ds_factor,
                                           'nb': gnb,
                                           'motion_correct': mot_corr,
                                           'init_batch': init_batch,
                                           'init_method': 'bare',
                                           'normalize': True,
                                           'sniper_mode': sniper_mode,
                                           'K': K,
                                           'max_shifts_online': max_shifts_online,
                                           'pw_rigid': pw_rigid,
                                           'dist_shape_update': True,
                                           'min_num_trial': 10,
                                           'show_movie': show_movie,
                                           'epochs':epochs,
                                           'use_cnn': use_cnn,
                                           'min_cnn_thr': min_cnn_thr,
                                           'cnn_lowest': cnn_lowest,
                                           'fudge_factor':fudge_factor
                                            }
        
    def apply_caiman(self): 
        # this is if there is already a ciamna file
        if self.caiman_path and self.mc_onacid_path:
        
                if not self.force_run:
                    module_logger.info('Caiman not run, files already there ' + self.bidishifted_movie_path )
                    self.load_cnmf_object()

                elif self.force_run and self.caiman_path==self.caiman_custom_path and self.mc_onacid_path==self.mc_onacid_custom_path :
                    module_logger.info('Caiman not run, files already there with custom file' + self.bidishifted_movie_path )
                    self.load_cnmf_object()
                
                elif self.force_run and self.caiman_path==self.caiman_full_path and self.mc_onacid_path==self.mc_onacid_full_path and not self.deep:
                    module_logger.info('Rerunning caiman with custom length movie ' + self.bidishifted_movie_path )
        
                    try:
                        self.temp_file_to_fast_disk()
                        self.cnm_object=run_on_acid(self, self.dataset_caiman_parameters, mot_corretc=self.save_mot_correct, save_mot_correct=self.save_mot_correct, initial_shallow=True)
                        self.copy_fast_results_remove_all()
                        # self.remove_unclipped_issue_shifted_movies()
                    except:
                            module_logger.exception('Something wrong with On Acid processing ' + self.bidishifted_movie_path )

                elif self.force_run and self.deep:
                    module_logger.info('Running deep caiman ' + self.bidishifted_movie_path )
                    self.dataset_caiman_parameters['fnames']=self.mc_onacid_path
        
                    try:
                        self.temp_file_to_fast_disk()
                        self.cnm_object_deep=run_on_acid(self, self.dataset_caiman_parameters, mot_corretc=self.save_mot_correct, save_mot_correct=self.save_mot_correct)
                        self.copy_fast_results_remove_all()
                        # self.remove_unclipped_issue_shifted_movies()
                    except:
                            module_logger.exception('Something wrong with deep On Acid ' + self.bidishifted_movie_path )
        #this is the initial extraction with motion correction            
        else:  
            try:
                self.temp_file_to_fast_disk()
                self.cnm_object=run_on_acid(self, self.dataset_caiman_parameters, mot_corretc=self.save_mot_correct, save_mot_correct=self.save_mot_correct, initial_shallow=True)
                self.copy_fast_results_remove_all()
                self.check_motion_corrected_on_acid()
                self.check_caiman_files()

            except:
                    module_logger.exception('Something wrong with On Acid processing' + self.bidishifted_movie_path )

    def load_cnmf_object(self):
        try:
            if self.caiman_path:
                if os.path.isfile(self.caiman_path):
                    self.cnm_object=cnmf.cnmf.load_CNMF(self.caiman_path)
        except:
            module_logger.exception('Could not load cnmf object' + self.temporary_path )

    def unload_cnmf_object(self):
             
        if self.cnm_object:
            del self.cnm_object
            gc.collect()
            # sys.stdout.flush()
            self.cnm_object=None

    def check_motion_corrected_on_acid(self):
          self.mc_onacid_custom_pats=[]
          self.mc_onacid_full_paths=[]
          self.mc_onacid_custom_path=None
          self.mc_onacid_full_path=None
          self.mc_onacid_path=None
          
          self.mc_onacid_full_paths=glob.glob(self.temporary_path+'\\**Movie_MC_OnACID_d1**.mmap')
          self.mc_onacid_custom_paths=glob.glob(self.temporary_path+'\\**end_MC_OnACID_d1**.mmap')

          if self.mc_onacid_full_paths:
              self.mc_onacid_full_path= self.mc_onacid_full_paths[0]
          if self.mc_onacid_custom_paths:
              self.mc_onacid_custom_path= self.mc_onacid_custom_paths[0]
          
          if self.mc_onacid_custom_path:  
              self.mc_onacid_path=self.mc_onacid_custom_path
          elif self.mc_onacid_full_path:
              self.mc_onacid_path=self.mc_onacid_full_path

    def check_caiman_files(self):

        self.caiman_custom_pats=[]
        self.caiman_full_paths=[]
        self.caiman_custom_path=None
        self.caiman_full_path=None
        self.caiman_path=None
        
        self.caiman_full_paths=sorted(glob.glob(self.temporary_path+'\\**Movie_MC_OnACID_**.hdf5'), key=os.path.getmtime) 
        self.caiman_custom_paths=sorted(glob.glob(self.temporary_path+'\\**end_MC_OnACID_**.hdf5'), key=os.path.getmtime) 
        
        self.caiman_sorted_files=sorted(glob.glob(self.temporary_path+'\\**sort.mat'), key=os.path.getmtime) 

   
        if  self.first_pass_mot_correct:
            indxx=0
        elif not self.first_pass_mot_correct:
            indxx=-1

        if self.caiman_full_paths:
            self.caiman_full_path= self.caiman_full_paths[indxx]
            self.all_caiman_full_paths=self.caiman_full_paths             
        if self.caiman_custom_paths:
            self.caiman_custom_path= self.caiman_custom_paths[indxx]
            self.all_caiman_full_paths=self.caiman_custom_paths
            
            
        if self.caiman_custom_path:  
            self.caiman_path=self.caiman_custom_path
        elif self.caiman_full_path:
            self.caiman_path=self.caiman_full_path    
            
        if self.caiman_path and len(self.caiman_path)>200:
            src_path=self.caiman_path
            dst_path=os.path.join(os.path.split(self.caiman_path)[0],'shortened_' +self.caiman_path[self.caiman_path.find('MC_OnACID'):])
            if not os.path.isfile(dst_path):
                print('shortening ' + self.caiman_path)
                shutil.copy(src_path, dst_path)
            self.caiman_path=dst_path
                
            
            
            
            
    def check_shifts_files(self):
    
         self.caiman_shifts_custom_pats=[]
         self.caiman_shifts_full_paths=[]
         self.caiman_shifts_custom_path=None
         self.caiman_shifts_full_path=None
         self.caiman_shifts_path=None
         
         self.caiman_shifts_full_paths=glob.glob(self.temporary_path+'\\**Movie_MC_OnACID_shifts**.pkl') 
         self.caiman_shifts_custom_paths=glob.glob(self.temporary_path+'\\**end_MC_OnACID_shifts**.pkl') 

         if self.caiman_shifts_full_paths:
             self.caiman_shifts_full_path= self.caiman_shifts_full_paths[0]
         if self.caiman_shifts_custom_paths:
             self.caiman_shifts_custom_path= self.caiman_shifts_custom_paths[0]
         
         if self.caiman_shifts_custom_path:  
             self.caiman_shifts_path=self.caiman_shifts_custom_path
         elif self.caiman_shifts_full_path:
             self.caiman_shifts_path=self.caiman_shifts_full_path            
                   
    def remove_unclipped_issue_shifted_movies(self):
        module_logger.info('removing unclipped ')

        if self.caiman_custom_path and  self.caiman_full_path :
            if os.path.isfile(self.caiman_full_path):
                os.remove(self.caiman_full_path)
        if self.mc_onacid_custom_path and  self.mc_onacid_full_path :
            if os.path.isfile(self.mc_onacid_full_path):
                os.remove(self.mc_onacid_full_path)
                
        if self.caiman_shifts_custom_path and  self.caiman_shifts_full_path :
            if os.path.isfile(self.caiman_shifts_full_path):
                os.remove(self.caiman_shifts_full_path)
          


    def load_results_object(self, caiman_file_path=None):    
        if not caiman_file_path:
            caiman_file_path=self.caiman_path     
        self.CaimanResults_object=CaimanResults(caiman_file_path,  dataset_object=self.dataset_object, caiman_object=self)
        
        
    def run_galois_caiman(self):
        
        self.cnm_object=None
       
        filetocopy= self.dataset_caiman_parameters['fnames']
        scripttocoy=r'C:\Users\sp3660\Documents\Github\LabNY\ny_lab\data_pre_processing\galois_caiman.py'
        paramstocopy= os.path.join( self.temp_path,'parameter_dict.pkl' )

        destination=self.galoistempdir+'/'+ os.path.split(filetocopy)[1]
        scriptdest= self.galoistempdir+'/'+ os.path.split(scripttocoy)[1]
        paramsdest= self.galoistempdir+'/'+ os.path.split(paramstocopy)[1]
        
        dirosavehdf5=os.path.split(filetocopy)[0]
        
        self.dataset_caiman_parameters['fnames']=destination
        self.dataset_caiman_parameters['pipeline']='onacid'#'cnmf' 'onacid'
        self.dataset_caiman_parameters['refit']=False

        with open(paramstocopy, 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump( self.dataset_caiman_parameters, f, pickle.HIGHEST_PROTOCOL)
               

        with paramiko.SSHClient() as ssh:
            
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect("192.168.0.244", username="sp3660",
                        key_filename=os.path.join(os.path.expanduser('~'), ".ssh", "galois"))

            start_transfer=time.time()
            sftp = ssh.open_sftp() 
            # sftp.put(filetocopy, destination)
            sftp.put(scripttocoy, scriptdest)
            sftp.put(paramstocopy, paramsdest)
            sftp.close()
            transfer_duration = time.time() - start_transfer
            print('Transfer Finsihed: {}'.format(transfer_duration/60))
            
            start_processing=time.time()
            
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print('Start Processing: {}'.format(current_time))
            cmd_run_caiman="source ~/miniconda3/bin/activate caiman; cd Desktop/caiman_temp ;python galois_caiman.py" 
            stdin, stdout, stderr = ssh.exec_command(cmd_run_caiman)
            channel = stdout.channel
            status = channel.recv_exit_status()
            err=stderr.readlines()

            processing_duration = time.time() - start_processing
            print('Processing Finsihed: {}'.format(processing_duration/60))
            
            cmd="ls  '/home/sp3660/Desktop/caiman_temp' | grep \.hdf5" 
            stdin, stdout, stderr = ssh.exec_command(cmd)
            channel = stdout.channel
            status = channel.recv_exit_status()
            file=stdout.readlines()
            fullfilepath=self.galoistempdir+'/'+ file[0][:-1]
            hdf5destination=os.path.join(dirosavehdf5, os.path.split(fullfilepath)[1])
            
            sftp = ssh.open_sftp() 
            sftp.get(fullfilepath, hdf5destination)
            sftp.close()
            
            self.cnm_object=cnmf.cnmf.load_CNMF(hdf5destination)
            
            
            cmd="rm -rf '/home/sp3660/Desktop/caiman_temp/*' " 
            stdin, stdout, stderr = ssh.exec_command(cmd)
            
            filelist = glob.glob(os.path.join(self.temp_path, "*"))
            for f in filelist:
                os.remove(f)
    
       

if __name__ == "__main__":
    # temporary_path='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\StandAloneDataset\SPIK3planeallen\Plane3'
    # # image_sequence_path=os.path.join(temporary_path,'210930_SPKI_2mintestvideo_920_50024_narrow_without-000_Shifted_Movie_MC_kalman.tiff')
    # metadata_file_path=os.path.join(temporary_path,'211007_SPIK_FOV2_AllenA_20x_920_50024_narrow_without-000.xml')
    # dataset_full_file_mmap_path=os.path.join(temporary_path,'211007_SPIK_FOV2_AllenA_20x_920_50024_narrow_without-000_Shifted_Movie_d1_256_d2_256_d3_1_order_F_frames_62499_.mmap')
    
    filename=r'220525_SPKU_FOV1_AllenB_940_25x_hakim_00001'
    temporary_path='\\\\?\\'+r'G:\Projects\TempProcessing\220525_SPKU_FOV1_AllenB_940_25x_hakim_00001'
    metadata_file_path=os.path.join(temporary_path,filename+'.xml')
    dataset_full_file_mmap_path=os.path.join(temporary_path, filename+'_Shifted_Movie_d1_256_d2_256_d3_1_order_F_frames_64416_.mmap')
    
    filename=r'211022_SPKS_FOV1_AllenA_20x_920_50024_narrow_with-000'
    temporary_path='\\\\?\\'+r'C:\Users\sp3660\Desktop\TemporaryProcessing\StandAloneDataset\211022_SPKS_FOV1_AllenA_20x_920_50024_narrow_with-000\Plane1'
    dataset_full_file_mmap_path=os.path.join(temporary_path, filename+'_Shifted_Movie_d1_256_d2_256_d3_1_order_F_frames_226002_.mmap')

    
    
    metadata_file_path=os.path.join(temporary_path,filename+'.xml')
    dump='\\\\?\\'+r'C:\Users\sp3660\Desktop\CaimanTemp'
    CaimanExtr = CaimanExtraction(dataset_full_file_mmap_path, metadata_file_path, temporary_path=temporary_path)
    cnm=CaimanExtr.cnm_object




    
                
