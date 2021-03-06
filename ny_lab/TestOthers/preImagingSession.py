# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 16:22:07 2022

@author: sp3660
"""
import os
from pprint import pprint
import shutil
import itertools
import subprocess



class PreImagingSession():
    
    def __init__(self, sessiondate, mice):
        
        self.mice=mice
        self.sessiondate=sessiondate
        self.tempdir=r'C:\Users\sp3660\Desktop\SessionTemplatesDirectories'
        self.template=r'C:\Users\sp3660\Desktop\ImagingSessionDate' 
        self.transfer='scp -r '
        self.removedir='rmdir /s '
        self.copyfolder='Xcopy /E /I '

        self.WS1_IP='yustelab@128.59.247.96'
        self.WS2_IP='rylab@128.59.39.236'
        self.Prairire_IP='user@10.60.246.66'
        # icacls . /grant yustelab:(OI)(CI)F /T
        
        # yustelab@128.59.247.96:C:\Users\yustelab\Documents\Sam\ssh
        
        # ::# Set Key File Variable:
#     Set Key="C:\Users\sp3660\.ssh\id_rsa"

# ::# Remove Inheritance:
#     Icacls %Key% /c /t /Inheritance:d

# ::# Set Ownership to Owner:
#     :: # Key's within %UserProfile%:
#          Icacls %Key% /c /t /Grant %sp3660%:F

#     :: # Key's outside of %UserProfile%:
#          TakeOwn /F %Key%
#          Icacls %Key% /c /t /Grant:r %DESKTOP-OKLQSQS\sp3660%:F

# ::# Remove All Users, except for Owner:
#     Icacls %Key% /c /t /Remove:g "Authenticated Users" BUILTIN\Administrators BUILTIN Everyone System Users

# ::# Verify:
#     Icacls %Key%

# ::# Remove Variable:
#     set "Key="
        
#         ssh --% yustelab@128.59.247.96 icacls.exe "C:\Users\yustelab\.ssh\authorized_keys" /inheritance:r /grant "Administrators:F" /grant "SYSTEM:F"
# scp -r C:\Users\sp3660\Desktop\authorized_keys yustelab@128.59.247.96:C:\Users\yustelab\.ssh\
    
#        
        self.widefield_dir_WS1=r'C:\Users\yustelab\Documents\Sam\WideField'
        self.visstimsessions_dir_WS1=r'C:\Users\yustelab\Documents\Sam\VisualStim\MATLAB\Sessions'
        self.eyecam_dir_WS2=r'C:\Users\rylab\Documents\Sam\EyeCamera'
        self.stimdaq_dir_WS2=r'C:\Users\rylab\Documents\Sam\stim_scripts-master\behavior\NI_DAQ\output_data'
        self.prairireraw1=r'E:\Sam'
        self.prairireraw2=r'F:\Sam'
        self.prairireraw3=r'G:\Sam'
        self.permanent=r'F:\Projects\LabNY\Imaging'
        direchange='cd '
        self.prairiredrives=['e:','f:','g:']
        priairiredrive1='e: & cd '
        priairiredrive2='f: & cd '
        priairiredrive3='g: & cd '
        self.session_year=sessiondate[:4]

        self.main_dir=[self.widefield_dir_WS1, self.visstimsessions_dir_WS1, self.eyecam_dir_WS2, self.stimdaq_dir_WS2, self.prairireraw1, self.prairireraw2]
        self.sshfolder=r'G:\Projects\TemPrairireSSH'
        
        self.prairireraw=[r'E:\Sam',r'F:\Samuel',r'G:\Sam']
        print('prepare_empty_session'.upper())

        self.prepare_empty_session()
        print('prepare_session'.upper())

        self.prepare_session()
        print('copy_session_to_ssh'.upper())
        self.copy_session_to_ssh()
        
        print('copy sessionto permanent folder'.upper())
        print(f'Xcopy /E /I {os.path.join(self.sshfolder,self.sessiondate)} {os.path.join(self.permanent, self.session_year,self.sessiondate)}')
        
        self.remove_session_from_computer()
        print('ssh '+ self.WS1_IP)
        print('ssh '+ self.WS2_IP)
        print('scp -r wjyang@192.168.0.117:C:\\Users\\wjyang\\Documents\\Sam\\{} G:\\Projects\\TemPrairireSSH'.format(self.sessiondate))
        print(f'scp -r G:\\Projects\\TemPrairireSSH\\20220525Hakim\\Mice\\SPKU\\ToTrack wjyang@192.168.0.117:C:\\Users\\wjyang\\Documents\\Sam\\{self.sessiondate}\\Mice\\SPKU')
        
        print('remove session from ssh folder'.upper())
        print(f'rmdir /s {os.path.join(self.sshfolder,self.sessiondate)}')

        
    def prepare_empty_session(self):
        
        try:
            self.session_dir=shutil.copytree(self.template, os.path.join(self.tempdir,self.sessiondate), symlinks=False, ignore=None, ignore_dangling_symlinks=False, dirs_exist_ok=False)
        except:
            self.session_dir=os.path.join(self.tempdir,self.sessiondate)
            print('Alredy done')
                
            
        for mouse in self.mice:
            try:
            
                shutil.copytree(os.path.join(self.session_dir, r'Mice\SP__'), os.path.join(self.session_dir, 'Mice',mouse), symlinks=False, ignore=None, ignore_dangling_symlinks=False, dirs_exist_ok=False)
                
            except:
                    print('Mouse Alredy done')

        try   :             
            shutil.rmtree(os.path.join(self.session_dir, r'Mice\SP__'))
        except:
                print('Not mouse template')
        

        
    def prepare_session(self):
        
        all_ips=[self.Prairire_IP, self.WS1_IP, self.WS2_IP]
        
        all_target_dirs=[[self.prairireraw1, self.prairireraw2],[self.widefield_dir_WS1, self.visstimsessions_dir_WS1],[self.eyecam_dir_WS2, self.stimdaq_dir_WS2]]
        
        
        alltransfers=[[  self.transfer+self.session_dir + ' '+pc+':'+targdir  for targdir in all_target_dirs[i]] for i, pc in enumerate(all_ips)]
        flat_list = list(itertools.chain(*alltransfers))

        
        for tr in flat_list:
            print(tr)
            
    def transfer_sessions(self):
        all_ips=[self.Prairire_IP, self.WS1_IP, self.WS2_IP]
        all_target_dirs=[[self.prairireraw1, self.prairireraw2],[self.widefield_dir_WS1, self.visstimsessions_dir_WS1],[self.eyecam_dir_WS2, self.stimdaq_dir_WS2]]


        
        for i, pc in enumerate(all_ips):
            for targdir in all_target_dirs[i]:
                subprocess.run(["scp","-r", self.session_dir, pc+":"+targdir]) 


        

    def copy_session_to_ssh(self):
        new_sessions=[ os.path.join(i,self.sessiondate) for j,i in enumerate(self.main_dir) ]

        sshtransfers=[]
        for i,j in enumerate(new_sessions):
            if i in [0,1]:
                ip=self.WS1_IP
            elif i in [2,3]:
                ip=self.WS2_IP
            elif i in [4,5]:
                ip=self.Prairire_IP
            sshtransfers.append(self.transfer+ ip+':'+j+' '+self.sshfolder )
            print(sshtransfers[i])
            
    def remove_session_from_computer(self):
        
        
        trasnfertoslowprairire=self.copyfolder+os.path.join(self.prairireraw1,self.sessiondate) +' '+ os.path.join(self.prairireraw3,self.sessiondate)

        
        prairiredrivescd=[i+' & cd ' for i in self.prairiredrives]
        new_sessions=[ os.path.join(i,self.sessiondate) for j,i in enumerate(self.main_dir) ]


        print(  'ssh '+self.Prairire_IP)
        print(  trasnfertoslowprairire)


        for i,j in enumerate(new_sessions[4:]):
            print( prairiredrivescd[i]+j)

        
        
        removeprairiresession=[self.removedir+os.path.join(i,self.sessiondate) for i in self.prairireraw[:-1]]
        
        for i in removeprairiresession:
            print(i)
     
    def copy_ssh_to_permanent_dir(self):
        
        shutil.copytree(os.path.join(self.sshfolder, self.sessiondate), os.path.join(self.permanent, self.session_year,self.sessiondate), symlinks=False, ignore=None, ignore_dangling_symlinks=False, dirs_exist_ok=False)


      
        # visstims=r'C:\Users\sp3660\Documents\Github\LabNY\ny_lab\visual_stim\BehaviourCode\FinalWithOptoAndTriggers'        
        # trasnfervisstim=transfer +visstims+' ' +WS1_IP+':'+os.path.split(visstimsessions_dir_WS1)[0]

        # for i in new_sessions[:4]:
        #     print( direchange+i)
            
      

    

            

# '''
# --------------------------connections and betwen computer copying----------------------------------------
# --------------------VISSTIM/WIDEFILED/MIDDLE/WS1-------------------------
# ssh yustelab@128.59.247.96


# processing session(cvissitm)
#    mistmatch
# 	cd C:\Users\yustelab\Documents\Sam\VisualStim\MATLAB\visual_stim\stim_scripts_output\visual\FILENAME
# 	move C:\Users\yustelab\Documents\Sam\VisualStim\MATLAB\visual_stim\stim_scripts_output\visual\FILENAME C:\Users\yustelab\Documents\Sam\VisualStim\MATLAB\Sessions\2022XXXX\Mice\SPXX\UnprocessedVisStim
#    allen
# 	cd C:\Users\yustelab\Documents\Sam\VisualStim\MATLAB\Newest
# 	move C:\Users\yustelab\Documents\Sam\VisualStim\MATLAB\Newest\NAMEXXX C:\Users\yustelab\Documents\Sam\VisualStim\MATLAB\Sessions\XXXXXXXX\Mice\SPXX\UnprocessedVisStim
# 	move C:\Users\yustelab\Documents\Sam\VisualStim\MATLAB\Newest\220415_SPKH_FOV1_10MinSpontSpont_22_4_15_stim_data_17h_32m.mat C:\Users\yustelab\Documents\Sam\VisualStim\MATLAB\Sessions\20220415\Mice\SPKH\UnprocessedVisStim
# 	move C:\Users\yustelab\Documents\Sam\VisualStim\MATLAB\Sessions\20220331\220331_SPJZ_FOV1_10MinSpont1CellAblationAllenSessionB_22_3_31_stim_data_10h_50m.mat C:\Users\yustelab\Documents\Sam\VisualStim\MATLAB\Sessions\20220331\Mice\SPJZ\UnprocessedVisStim



# --------------------EYECAM/DAQ/RIGTH/WS2-------------------------
# ssh rylab@128.59.39.236

# processingsession
# 	
# 	move C:\Users\rylab\Documents\Sam\stim_scripts-master\behavior\NI_DAQ\output_data\2200330_SPJY_* C:\Users\rylab\Documents\Sam\stim_scripts-master\behavior\NI_DAQ\output_data\20220304\Mice\SPJY\UnprocessedDaq
# 	move 220415_SPKH_FOV1_10MinSpont_4_15_2022_17_21.mat C:\Users\rylab\Documents\Sam\stim_scripts-master\behavior\NI_DAQ\output_data\20220415\Mice\SPKH\UnprocessedDaq
# 	move C:\Users\rylab\Documents\Sam\stim_scripts-master\behavior\NI_DAQ\output_data\220414_SPMT_FOV1_AllenB_4_14_2022_11_25.mat C:\Users\rylab\Documents\Sam\stim_scripts-master\behavior\NI_DAQ\output_data\20220414\Mice\SPMT\UnprocessedDaq


# --------------------PRAIRIE-------------------------
# ssh user@10.60.246.66
# preparingsession
# 	e: & cd E:\Sam\20220303\Mice
# 	f: & cd F:\Samuel\202200303\Mice
# 	g: & cd G:\Sam\20220303\Mice

# processingsession
# 	Xcopy /E /I E:\Sam\20220415 G:\Sam\20220415
# 	Xcopy /E /I F:\Samuel\20220414 G:\Sam\20220414

# mkdir hope
# Xcopy /E /I SourceFolder DestinationFolder
# ren Folder NewFolderName
# rmdir /s 20220225
# copy 
# del
# move 20220216
# Xcopy /E /I H:\Projects\LabNY\20220306 G:\Projects\TemPrairireSSH\20220306 
# Xcopy /E /I H:\Projects\LabNY\20220314 G:\Projects\TemPrairireSSH\20220314
# ren SPJ_ SPXX
# Xcopy /E /I ImagingSession 20220217
# Xcopy /E /I ImagingSessionDate 20220217
# copy 220216_SPJZAllenSessionA_22_2_16_stim_data_21h_32m.mat ..\Sessions\20220216\Mice\SPJZ\UnprocessedVisStim\
# copy SessionA_timecount_fliptimes_finaltestAllenSessionC_22_2_16_stim_data_19h_18m.mat ..\Sessions\20220216\

# ren 20220213\Mice\SP_ SPJZ
# copy 220213_SPJZ_AllenCAllenSessionC_22_2_13_stim_data_21h_23m.mat ..\Sessions\20220213\Mice\SPJZ\UnprocessedVisStim\


# robocopy K:\Projects\LabNY\Full_Mice_Pre_Processed_Data\Mice_Projects\Interneuron_Imaging D:\Projects\LabNY\Full_Mice_Pre_Processed_Data\Mice_Projects\Interneuron_Imaging /MIR
# robocopy K:\Projects\LabNY\Full_Mice_Pre_Processed_Data\Mice_Projects\Interneuron_Optogenetics D:\Projects\LabNY\Full_Mice_Pre_Processed_Data\Mice_Projects\Interneuron_Optogenetics /MIR
# robocopy K:\Projects\LabNY\Full_Mice_Pre_Processed_Data\Mice_Projects\Collaborations D:\Projects\LabNY\Full_Mice_Pre_Processed_Data\Mice_Projects\Collaborations /MIR


# Xcopy /E /I G:\Projects\TemPrairireSSH\20220223 F:\Projects\LabNY\Imaging\2022\20220223
# '''
if __name__ == "__main__":
    # execute only if run as a script
    sessiondate='20220722'
    mice=[
    'SPOK',
    'SPOM',
    'SPOO',
]
    pressesion=PreImagingSession(sessiondate, mice)    
    # pressesion.copy_ssh_to_permanent_dir()
        

