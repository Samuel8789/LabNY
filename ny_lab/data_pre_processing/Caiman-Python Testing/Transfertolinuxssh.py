# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 11:51:14 2021

@author: sp3660
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 11:02:04 2021

@author: sp3660
"""
import caiman as cm
from PathStructure import PathStructure
import os
from tkinter import filedialog
from tkinter import *

folders_to_process=[]
number_of_folders=1
for videos in range(number_of_folders):
    root = Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory(parent=root,
                                      initialdir="F:\Imaging",
                                      title='Please select a directory')
    folders_to_process.append(folder_selected)
    
for i, video in enumerate(folders_to_process):
    video_name=os.path.basename(video)
    folder_selected_list = os.listdir(video)    

    directory_green=(video + os.sep + 'Ch2Green')
    gcampframes=os.listdir(directory_green)
    



#mv.play()
#%%.
import paramiko
from scp import SCPClient
host_ip = '192.168.0.244'
user = 'sp3660'
pw = 'Cajal1852'
port = 22

origin="F:\\Imaging\\2021\\20210111\\SingleImage-01112021-1307-000\\"
destination='//home//sp3660//CodeTempRawData//'
file='SingleImage-01112021-1307-000.env'

def createSSHClient(host_ip, port, user, pw):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host_ip, port, user, pw)
    return client
def scp_upload(ssh):
    scp = SCPClient(ssh.get_transport())
    print ("Fetching SCP data..")
    scp.put(origin, destination, recursive=True)
    print ("File download complete.")
    
def main():
    # Get the ssh connection
    global ssh
    ssh = createSSHClient(host_ip, port, user, pw)
    print ("Executing command...")

    # Command list
    ##run_list_dir(ssh)
    ##run_getmerge(ssh)
    ##scp_download(ssh)

    # Run MapReduce
    scp_upload(ssh)

    # Close ssh connection
    ssh.close()

if __name__ == '__main__':
    main()