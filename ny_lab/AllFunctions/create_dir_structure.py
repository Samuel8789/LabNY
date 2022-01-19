# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 12:05:39 2021

@author: sp3660
"""
import os
# directory=r'G:\CodeTempRawData\LabData\Interneuron_Imaging\SPIH'
# structure={'surgeries',
#             'treatments',
#             'training',
#             'imaging',
#             'euthanasia',   
#             'histology',
#             'ex vivo',
#             'data'
            # }

# structure={'group1':'injection',
#                'group2':'cranial window',
#                }
def create_dir_structure(directory, structure):
    pass # For compatibility between running under Spyder and the CLI
    

    
#%%    
    if os.path.isdir(directory):
      structure_paths=[ os.path.join(directory, structure_dir) for structure_dir in os.listdir(directory)  if os.path.isdir(os.path.join(directory, structure_dir))]
      structure_names=[  structure_dir for structure_dir in os.listdir(directory) if os.path.isdir(os.path.join(directory, structure_dir)) ]
   
      for struc_element in structure:
          if not struc_element in structure_names:
              if not os.path.isdir(os.path.join(directory,struc_element)):
                  os.mkdir(os.path.join(directory,struc_element))
              
    elif not os.path.isdir(directory):
        for struc_element in structure:
              if not os.path.isdir(os.path.join(directory,struc_element)):
                  os.makedirs(os.path.join(directory,struc_element))

    
    
#%%
# This is to mask the differences between running this demo in Spyder
# versus from the CLI
if __name__ == "__main__":
    main()