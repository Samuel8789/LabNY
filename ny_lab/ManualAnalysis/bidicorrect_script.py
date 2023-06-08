# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 14:20:14 2021

@author: sp3660
"""
import os
import glob
import numpy as np
import tifffile
from bidicorrect_image import shiftBiDi, biDiPhaseOffsets
import matplotlib.pyplot as plt
import shutil

folder_path= r'C:\Users\sp3660\Desktop\Volumes'
files=os.listdir(folder_path)
minivolumes=sorted(glob.glob(folder_path+'\\20210917**'))


for minivolume in minivolumes:
        channel1_path=os.path.join(minivolume, 'Ch1')
        channel2_path=os.path.join(minivolume, 'Ch2')

        for file in glob.glob(minivolume+'\\**.tif'):
            if 'Ch1' in file:
                if not os.path.exists(channel1_path):
                    os.mkdir(channel1_path)
                shutil.move(file, channel1_path)
            if 'Ch2' in file:
                if not os.path.exists(channel2_path):
                    os.mkdir(channel2_path)
                shutil.move(file, channel2_path)
    
        for file in sorted(glob.glob(minivolume+'\\**')):
            if 'Ch' in file:
                channel=file
                numbers=[]
                tiles_numbers=[]
                for file in glob.glob(channel+'\\**'):
                    numbers.append(file[file.find('.ome')-2:file.find('.ome')])  
                    tiles_numbers.append(file[file.find('_Ch')-2:file.find('_Ch')])
                planes=len(set(numbers))       
                tiles_number=len(set(tiles_numbers))                
                for i in range(1,planes+1):
                    plane_path=os.path.join(channel, 'Plane_{}'.format(f"{i:02}"))
                    if not os.path.exists(plane_path):
                        os.mkdir(plane_path)
                    for file in glob.glob(channel+'\\**'):
                        if f"{i:02}" in file[file.find('.ome')-2:file.find('.ome')]:
                            shutil.move(file, plane_path)
                            
                for planes in glob.glob(channel+'\\**'):      
                    tiles=sorted(glob.glob(planes+'\\**.tif'))
                    if not os.path.exists( os.path.join(os.path.split(tiles[i])[0],'Corrected')):
                        os.mkdir(os.path.join(os.path.split(tiles[i])[0],'Corrected'))
                        
                    if len(glob.glob(os.path.join(os.path.split(tiles[i])[0],'Corrected')+'\\**.tif'))<tiles_number:
                        # input_arr=cm.load(tiles)
                        # if input_arr.shape.index(min(input_arr.shape))==0:
                        #     input_arr=np.reshape(input_arr, (input_arr.shape[1],input_arr.shape[2],input_arr.shape[0]), order='F')
                        for file in tiles:
                            input_arr=np.zeros((512,512,len(tiles)))
                            for i, file_name in enumerate(tiles):
                                with tifffile.TiffFile(file_name) as tffl:
                                    input_image= tffl.asarray()
                                    if len(input_image.shape)==4:
                                          input_image=input_image[0,0,:,:]   
                                    if len(input_image.shape)==3:
                                          input_image=input_image[0,:,:] 
                                    input_arr[:,:,i] = input_image
                                      
                                     
                        shifted_images=np.zeros(input_arr.shape).astype('float32')
                        for i in range(min(input_arr.shape)):
                            frame=input_arr[:,:,i]
                            BiDiPhase=biDiPhaseOffsets(frame)
                        
                            shifted_image=shiftBiDi(BiDiPhase, frame)
                            shifted_images[:,:,i]=shifted_image
                            tifffile.imsave( os.path.join(os.path.split(tiles[i])[0],'Corrected\\Tile_shifted_{}.tif'.format(f"{i:04}")),shifted_image)

    #%% one image
    
# image_path=r'C:\Users\sp3660\Desktop\Volumes\20210917_SPGT_Atlas_920_50024_607_without_175_205z_5z_1ol-008\Ch2\Plane_04\20210917_SPGT_Atlas_920_50024_607_without_175_205z_5z_1ol-008_Cycle00011_Ch2_000004.ome.tif'
# with tifffile.TiffFile(image_path) as tffl:
#     input_image= tffl.asarray()
#     tifffile.imsave(os.path.join(os.path.split(image_path)[0],'Corrected\\Tile_ORIGINAL_{}.tif'.format(f"{i:04}")),input_image)
#     BiDiPhase=biDiPhaseOffsets(input_image)
#     shifted_image=np.zeros(input_image.shape).astype('float32')                      
#     shifted_image=shiftBiDi(BiDiPhase, input_image)
#     i=11
#     tifffile.imsave(os.path.join(os.path.split(image_path)[0],'Corrected\\Tile_shifted_{}.tif'.format(f"{i:04}")),shifted_image)
#     if BiDiPhase!=0:
#         double_shifted_image=shiftBiDi(BiDiPhase*2, input_image)
#         tifffile.imsave(os.path.join(os.path.split(image_path)[0],'Corrected\\Tile_double_shifted_{}.tif'.format(f"{i:04}")),double_shifted_image)

#     else:
#         negative_shifted_image=shiftBiDi(-1, input_image)
#         positive_shifted_image=shiftBiDi(1, input_image)
#         double_negative_shifted_image=shiftBiDi(-2, input_image)
#         double_positive_shifted_image=shiftBiDi(2, input_image)
#         tifffile.imsave(os.path.join(os.path.split(image_path)[0],'Corrected\\Tile_negative_shifted_{}.tif'.format(f"{i:04}")),negative_shifted_image)
#         tifffile.imsave(os.path.join(os.path.split(image_path)[0],'Corrected\\Tile_positive_shifted_{}.tif'.format(f"{i:04}")),positive_shifted_image)
#         tifffile.imsave(os.path.join(os.path.split(image_path)[0],'Corrected\\Tile_double_negative_shifted_{}.tif'.format(f"{i:04}")),double_negative_shifted_image)
#         tifffile.imsave(os.path.join(os.path.split(image_path)[0],'Corrected\\Tile_double_positive_shifted_{}.tif'.format(f"{i:04}")),double_positive_shifted_image)

  

# pre_figs={}
# pre_axs={}
# post_figs={}
# post_axs={}
# for i in range(shifted_images.shape[-1]):
#         pre_figs[i]=plt.figure()
#         pre_axs[i]=pre_figs[i].add_subplot()
#         pre_axs[i].imshow(input_arr[:,:,i])
#         pre_axs[i].set_title('Pre')
#         post_figs[i]=plt.figure()th
#         post_axs[i]=post_figs[i].add_subplot()
#         post_axs[i].imshow(shifted_images[:,:,i])
#         post_axs[i].set_title('Post')

import imagej
from scyjava import jimport
import glob
import os
ij = imagej.init('sc.fiji:fiji')
FileSaver = jimport('ij.io.FileSaver')
#%%
plugin = 'Grid/Collection stitching'
volumes=r'C:/Users/sp3660/Desktop/Volumes'
for directory in glob.glob(volumes+'\\**\\Ch**\\PLane**\\Corrected'):
    args = {
        'type':['Grid: snake by rows'],
        'order':['Right & Down']   ,
        'grid_size_x':5 ,
        'grid_size_y':3 ,
        'tile_overlap':1 ,
        'first_file_index_i':0 ,
        'directory':directory,
        'file_names':'Tile_shifted_00{ii}.tif' ,
        'output_textfile_name':'TileConfiguration.txt' ,
        'fusion_method':['Linear Blending'] ,
        'regression_threshold':0.30 ,
        'max/avg_displacement_threshold':2.50 ,
        'absolute_displacement_threshold':3.50 ,
        'compute_overlap':False,
        'subpixel_accuracy':True,
        'computation_parameters':['Save memory (but be slower)'],
        'image_output':['Fuse and display'],   
    }
    
    ij.py.run_plugin(plugin, args)
    imp = ij.py.active_image_plus()
    fs = FileSaver(imp)
    fs.saveAsTiff( os.path.join(directory,'Mosaic_1%'))
