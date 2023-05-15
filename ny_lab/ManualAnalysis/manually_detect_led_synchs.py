# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 11:00:03 2023

@author: sp3660
"""
import os
import matplotlib.pyplot as plt
import numpy as np

import scipy.signal as sg
import scipy.stats as st

def manually_detect_led_synchs(image_sequence):
    m_mean = image_sequence.mean(axis=(1, 2))
    scored=st.zscore(m_mean)
    dif=np.diff(scored)
    median=sg.medfilt(dif, kernel_size=1)
    rounded=np.round(median,decimals=2)

    
    f,axs=plt.subplots(2)
    axs[0].plot(m_mean,'k')
    axs[1].plot(scored,'r')
    axs[1].plot(median,'b')
    axs[1].plot(abs(rounded),'k')
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(50,100,2000, 1000)
    plt.show(block = False)
    plt.pause(0.001)
    

    
    raw_fluorescence_threshold = int(input('Integer raw florescence threshold\n'))
    scored_threshold = int(input('Integer scored threshold\n'))
    # transition_up_threshold = int(input('Integer transitions threshold\n'))
    plt.close(f)
    
    no_led_start = int(input('Type 1 if no LED signal\n'))
    no_led_end = int(input('Type 1 if no LED signal\n'))
    
    if not no_led_start and not no_led_end:
        led_on_frames=np.where(m_mean>raw_fluorescence_threshold)[0]
        
        movie_midpoint=int(np.floor(len(m_mean)/2))
        
        led_on_frames_start=led_on_frames[led_on_frames<movie_midpoint]
        led_on_frames_start_first=led_on_frames_start[0]
        led_on_frames_start_last=led_on_frames_start[-1]
        pad_frames=5
        
        prepad=np.arange(led_on_frames_start_first-pad_frames,led_on_frames_start_first)
        postpad=np.arange(led_on_frames_start_last+1,led_on_frames_start_last+pad_frames+1)
        
        
        extended_LED_frames = np.concatenate((led_on_frames_start,prepad,postpad))
        extended_LED_frames.sort(kind='mergesort')
        
        
        f,axs=plt.subplots(1)
        axs.plot(extended_LED_frames,m_mean[extended_LED_frames],'k')
        axs.plot(prepad,m_mean[prepad],'r')
        axs.plot(postpad,m_mean[postpad],'y')
        axs.plot(led_on_frames_start_first,m_mean[led_on_frames_start_first],'mo')
        axs.plot(led_on_frames_start_last,m_mean[led_on_frames_start_last],'mo')
        axs.set_xticks(extended_LED_frames,)
        axs.set_xticklabels(axs.get_xticks(), rotation = 45)
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(50,100,2000, 1000)
        plt.show(block = False)
        plt.pause(0.001)
        
        new_start_LED_start=int(input('Integer correct led start\n'))
        new_start_LED_end=int(input('Integer correct led start\n'))
        
        peri_led_pad=5
        plt.close(f)
        
        
        movie_start_frame=new_start_LED_end+peri_led_pad
        

        led_on_frames_end=led_on_frames[led_on_frames>movie_midpoint]
        
        
        led_on_frames_end_first=led_on_frames_end[0]
        led_on_frames_end_last=led_on_frames_end[-1]
        pad_frames=5
        
        prepad=np.arange(led_on_frames_end_first-pad_frames,led_on_frames_end_first)
        postpad=np.arange(led_on_frames_end_last+1,led_on_frames_end_last+pad_frames+1)
        
        
        extended_LED_frames = np.concatenate((led_on_frames_end,prepad,postpad))
        extended_LED_frames.sort(kind='mergesort')
        
        
        f,axs=plt.subplots(1)
        axs.plot(extended_LED_frames,m_mean[extended_LED_frames],'k')
        axs.plot(prepad,m_mean[prepad],'r')
        axs.plot(postpad,m_mean[postpad],'y')
        axs.plot(led_on_frames_end_first,m_mean[led_on_frames_end_first],'mo')
        axs.plot(led_on_frames_end_last,m_mean[led_on_frames_end_last],'mo')
        axs.set_xticks(extended_LED_frames,)
        axs.set_xticklabels(axs.get_xticks(), rotation = 45)
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(50,100,2000, 1000)
        plt.show(block = False)
        plt.pause(0.001)
        
        
        
        new_finish_LED_start=int(input('Integer correct led start\n'))
        new_finish_LED_end=int(input('Integer correct led start\n'))
        plt.close(f)
        peri_led_pad=5
        
        movie_end_frame=new_finish_LED_start-peri_led_pad
        
        movie_range=np.arange(movie_start_frame,movie_end_frame)
        f,axs=plt.subplots(1)
        axs.plot(m_mean,'k')
        axs.plot(movie_range,m_mean[movie_range],'r')
        axs.set_xticklabels(axs.get_xticks(), rotation = 45)
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(50,100,2000, 1000)
        plt.show(block = False)
        plt.pause(0.001)
    
    
    if no_led_start:
        movie_start_frame=0
    if no_led_end:
        movie_end_frame=len(m_mean)
        
    return movie_start_frame, movie_end_frame

if __name__ == "__main__":
    image_sequence=movie
    movie_start_frame, movie_end_frame=manually_detect_led_synchs(image_sequence)
    filetosave=r'C:\Users\sp3660\Desktop\TemporaryProcessing\LED_Start_End.txt'
    with open(filetosave, 'w') as f:
        f.writelines((str( movie_start_frame),'\n', str(movie_end_frame)))
