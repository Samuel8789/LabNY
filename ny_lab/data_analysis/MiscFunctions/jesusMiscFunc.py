# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 09:56:28 2022

@author: sp3660
"""

import matplotlib.pyplot as plt

# import h5py
# import scipy.io
import mat73
import numpy as np
from scipy.spatial.distance import squareform, pdist
from scipy.stats import kendalltau, pearsonr, spearmanr, ttest_ind
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.io import loadmat, savemat
import networkx as nx 
from pprint import pprint
import time
import copy

def contrast_index( groups, similarity, indices,exclude=False):
    '''
    % Contrast index
    % Get the Contrast index for g groups given a similarity matrix.
    % (Michelson contrast 1927, Plenz 2004)
    %
    %       contrast_index = Contrast_Index(groups,similarity,indices,exclude)
    %
    %       default: exclude = [];
    %
    % Inputs
    % g = number of groups
    % sim = similarity as matrix PxP (P = #peaks)
    % idx = indexes of group to which each data point belongs
    % 
    % Outputs
    % CstIdx = Contrast index

    '''
    '''JESUS HAS added some more analysis her.
    FISRT substart Standad error of the mena form the average sna dget the max values, if
    if it max values is the max number of clusters 10 then try without sem extraction
    if that gives the same number of clusters again try the max first derivative
    '''

       
    # # % Remove diagonal values from similarity matrix
    sim = similarity-np.diag(np.diag(similarity))
    avg_in=np.zeros((groups))
    avg_out=np.zeros((groups))
    for i in range(1,groups+1):
        id_in = np.where(indices==i)[0]
        id_out = np.where(indices!=i)[0]
    
        # % Similarity average inside group
        avg_in[i-1] = np.sum(np.sum( sim[id_in[:, np.newaxis], id_in]))/id_in.size**2

        # % Similarity average outside group
        avg_out[i-1] = np.sum(np.sum(sim[id_in[:, np.newaxis], id_out]))/(id_in.size*id_out.size)
    
    # # % Identify the group to exclude
    if exclude and groups>2:
        group_excluded = np.argmin(avg_in/avg_out)+1
        ids =np.setdiff1d(np.arange(1,groups+1), group_excluded)
        #reset avrerages
        avg_in=np.zeros((groups))
        avg_out=np.zeros((groups))
        
        for i in ids:
            # % Get indices from inside group and outside group
            id_in = np.where(indices==i)[0]
            id_out = np.where(indices!=i)[0]
            
            # % Similarity average inside group
            avg_in[i-1] = np.sum(np.sum( sim[id_in[:, np.newaxis], id_in]))/id_in.size**2
    
            # % Similarity average outside group
            avg_out[i-1] = np.sum(np.sum(sim[id_in[:, np.newaxis], id_out]))/(id_in.size*id_out.size)
    
    # # % Sum the similarities
    S_in = np.sum(avg_in)
    S_out = np.sum(avg_out)
    
    # # % Compute the contrast index
    contrast_index = (S_in-S_out)/(S_in+S_out)
    return contrast_index

def get_peaks_similarity(vectors, method):
    '''
    % Similarity Index
    %
    % Compute similarity between peak vectors 
    %
    %       [sim,distance] = get_peaks_similarity(vectors,method)
    %
    '''
    
    if vectors.shape[0]>1:
        # % Compute similarity
        distance = squareform(pdist(vectors, method))
        if np.max(distance[:])==0:
            sim = np.ones((len(distance),(len(distance))))
        else:
            if method in ['cosine','correlation','spearman','hamming','jaccard']:
                sim = 1-distance
            else:
                sim = 1-distance/np.max(distance[:])# Normalization
    else:
        sim=[]
        distance=[]
    return sim, distance

def change_raster_bin_size (raster,window, binary=True):
    c,n= raster.shape
    if window:
        new_raster=raster
    else:
        new_n = np.floor(n/window).astype('int')
        new_raster = np.zeros((c,new_n));        
        for i in range(new_n):
            ini = i*window
            fin = i*window+window
            new_raster[:,i] = np.sum(raster[:,ini:fin],1)
        if binary:
            new_raster = new_raster>0
    return new_raster


def get_peak_indices(data, threshold,detect_peaks=True):

    if detect_peaks:
        indices = np.where(data> threshold)[0]
    else:
    # % detect valleys
        indices = np.where(data< threshold)[0]
    count = indices.size
    return indices, count
 
    
def filter_raster_by_network(raster, network):
    raster_filtered =copy.copy(raster)
    n_frames = raster_filtered.shape[1]
    # % Evaluate for each frame
    for frame in range(n_frames):
        # % Find active neurons on single frame
        active = np.where(raster[:,frame]==1)[0]
        if active.any():
    #         % Identify active neurons without significant coactivation
            no_significant =  np.where(np.sum(network[active[:, np.newaxis], active], axis=0)==0)[0]
            if no_significant.shape!=0:
    #             % Delete no significant neuronal coactivity from frame
                 raster_filtered[active[no_significant],frame] = 0
    # # % Get fraction of removed spikes
    n_initial_spikes = np.sum(raster[:])
    n_final_spikes = np.sum(raster_filtered[:])
    removed = n_initial_spikes-n_final_spikes
    fraction_removed = removed/n_initial_spikes
    pprint('      {removed}({fraction_removed:.2f} %) spikes removed!'.format(removed=removed, fraction_removed=100*fraction_removed))
    return raster_filtered,fraction_removed
     
def get_significant_network_from_raster(raster,window,
                                        iterations,alpha,
                                        networkMethod,shuffleMethod,
                                        singleTh):
    
    start=time.time()
    raster_binned=change_raster_bin_size(raster, window)
    A_raw=get_adjacency_from_raster(raster_binned, networkMethod)
    nNeurons = len(A_raw)
    As = np.zeros((iterations,int((nNeurons**2-nNeurons)/2)))
    pprint('   Shuffling data...')
    for i in range(iterations):
        shuffled,_ = shuffle(raster, shuffleMethod)
        shuffled_bin = change_raster_bin_size(shuffled, window)
        As[i,:] = squareform(get_adjacency_from_raster(shuffled_bin, networkMethod),force='tovector')
        if not i%100:
            t = time.time()-start
            pprint( '   {i}/{iterations} iterations, {time:.1f} s'.format(i=i, iterations=iterations, time=t))
    if not singleTh:
        n_edges = As.shape[1]
        th = np.zeros(n_edges)
        for i in range( n_edges):
            th[i] = find_threshold_in_cumulative_distribution(As[:,i],alpha)
        th = squareform(th)
    else:
        th = find_threshold_in_cumulative_distribution(As[:],alpha)
    A_significant = A_raw>th    
    return A_significant, A_raw, th, As
    
def get_adjacency_from_raster(raster,connectivity_method='coactivity'):   
    
    cells=raster.shape[0]
    if connectivity_method=='coactivity':
        raster=raster.astype('float')
        # this is a measure of how coactive are 2 cells, max caoactivity values is equela to number of frames, 
        #in a binary matrix will mean products are all 1. The diagonal will indicta ethe number of frames a given cell is active(in binary matrix)
        adjacency=raster@raster.T*( 1-np.eye(cells))  
        
    elif connectivity_method=='jaccard':
        raster=raster.astype('float')
        adjacency=squareform(1-pdist(raster,'jaccard'))
        adjacency[np.isnan(adjacency)]=0 
        
    elif connectivity_method=='pearson':        
        adjacency=np.corrcoef(raster)
        # adjacency2 = np.zeros(shape=(raster.T.shape[1], raster.T.shape[1])) scipy version uch slower
        # for i in range(raster.T.shape[1]):
        #     for j in range(raster.T.shape[1]):
        #         r, _ = pearsonr(raster.T[:,i], raster.T[:,j])
        #         adjacency2[i,j] = r
        adjacency[adjacency<0]=0
        adjacency[np.isnan(adjacency)]=0
        adjacency=adjacency*(1-np.eye(cells))             
    elif connectivity_method=='spearman':
        adjacency, _ =spearmanr(raster.T)
        adjacency[adjacency<0]=0
        adjacency[np.isnan(adjacency)]=0
        adjacency=adjacency*(1-np.eye(cells))             
    elif connectivity_method=='kendall':
        adjacency = np.zeros(shape=(raster.T.shape[1], raster.T.shape[1])) 
        for i in range(raster.T.shape[1]):
            for j in range(raster.T.shape[1]):
                r, _ = kendalltau(raster.T[:,i], raster.T[:,j])
                adjacency[i,j] = r
        adjacency[adjacency<0]=0
        adjacency[np.isnan(adjacency)]=0
        adjacency=adjacency*(1-np.eye(cells))  
        
    return adjacency

def shuffle(raster, method='frames'):
    '''
    %shuffle Shuffles raster data using various different methods
    % Shuffles spike data (0 or 1) using three differnt methods
    % assumes rows are individual cells and columns are time frames
    %
    % 'frames' - shuffles the frames in time, maintains activity pattern of
    % each frame
    %
    % 'time' - shuffles the activity of each individual cell in time
    %           each cell maintains its total level of activity
    %
    % 'time_shift' - shifts the time trace of each cell by a random amount
    %           each cell maintains its pattern of activity
    %
    % Methods fom synfire chains paper
    %
    % 'isi' - Inter-Spike Interval shuffling within cells
    %           each cell maintains in level of activity
    %
    % 'cell' - shuffles the activity at a given time between cells
    %           each frame maintains the number of active cells
    %
    % 'exchange' - exchange pairs of spikes across cells
    %           slow, but each cell and frame maintains level of activity
    %
    % jzaremba 01/2012
    %
    % modified Perez-Ortega Jesus - Aug 2018
    % modified - Feb 2019
    '''
    if method not in ['frames','time','time_shift','isi','cell','exchange']:
        method = 'frames';
    
    rand_id =[]
    shuffled=copy.copy(raster)
    
    if method== 'frames':
        n = raster.shape[1]
        randp = np.random.permutation(n);
        # shuffled = sortrows([randp;x].T).T
    #     shuffled = shuffled{2:,:)
        
    # elif  method=='time':
    #     n = size(x,2)
    #     for i = 1:size(x,1)
    #         randp = randperm(n)
    #         temp = sortrows([randp; x(i,:)]')'
    #         shuffled(i,:) = temp(2,:)
        
    elif  method== 'time_shift':
        n = raster.shape[1]
        for i in range( raster.shape[0]):
            randp = np.random.randint(0,n)
            shuffled[i,:] = np.concatenate((raster[i,n-randp:] ,raster[i,0:n-randp] ))
            rand_id.append(randp)
       
    # elif  method== 'isi':
    #     n = size(x,2)
    #     shuffled = zeros(size(x,1),n)
        
    #     for i = 1:size(x,1)
    #         # % Pull out indices of spikes, get ISIs, buffer at start and end
    #         isi = diff(find([1 x(i,:) 1]));
    #         isi = isi(randperm(length(isi))); # Randomize ISIs
            
    #         temp = cumsum(isi);
    #         temp = temp(1:end-1); # Removes the end spikes that were added
    #         # % Put the spikes back
    #         shuffled(i,temp) = true
        
        
    # elif  method== 'cell':
    #     [n,len] = size(x)
    #     for i = 1:len
    #         randp = randperm(n)
    #         temp = sortrows([randp' x(:,i)])
    #         shuffled(:,i) = temp(:,2)
        
    # elif  method== 'exchange':
    #     n = sum(x(:));
    #     for i = 1:2*n
    #         randp = randi(n,1,2);
    #         [r,c] = find(shuffled)
            
    #         # % Make sure that the swap will actually do something
    #         while randp(1)==randp(2) || r(randp(1))==r(randp(2)) || c(randp(1))==c(randp(2)) ||...
    #                 shuffled(r(randp(2)),c(randp(1)))==true ||...
    #                 shuffled(r(randp(1)),c(randp(2)))==true 
    #             randp = randi(n,1,2);
            
    #         # % Swap
    #         shuffled(r(randp(2)),c(randp(1))) = true;
    #         shuffled(r(randp(1)),c(randp(1))) = false;
            
    #         shuffled(r(randp(1)),c(randp(2))) = true;
    #         shuffled(r(randp(2)),c(randp(2))) = false;
            
    return shuffled, rand_id
              
def find_threshold_in_cumulative_distribution( data, alpha):

    if(np.min(data)==np.max(data)):
        th = np.max(data)+1
    else:
        if(np.max(data)<=1):
            x = np.arange(0,1+0.001,0.001)
        elif(np.max(data)<=10):
            x = np.arange(0,10+0.01,0.01)
        else:
            x = np.arange(0,np.max(data)+1)
            
        y,_ = np.histogram(data,x)
        cdy = np.cumsum(y)
        cdy2 = cdy/np.max(cdy)
        idd = (cdy2>(1-alpha)).argmax(axis=0)
        th = x[idd]
    return th

def seriation(Z,N,cur_index):
    '''
        input:
            - Z is a hierarchical tree (dendrogram)
            - N is the number of points given to the clustering process
            - cur_index is the position in the tree for the recursive traversal
        output:
            - order implied by the hierarchical tree Z
            
        seriation computes the order implied by a hierarchical tree (dendrogram)
    '''
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index-N,0])
        right = int(Z[cur_index-N,1])
        return (seriation(Z,N,left) + seriation(Z,N,right))   
    
def find_peaks(data, threshold=0, join=True, detect_peaks=True, minimum_width=0, fixed_width=0, ignore_ini_fin=False):
    # if data.shape[0]==1:
    #     data = data.T
    original_data = copy.copy(data)
    idx,count = get_peak_indices(data,threshold, detect_peaks=detect_peaks)
    F = data.size
    indices=np.zeros((F,1))
    if not count:
        if detect_peaks:
            pprint('No peaks found!')
            widths = []
            amplitudes = []
            ini_fin_times = []
        else:
            pprint('No valleys found!')
            widths = []
            amplitudes = []
            ini_fin_times = []
            
    if ignore_ini_fin:
        # % Delete if start above threshold
        last=0
        # idx=idx.T
        for i in idx:
            if last==i:
                if detect_peaks:
                    data[i]=threshold-1
                else:
                    data[i]=threshold+1
                last=last+1
            else:
                break
      
        # % Delete if ends above threshold
        last = F
        idx = np.flip(idx, axis=0)
        for i in idx:
            if last==i:
                if detect_peaks:
                    data[i]=threshold-1;
                else:
                    data[i]=threshold+1;
                last=last-1
            else:
                break
      
        # % Get peak or valley indices 
        idx,count = get_peak_indices(data, threshold, detect_peaks=detect_peaks)
        if not count:
            if detect_peaks:
                pprint('No peaks found!')
            else:
                pprint('No valleys found!')
    
    #this eliminate speaks with equal or les than the minimum
    if minimum_width:

        # % Join peaks or valleys
        iss = np.where(idx!=np.concatenate((np.array([0]), idx[:idx.size-1]+1), axis=0)) [0]   #% index of same peak
        # % number of total peaks or valleys
        count = iss.size                                       
        if count:
            for j in range(count-1):
                indices[idx[iss[j]]:idx[iss[j+1]-1]+1,0]=j+1 # % set #peak
                        #start of peak idx[iss[j]]
                        #endofpeak=idx[iss[j+1]-1]
            indices[idx[iss[count-1]]:np.max(idx)+1,0]=count
    
        # % Get peaks or valleys width
        widths=[]
        for i in range(1,count+1):
            widths.append(len(np.where(indices==i)[0])   )  
        widths=np.array(widths)
        # % Evaluate peaks less than or equal to minimum width
        idx_eval=np.where(widths<=minimum_width)[0]
        widths=widths[idx_eval]
    
        # % number of peaks to eliminate
        count_less=len(widths)
    
        # % Detect initial and final times
        if count_less>0:
            for i in range(count_less):
                peak=np.where(indices==idx_eval[i]+1)[0]
                ini_peak=peak[0]
                end_peak=peak[-1]
                if detect_peaks:
                    data[ini_peak:end_peak+1]=threshold-1
                else:
                    data[ini_peak:end_peak+1]=threshold+1
    
        # % Get peak or valley indices 
        idx,count = get_peak_indices(data, threshold, detect_peaks=detect_peaks)
        if not count:
            if detect_peaks:
                pprint('No peaks found!')
            else:
                pprint('No valleys found!')
                
    # % 4. Set fixed width 
    if fixed_width:
        last_end = -1
        end_before = False
        for i in idx:
            if i==last_end+1:
                if detect_peaks:
                    data[i] = threshold-1
                else:
                    data[i] = threshold+1
                if end_before:
                    end_before = False
                else:
                    last_end = i
            else:
                if i>last_end:
                    if fixed_width<0:
                        ini = i+fixed_width
                        fin = i-1
                        if ini<1:
                            ini = 1
                        fixed_width_peak = list(range(ini,fin+1))
                        last_end = fin+1
                    else:
                        fin = i+fixed_width
                        if fin>F:
                            fin = F
                        fixed_width_peak = list(range(i,fin))
                        last_end = fin-1
    
                    if detect_peaks:
                        data[fixed_width_peak] = threshold+1
                        if fixed_width<0:
                            fixed_width_peak = [i+len(fixed_width_peak) for i in fixed_width_peak]
                            data[fixed_width_peak] = threshold-1
                        elif np.sum(data[fixed_width_peak]<threshold):
                            end_before = True
                    else:
                        data[fixed_width_peak] = threshold-1
                        if fixed_width<0:
                            data[fixed_width_peak-fixed_width] = threshold+1 # not reviewed
                        elif np.sum(data[fixed_width_peak]>threshold):
                            end_before = True
  
        # % Get peak or valley indices 
        idx,count = get_peak_indices(data,threshold, detect_peaks=detect_peaks)
        
        
    # % 5. Put numbers to peaks
    indices=np.zeros((F,1))  
    for i in range(count):
        indices[idx[i]]=i+1

    # % 6. Join peaks or valleys
    if join:
        iss = np.where(idx!=np.concatenate((np.array([0]), idx[:idx.size-1]+1), axis=0)) [0]   #% index of same peak
        # % number of total peaks or valleys
        count = iss.size                                       
        if count:
            for j in range(count-1):
                indices[idx[iss[j]]:idx[iss[j+1]-1]+1,0]=j+1 # % set #peak
                        #start of peak idx[iss[j]]
                        #endofpeak=idx[iss[j+1]-1]
            indices[idx[iss[count-1]]:np.max(idx)+1,0]=count
           
    # % Get peaks or valleys width
    widths = np.zeros((count,1))
    ini_fin_times = np.zeros((count,2));
    for i in range(count):
        ids = np.where(indices==i+1)[0]
        ini_fin_times[i,0] = ids[0]
        ini_fin_times[i,1] = ids[-1]
        widths[i] = len(ids);

    # % Get peaks or vallesys amplitud
    amplitudes = np.zeros((count,1))
    for i in range(count):
        if detect_peaks:
            value = np.max(original_data[np.squeeze(indices.T)==i+1])
        else:
            value = np.min(original_data[np.squeeze(indices.T)==i+1])
        
        amplitudes[i] = value
    
    
    return indices, widths, amplitudes, ini_fin_times  

def get_evoked_neurons(activity,stim1,stim2=False):
    '''
    % Identify the neurons that are evoked by stimulation versus
    % without stimulation or a second stimulation
    %
    %       [tuned1,cp,weights,p,tuned2] = Get_Evoked_Neurons(activity,stim1,stim2)
    %
    %       stim2 can be omited.
    '''
    

    
    # % Get number of neurons
    n_neurons = activity.shape[0]
    
    
    # % Get indices from each stimulus
    stimID,_,_,_ = find_peaks(stim1, 0.1 ,True,True);
    
    if np.max(stimID)<2:
        tuned1 =np.zeros((1,n_neurons))
        cp = np.zeros((1,n_neurons))
        weights = np.zeros((1,n_neurons))
        p = np.ones((1,n_neurons))
        tuned2 = np.zeros((1,n_neurons))
        
    threshold=0.1
    if stim2:
        noStimID,_,_,_ = find_peaks(stim2,threshold);
    else:
        noStimID,_,_,_ = find_peaks(stim1, threshold, detect_peaks=False, ignore_ini_fin=True)
        
    
    # % Get the average of evoked spikes
    spikesStim = get_peak_vectors(activity,stimID,'average').T
    avgStim = np.mean(spikesStim,axis=1).T
    
    # % Get the average of spontaneous spikes
    spikesNoStim = get_peak_vectors(activity,noStimID,'average').T
    avgNoStim = np.mean(spikesNoStim,axis=1).T
    
    # % Get weigths
    weights =np.array([avgStim, avgNoStim]).T
    
    # % Compute selectivity
    tuned=np.zeros((n_neurons))
    p=np.zeros((n_neurons))
    cp=np.ones((n_neurons))
    
    for i in range(n_neurons):
        total_spikes = np.sum(spikesStim[i,:])+np.sum(spikesNoStim[i,:])
        if total_spikes:
            tuned[i],p[i] = ttest_ind(spikesStim[i,:],spikesNoStim[i,:])
            if p[i]>0.05:
                tuned[i]=False
            else:
                tuned[i]=True
            
            cp[i] = (np.sum(spikesStim[i,:])-np.sum(spikesNoStim[i,:]))/total_spikes
        else:
            p[i] = 1;
            tuned[i] = False;
            cp[i] = 0;
    tuned1 = copy.copy(tuned)
    tuned2 =copy.copy(tuned)
    
    # % Get significant to stim 1
    tuned1[np.squeeze(np.diff(weights.T, axis=0)>0)] = 0
    tuned1 = tuned1.astype(bool)
    
    # % Get significant to stim 2
    tuned2[np.squeeze(np.diff(weights.T, axis=0)<0)] = 0
    tuned2 = tuned2.astype(bool)
    
    return  tuned1,cp,weights,p,tuned2


def get_peak_vectors( data, peak_indices , vector_method, connectivity_method=None, bin_network=1):
    """
    % Get Peak Vectors
    % Join the vectors of the same peak.
    %
    %           vectors = get_peak_vectors(data,peak_indices,vector_method,connectivity_method,bin_network)
    %
    % Inputs
    % data                  = data as C x F matrix (C = #cells, F = #frames)
    % peak_indices              = Fx1 vector containing the peaks indexes
    % vector_method         = choose the method for build the vetor ('sum','average','binary','network')
    % connectivity_method   = connectivity method is used in case of
    %                         'Vector_method' is 'network' ('coactivity','jaccard','pearson','kendall','spearman')
    % bin_network           = bin is used in case of 'Vector_method' is 'network'
    % 
    % Outputs
    % DataPeaks = data as matrix PxC (P = #peaks)
    %
    %           Default:    connectivity_method = 'none'; bin_network = 1;
    %

    """
    peaks = np.max(peak_indices).astype('uint64')
    if peaks:
        C = data.shape[0]
        if  vector_method=='sum':
            vectors = np.zeros((peaks,C))
            for i in range(1,int(peaks+1)):
                data_peak_i = data[:,np.squeeze(peak_indices)==i]
                vectors[i-1,:] = np.sum(data_peak_i, axis=1)
        if  vector_method== 'binary':
            vectors = np.zeros((peaks,C))
            for i in range(1,int(peaks+1)):
                data_peak_i = data[:,np.squeeze(peak_indices)==i]
                vectors[i-1,:] = np.sum(data_peak_i, axis=1)>0
        if  vector_method== 'average':
            vectors = np.zeros((peaks,C))
            for i in range(1,int(peaks+1)):
                data_peak_i = data[:,np.squeeze(peak_indices)==i]
                vectors[i-1,:] = np.mean(data_peak_i, axis=1)
        if  vector_method== 'network':
            vectors = np.zeros((peaks,int(C*(C-1)/2)))
            for i in range(1,int(peaks+1)):
                data_peak_i = data[:,np.squeeze(peak_indices)==i]
                A = get_adjacency_from_raster(change_raster_bin_size(data_peak_i,bin_network))# here add option for connectivity method in adjacendy matrix and for windows, generalize more rthes functions
                vectors[i-1,:] = squareform(A,force='tovector')
    else:
        vectors = [];
    return vectors
