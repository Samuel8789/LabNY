# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 10:37:21 2021

@author: sp3660
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pickle
import glob

# import h5py
# import scipy.io
import mat73
from scipy.spatial.distance import squareform, pdist
from scipy.stats import distributions
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.cluster.hierarchy import fcluster
from scipy.io import loadmat, savemat
import networkx as nx 
import random
from scipy.stats import norm

from pprint import pprint
import time
import copy
from scipy.spatial.distance import cdist
try:
    from .MiscFunctions.jesusMiscFunc import  shuffle, find_threshold_in_cumulative_distribution, seriation, get_peaks_similarity, contrast_index,get_evoked_neurons,get_peak_vectors
    from .MiscFunctions.jesusMiscFunc import get_significant_network_from_raster, filter_raster_by_network, find_peaks, change_raster_bin_size, get_peak_indices, get_adjacency_from_raster
except:
    from MiscFunctions.jesusMiscFunc import  shuffle, find_threshold_in_cumulative_distribution, seriation, get_peaks_similarity, contrast_index,get_evoked_neurons,get_peak_vectors
    from MiscFunctions.jesusMiscFunc import get_significant_network_from_raster, filter_raster_by_network, find_peaks, change_raster_bin_size, get_peak_indices, get_adjacency_from_raster
       
    
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["k", "r", "b",'g','y','c','m', 'tab:brown']) 

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams["figure.figsize"] = [16, 9]

class JesusEnsemblesResults():
    
    def __init__(self, analysis_object=None, analysis_path=False, input_raster=np.empty([0]), plotting=None):
        
        self.analysis_object=analysis_object
        self.analysis_path=analysis_path
        self.input_raster=input_raster
       
        
            
        if not analysis_path:  
            if self.analysis_object:
                if  self.analysis_object.jesus_binary_spikes.any():
                    self.raster=self.analysis_object.jesus_binary_spikes
                # this is the testing rasetr
                # self.raster=loadmat('toyraster.mat')['raster']
            elif self.input_raster.any():
                self.raster=self.input_raster
                
            # self.export_raster_to_mat()
            np.random.seed(0)
            t_initial=time.time()
            self.set_default_options()
            #%% get similarity between cells and threshold it based on shuffled activity
            network, adjacency,thres,adjacency_squareform=get_significant_network_from_raster(self.raster,  self.options['Network']['Bin'],
                                                    self.options['Network']['Iterations'],self.options['Network']['Alpha'],
                                                    self.options['Network']['NetworkMethod'],self.options['Network']['ShuffleMethod'],
                                                    self.options['Network']['SingleThreshold'])
            #%%remove spikes of frames that are not significnatly connecteed to anything
            raster_filtered,spikes_fraction_removed=filter_raster_by_network(self.raster, network)
            #%% get high activity vecrtos
            vector_id, _, _, _=find_peaks(np.sum(raster_filtered, axis=0),   self.coactive_neurons_threshold, False)
            if not vector_id.any():
                analysis = []
    
            # % Get neural vectors
            pprint('   Getting vectors...')
            self.raster_vectors = get_peak_vectors(self.raster, vector_id, 'binary')
            #%% Get similarity betwen high active vectors
            pprint('   Getting similarity...')
            similarity,_ = get_peaks_similarity(self.raster_vectors, self.options['Clustering']['SimilarityMeasure'])
            
        #%% linkage and clustering based on vector similarity matrix  
            
            tree = linkage(squareform(1-similarity,force='tovector'),self.options['Clustering']['LinkageMethod'])
            # N=self.raster_vectors.shape[0]
            # res_ord = self.seriation(tree, N, N+N-2)
            # n_ensembles=10
    
            # try:
            #    fig = plt.figure(figsize=(10, 6))
            #    dn = dendrogram(tree)
            #    plt.show()
            # except:
            #     pass
            
            #%% Get recommended number of ensembles by contrat index
            pprint(['   Finding optimum number of clusters (based on ' +self.options['Clustering']['EvaluationIndex']+ ' index)...'])
            n_ensembles, clustering_indices = self.cluster_test(tree,similarity, self.raster_vectors, self.options['Clustering']['EvaluationIndex'], clusteringMethod='hierarchical', groups=self.options['Clustering']['Range'])
            
            fig, ax=plt.subplots(1)
            ax.plot(self.options['Clustering']['Range'], clustering_indices)
            plt.show()
            
            #%% Get hierarchical clustering with recommended clusters
            pprint(['   Extracting '+ str(n_ensembles) +' ensembles...'])
            # sequence =AgglomerativeClustering(n_ensembles, linkage=self.options['Clustering']['LinkageMethod']).fit(1-similarity)
            sequence=fcluster(tree, n_ensembles,criterion='maxclust' )
            
            #%% Get neurons belonging to ensemble
            pprint('   Identifying significant neurons during ensemble activation...')
            structure_pre,structure_belongingness,structure_p,ensemble_activity_binary,ensemble_vectors,ensemble_indices = self.get_ensemble_neurons(self.raster, vector_id, sequence)
            
            #%% Get functional netwoek of cells belonging to each ensemble to do further filtering
            # this sorted neurons in ensmble base on significance of previous step
            pprint('   Getting ensemble networks...')
            ensemble_neurons=[]
            ensemble_networks=[]
            all_ensemble_networks = np.zeros(network.shape)
            for i in range(n_ensembles):
                # % Get the ensemble neurons
                neurons_i = np.where(structure_pre[i,:])[0]
                idd = np.argsort(structure_belongingness[i,neurons_i])[::-1]
                ensemble_neurons.append( neurons_i[idd])
                
                # % Get the connections from the ensemble neurons
                network_i = np.zeros(network.shape);
                network_i[neurons_i[:, np.newaxis], neurons_i] = network[neurons_i[:, np.newaxis], neurons_i]
                
                ensemble_networks.append(network_i)
                all_ensemble_networks = np.logical_or(all_ensemble_networks,  ensemble_networks[i])
            
            #%% Get final ensemble neurons (this step removes neurons that were significantly active during 
            # % ensemble activation but were not significantly connected)
            pprint('   Setting functional connected neurons to ensembles...')
            structure=np.zeros((n_ensembles,network.shape[0] ))
            for i in range(n_ensembles):
                structure[i,:] = np.sum(ensemble_networks[i], axis=0)>0;    
            
            #%% This gtes similarity betwen ensemble vectors to define significant ensembles
            pprint('   Getting ensemble activity...')
            neurons, frames = self.raster.shape; 
            n_ensembles = len(ensemble_networks);
            ensemble_activity =np.zeros((n_ensembles,frames))
            structure_weights=np.zeros((n_ensembles,network.shape[0]));
            for i in range(n_ensembles):
                # % Get activity weights
                idd = ensemble_indices[i];
                jaccard = 1-cdist(np.expand_dims(np.double(structure[i,:]>0), axis=0), np.double(ensemble_vectors[i].T),metric='jaccard');
                ensemble_activity[i,idd]=jaccard
                
                # % Get structure weights
                structure_weights[i,:] = np.expand_dims(np.mean(ensemble_vectors[i],axis=1), axis=0);
            structure_weights_significant = structure_weights*(structure>0);
            
            #%% Number of ensemble activation and duration
            pprint('   Getting ensemble durations...')
            widths ,peaks_count = self.get_ensembles_length(vector_id, sequence);
            
            #%% Evaluate similarity within ensemble vectors
            pprint('   Identifying significant ensembles...')
            # % Get similarity within rasters
            within,vector_count= self.similarity_within_rasters(ensemble_vectors);
            ensemble_p = self.test_ensemble_similarity(similarity,within,peaks_count, self.options['Ensemble']['Iterations']);
            h_ensemble = np.array(ensemble_p)< self.options['Ensemble']['Alpha']
            
            # % Get IDs of ensembles and non-ensembles
            id_ensemble = np.where(h_ensemble)[0]
            id_nonensemble = np.where(h_ensemble==False)[0];
            
            #%%Separate significant and non significant ensembles
            n_ensembles = len(id_ensemble);
            n_nonensembles = len(id_nonensemble);
            if n_nonensembles:
                # % Get new sequence (only significant ensembles)
                sequence_new = np.zeros(sequence.shape);
                for i in range(n_ensembles):
                    sequence_new[sequence==id_ensemble[i]+1] = i+1;
                sequence = sequence_new;
                
                sequence_non_ensembles = np.zeros(sequence.shape);
                for i in range(n_nonensembles):
                    sequence_non_ensembles[sequence==id_nonensemble[i]+1] = i+1;
                
                
                # % Non ensembles properties
                nonensemble_activity = ensemble_activity[id_nonensemble,:]
                nonensemble_activity_binary = ensemble_activity_binary[id_nonensemble,:]
                nonensemble_networks = [ensemble_networks[i] for i in id_nonensemble]
                nonensemble_vectors = [ensemble_vectors[i] for i in id_nonensemble]
                nonensemble_indices = [ensemble_indices[i] for i in id_nonensemble]
                nonensemble_within = [within[i] for i in id_nonensemble]
                nonensemble_vector_count = [vector_count[i] for i in id_nonensemble]
                nonensemble_structure = structure[id_nonensemble,:]
                nonensemble_structure_belongingness = structure_belongingness[id_nonensemble,:]
                nonensemble_structure_p = structure_p[id_nonensemble,:]
                nonensemble_structure_weights = structure_weights[id_nonensemble,:]
                nonensemble_structure_weights_significant = structure_weights_significant[id_nonensemble,:]
                nonensemble_neurons = [ensemble_neurons[i] for i in id_nonensemble]
                nonensemble_widths = [widths[i] for i in id_nonensemble]
                nonensemble_peaks_count = [peaks_count[i] for i in id_nonensemble]
                nonensemble_p = [ensemble_p[i] for i in id_nonensemble]
                
                #% Ensembles 
    
                ensemble_activity = ensemble_activity[id_ensemble,:]
                ensemble_activity_binary = ensemble_activity_binary[id_ensemble,:]
                ensemble_networks = [ensemble_networks[i] for i in id_ensemble]
                ensemble_vectors = [ensemble_vectors[i] for i in id_ensemble]
                ensemble_indices = [ensemble_indices[i] for i in id_ensemble]
                within = [within[i] for i in id_ensemble]
                vector_count = [vector_count[i] for i in id_ensemble]
                structure = structure[id_ensemble,:]
                structure_belongingness = structure_belongingness[id_ensemble,:]
                structure_p = structure_p[id_ensemble,:]
                structure_weights = structure_weights[id_ensemble,:]
                structure_weights_significant = structure_weights_significant[id_ensemble,:]
                ensemble_neurons = [ensemble_neurons[i] for i in id_ensemble]
                widths = [widths[i] for i in id_ensemble]
                peaks_count = [peaks_count[i] for i in id_ensemble]
                ensemble_p = [ensemble_p[i] for i in id_ensemble]
                
    
                pprint('      '+ str(n_ensembles)+ ' significant ensembles.')
                pprint('      ' +str(n_nonensembles)+ ' non significant ensembles.')
            
            #%% Sort ensembles by number of cells
            pprint('   Sorting ensembles from high to low participation...')
            structure_weights_sorted,neuron_id,ensemble_id_sorted,ensemble_avg_weights =self.sort_ensemble_weights(structure_weights_significant);
            ensemble_activity = ensemble_activity[ensemble_id_sorted,:]
            ensemble_activity_binary = ensemble_activity_binary[ensemble_id_sorted,:]
            ensemble_networks = [ensemble_networks[i] for i in ensemble_id_sorted]
            ensemble_vectors = [ensemble_vectors[i] for i in ensemble_id_sorted]
            ensemble_indices = [ensemble_indices[i] for i in ensemble_id_sorted]
            within = [within[i] for i in ensemble_id_sorted]
            vector_count = [vector_count[i] for i in ensemble_id_sorted]
            structure = structure[ensemble_id_sorted,:]
            structure_belongingness = structure_belongingness[ensemble_id_sorted,:]
            structure_p = structure_p[ensemble_id_sorted,:]    
            structure_weights = structure_weights[ensemble_id_sorted,:]
            structure_weights_significant = structure_weights_significant[ensemble_id_sorted,:]
            ensemble_neurons = [ensemble_neurons[i] for i in ensemble_id_sorted]
            widths = [widths[i] for i in ensemble_id_sorted]
            peaks_count = [peaks_count[i] for i in ensemble_id_sorted]
            ensemble_p = [ensemble_p[i] for i in ensemble_id_sorted]
            
            # % Get id of vectors sorted
            vectors_id = [];
            for i in range(n_ensembles):
                    vectors_id = np.concatenate((vectors_id, ensemble_indices[i]))
            for  i in range(n_nonensembles):
                vectors_id = np.concatenate((vectors_id, nonensemble_indices[i]))
    
            # %% Save analysis dictionary
            pprint('   Adding results to ''analysis'' variable output...')
            self.analysis={}
            self.analysis['Options']=self.options
            self.analysis['Raster']=self.raster
            self.analysis['Neurons']=neurons
            self.analysis['Frames']=frames
            self.analysis['Significant Network']=network
            self.analysis['Cell Adjacency Matrix']=adjacency

            
            self.analysis['Filter']={}
            self.analysis['Filter']['RasterFiltered']=raster_filtered
            self.analysis['Filter']['SpikesFractionRemoved']=spikes_fraction_removed
            self.analysis['Filter']['RasterVectors']=self.raster_vectors
            self.analysis['Filter']['VectorID']=vector_id
    
            self.analysis['Clustering']={}
            self.analysis['Clustering']['Similarity']=similarity
            self.analysis['Clustering']['Tree']=tree
            self.analysis['Clustering']['RecommendedClusters']=n_ensembles
            self.analysis['Clustering']['ClusteringIndex']=self.options['Clustering']['EvaluationIndex']
            self.analysis['Clustering']['ClusteringRange']=self.options['Clustering']['Range']
            self.analysis['Clustering']['ClusteringIndices']=clustering_indices
            # self.analysis['Clustering']['TreeID']=treeID
    

            self.analysis['Ensembles']={}
            self.analysis['Ensembles']['Count']=n_ensembles
            self.analysis['Ensembles']['ActivationSequence']=sequence
            self.analysis['Ensembles']['Activity']=ensemble_activity
            self.analysis['Ensembles']['ActivityBinary']=ensemble_activity_binary
            self.analysis['Ensembles']['Networks']=ensemble_networks
            self.analysis['Ensembles']['AllEnsembleNetwork']=all_ensemble_networks
            self.analysis['Ensembles']['Vectors']=ensemble_vectors
            self.analysis['Ensembles']['Indices']=ensemble_indices
            self.analysis['Ensembles']['Similarity']=within
            self.analysis['Ensembles']['VectorCount']=vector_count
            self.analysis['Ensembles']['Structure']=structure
            self.analysis['Ensembles']['StructureBelongingness']=structure_belongingness
            self.analysis['Ensembles']['StructureP']=structure_p
            self.analysis['Ensembles']['StructureWeights']=structure_weights
            self.analysis['Ensembles']['StructureWeightsSignificant']=structure_weights_significant
            self.analysis['Ensembles']['StructureSorted']=structure_weights_sorted
            self.analysis['Ensembles']['Weights']=ensemble_avg_weights
            self.analysis['Ensembles']['EnsembleNeurons']=ensemble_neurons
            self.analysis['Ensembles']['NeuronID']=neuron_id
            self.analysis['Ensembles']['VectorID']=vectors_id
            self.analysis['Ensembles']['Durations']=widths
            self.analysis['Ensembles']['PeaksCount']=peaks_count
            self.analysis['Ensembles']['Probability']=ensemble_p
            self.analysis['Ensembles']['AlphaEnsemble']=   self.options['Ensemble']['Alpha']
    
            self.analysis['NonEnsembles']={}
    
            if n_nonensembles:
                self.analysis['NonEnsembles']['Count']=n_nonensembles
                self.analysis['NonEnsembles']['ActivationSequence']=sequence_non_ensembles
                self.analysis['NonEnsembles']['Activity']=nonensemble_activity
                self.analysis['NonEnsembles']['ActivityBinary']=nonensemble_activity_binary
                self.analysis['NonEnsembles']['Networks']=nonensemble_networks
                self.analysis['NonEnsembles']['Vectors']=nonensemble_vectors
                self.analysis['NonEnsembles']['Indices']=nonensemble_indices
                self.analysis['NonEnsembles']['Similarity']=nonensemble_within
                self.analysis['NonEnsembles']['VectorCount']=nonensemble_vector_count
                self.analysis['NonEnsembles']['Structure']=nonensemble_structure
                self.analysis['NonEnsembles']['StructureBelongingness']=nonensemble_structure_belongingness
                self.analysis['NonEnsembles']['StructureP']=nonensemble_structure_p
                self.analysis['NonEnsembles']['StructureWeights']=nonensemble_structure_weights
                self.analysis['NonEnsembles']['StructureWeightsSignificant']=nonensemble_structure_weights_significant
                self.analysis['NonEnsembles']['EnsembleNeurons']=nonensemble_neurons
                self.analysis['NonEnsembles']['Durations']=nonensemble_widths
                self.analysis['NonEnsembles']['PeaksCount']=nonensemble_peaks_count
                self.analysis['NonEnsembles']['Probability']=nonensemble_p

            
            # % Display the total time
            t_final = time.time()-t_initial;
            pprint('You are all set! (total time: ' +str(t_final)+ ' seconds)')

            
        else:
            self.load_analysis_from_file()
            
#%% file managing

    # def save_results(self):
    #     print('Saving jesus run')
    #     datapath=os.path.join(self.jesus_runs_path, self.jesus_runs_path_name)
        
    #     if not os.path.isdir(self.jesus_runs_path):
    #         os.mkdir(self.jesus_runs_path)
            
    
    #     with open(datapath, 'wb') as f:
    #         # Pickle the 'data' dictionary using the highest protocol available.
    #         pickle.dump(self.jesus_run, f, pickle.HIGHEST_PROTOCOL)
    
    def load_analysis_from_file(self):
        self.timestr = time.strftime("%Y%m%d-%H%M%S")

        with open( self.analysis_path, 'rb') as file:
            self.results_from_file= pickle.load(file)
            
        self.analysis=self.results_from_file[-1] 
        self.input_options=self.results_from_file[:-1] 
        
    def export_raster_to_mat(self):
        savemat(r'C:\Users\sp3660\Documents\Github\LabNY\ny_lab\data_analysis\Jesus\Ensemblex\Ensemblex\test_raster.mat',{'raster':self.raster})



#%% methods
  

    def cluster_test(self, treeOrData, similarity,vectors,  metric='contrast',clusteringMethod='hierarchical',groups=np.arange(2,31),fig=None):
        '''
        # % Clustering indexes
        # % Get indexes for evaluating clustering from hierarchical cluster tree or
        # % data points to perform 
        # %
        # %       [recommended,indices] = cluster_test(treeOrData,similarity,metric,clusteringMethod,groups,fig)
        # %
        # %       default: metric = 'contrast'; clusteringMethod = 'hierarchical';
        # %                groups = 2:30; fig = []
        # %
        # % Inputs:
        # %      treeOrData = hierarchical cluster tree, or data for k-means
        # %      similarity = matrix PxP (P=#peaks) for metrics Dunn &
        # %                    Contrast; Xpeaks, peaks vectors as matrix PxC for metrics
        # %                    Connectivity & Davies
        # %                    (P = #peaks; C=#cells)
        # %      metric = index to compute ('dunn','connectivity','davies','contrast')
        # %      clusteringMethod = 'hierarchical' or 'kmeans'
        # %      groups = range of groups to analize
        # %      numFig = number of the figure to plot
        # %
        # % Outputs:
        # % indices = clustering indices of 'metric' from the range of 'groups'.
        # % recommended = recommended number of clusters
        # %
        '''

        dist = 1-similarity
        indices=np.zeros((len(groups)))
        for k, i in enumerate(groups):
            if clusteringMethod == 'hierarchical':
                # clus= AgglomerativeClustering(i, linkage=self.options['Clustering']['LinkageMethod']).fit(vectors)
                # T=clus.labels_

                T=fcluster(treeOrData, i, criterion='maxclust' )

            elif  clusteringMethod== 'kmeans':
                T = KMeans(treeOrData,i)# not reviews by my 
            g = np.max(T)
            
            # if metric==  'dunn':
            #     indices[j] = Dunn_Index(g,dist,T);
            # elif metric==  'davies':
            #     indices[j] = Davies_Index(g,similarity,T);
            # elif metric==  'contrast':
            #     indices[j] = self.contrast_index(g,similarity,T);    
            # elif metric==  'contrast_excluding_one':
            if metric==  'contrast_excluding_one':
                indices[k] = contrast_index(g, similarity, T)
        
        if metric== 'contrast_excluding_one':
            if np.count_nonzero(indices == 1):
                idd = np.where(indices == 1,1,'last')[0]
                recommended = groups[idd]
            else:
                idd = np.argmax(indices)
                recommended = groups[idd]
        else:
            '''
            TO DO CORREECT THIS, IGNIRED FOR THE MOMETN
            '''
            # % Select the best number of groups based on an index
            # % If any index is equal to 1 it is a perfect separation, so choose the
            # % maximum number of groups
            if np.count_nonzero(indices == 1):
                idd = np.where(indices == 1,1,'last')
                recommended = groups[idd]
            else:
                _, idd = np.where(np.diff(indices)>0,1,'first')
                if not idd.any() or idd==len(groups)-1:
                    # % The indices are decreasing, so select the first
                    recommended = groups(1)
                    idd = 1
                else:
                    # % Find the first peak of the indices
                    indicesCopy = indices
                    indicesCopy[1:idd] = 0
                    _,idd = np.where(np.diff(indicesCopy)<0,1,'first')[0]
                    if not idd.any():
                        # % If there is no peak find the first sudden increase
                        _,idd = np.where(np.diff(np.diff(indicesCopy))<0,1,'first')[0]
                        idd = idd+1;
                    recommended = groups[idd]
        
        # % Plot 
        # if fig:
        #     plot(groups,indices)
        #     hold on
        #     plot(recommended,indices(id),'*r')
        #     hold off
        #     title([replace(metric,'_','-') '''s index (' num2str(recommended) ' groups recommended)'])
        #     xlabel('number of groups')
        #     ylabel('index value')
        # end
        
            
        fig, ax=plt.subplots(1)
        ax.plot(self.options['Clustering']['Range'], indices)
        plt.show()
        return recommended, indices
     
    def get_ensemble_neurons(self, raster,vectorID,sequence):
        # % Get a network from each ensemble
        # %
        # %       [structure,structure_belongingness,structure_p,ensemble_activity_binary,vectors,indices] = ...
        # %   get_ensemble_neurons(raster,vectorID,sequence)
        # %
        # % By Jesus Perez-Ortega, Oct 2021
        
        # % Get number of ensembles
        ensembles = len(set(sequence))
        
        # % Get ensemble network
        vectors=[]
        indices=[]
        structure=np.zeros((ensembles,raster.shape[0]))
        structure_belongingness=np.zeros((ensembles,raster.shape[0]))
        structure_p=np.zeros((ensembles,raster.shape[0]))
        ensemble_activity_binary=np.zeros((ensembles, raster.shape[1]))
        for i in range(1,ensembles+1):
            
            
            # % Get raster ensemble
            peaks = np.where(sequence==i)[0]
            peak_indices = np.empty((0))
            for j in range(len(peaks)):
                peak_indices = np.concatenate((peak_indices, np.where(vectorID==peaks[j]+1)[0])).astype('uint64')
            vectors.append(raster[:,peak_indices])
            indices.append(peak_indices)
            
            # # % Get ensemble activity
            activity = np.zeros(raster.shape[1])
            
            activity[peak_indices] = True; 
            ensemble_activity_binary[i-1,:] = activity
            
            # # % Detect neurons significantly active with the ensemble
            h1, belongingness,_,p,_ = get_evoked_neurons(raster,activity)
            p[h1] = 1
        
            structure[i-1,:] = h1
            structure_belongingness[i-1,:] = belongingness
            structure_p[i-1,:] = p
            
        return structure,structure_belongingness,structure_p,ensemble_activity_binary,vectors,indices
     
    def get_ensembles_length(self, indices, sequence):
        '''
        % Identify the legnth of the ensembles and plot the distribution of each one
        %
        %       widths = Plot_Ensembles_Length(indices,sequence)

        '''
        
        # % Find vectors
        idd = np.where(indices>0)[0]
        seqEns=np.zeros((indices.shape))
        # % Get the ensemble id
        ensembles = list(set(sequence))
        nEns = len(ensembles);
        widths=[]
        nPeaks=[]
        
        # % Get colors
        for i in range(nEns):
            
            # % Create binary signal to identify the lenght of each activation
          
            seqEns=np.zeros((idd[sequence==ensembles[i]][-1]+1))
            seqEns[idd[sequence==ensembles[i]]] = 1;
            
            
            # % find peaks
            _, w,_,_ = find_peaks(seqEns)
            
            # % get number of peaks
            nPeaks.append(len(w))
            
            # % assign widths
            widths.append(w)
            #
        return widths, nPeaks
     
    def similarity_within_rasters(self,rastersEnsemble):
        '''
        % Get similarity within raster ensembles
        %
        %       [within,count] = similarity_within_rasters(rastersEnsemble)
        %
        % By Jesus Perez-Ortega, Apr 2020
        '''
        
        nEnsembles = len(rastersEnsemble);
        count=[]
        within=[]
        # % Similarity between A and B
        for i in range(nEnsembles):
            # % Get raster from A
            rA = rastersEnsemble[i]            
             # % Get number of peaks
            count.append(rA.shape[1])
            # % Get similarity
            within.append( np.mean(1-pdist(rA.T.astype('float'),'jaccard')))
        return within,count

    def test_ensemble_similarity(self, sim_matrix,sim_ensembles,count_ensembles,iterations=1000):
        '''
                % Test significance of similarity between vectors of ensembles
        %
        %       p = test_ensemble_similarity(sim_matrix,sim_ensembles,count_ensembles,iterations)
        %
        %       default: iterations = 1000
        %
        '''

        # % Get number of vectors
        n_vectors = len(sim_matrix);
        
        # % Get average of similarities
        np.random.seed(0)
        avg_sim=np.zeros((iterations,n_vectors-1 ))
        for i in range(iterations):
            # % Shuffle vectors
            vector_id = np.random.permutation(n_vectors)
            vector_i = vector_id[0]
            group_i = sim_matrix[vector_i,vector_id[1:]]
            
            # % Get average similarity from 2 to n_vectors
            sum_vectors = np.cumsum(group_i)
            avg_sim[i,:] =sum_vectors/(np.arange(1,n_vectors))
        
        # % Get probability for each ensemble
        
        # % Get number of ensembles
        n_ensembles = len(sim_ensembles);
        p=[]
        for i in range(n_ensembles):
            ensemble_sim = sim_ensembles[i];
            ensemble_count = count_ensembles[i];
            selected = avg_sim[:,ensemble_count];
            mu,sigma = distributions.norm.fit(selected);
            # % [~,pCov] = normlike([mu,sigma],selected); % this is for confidence intervals (pLo and pUp)
            # % [p,pLo,pUp] = normcdf(ensemble_sim,mu,sigma,pCov);
            p.append( 1-norm.cdf(ensemble_sim ,mu,sigma))
        return p
    
    def sort_ensemble_weights(self,structure,weight_threshold=0):
        '''
                % Sort ensemble structure weighted
        %
        %   [structureSorted,neuronsID,ensemblesID,avgWeights] = 
        %       sort_ensemble_weights(structure,weight_threshold)
        %
        %       default: weight_threshold = 0;

        '''

        # % Get the number of 
        nEnsembles,nNeurons= structure.shape
        
        # % Sort ensembles
        avgWeights = np.mean(structure,axis=1);
        ensemble_id=np.argsort(avgWeights)[::-1]
        
        # % Structure sorted by ensembles
        structure_sorted = structure[ensemble_id,:]
        structure_sorted[structure_sorted<weight_threshold] = 0;
        
        # % Sort neurons
        neuron_id = np.arange(0,nNeurons)
        for i in reversed(range(nEnsembles)):

            idd = np.argsort(structure_sorted[i,:], kind='mergesort')
            # structure_sorted.sort(axis=1)
            
            structure_sorted = structure_sorted[:,idd]
  

            neuron_id = neuron_id[idd] 
        
        # % Set structure sorted
        structure_sorted = np.fliplr(structure[ensemble_id[:,np.newaxis],neuron_id])
        avg_weights_sorted = avgWeights[ensemble_id]
        return structure_sorted, neuron_id, ensemble_id, avg_weights_sorted

  
        
    def set_default_options(self):
        
        self.options={}
        self.options['Network']={'Bin':1,
                            'Iterations':1000,
                            'Alpha':0.05,
                            'NetworkMethod': 'coactivity',# 'coactivity','jaccard',Pearson,Kendall,Spearman
                            'ShuffleMethod':'time_shift',
                            'SingleThreshold':False}
        self.coactive_neurons_threshold = 2;
        self.options['Clustering']={'SimilarityMeasure':'jaccard',
                    'LinkageMethod':'ward',
                    'EvaluationIndex':'contrast_excluding_one',
                    'Range': np.arange(3,11)
                    }
        self.options['Ensemble']={'Iterations':1000,
                    'Alpha':0.05
                    }
    
    #%%plotting
    def save_multi_image(self, filename):
       pp = PdfPages(filename)
       fig_nums = plt.get_fignums()
       figs = [plt.figure(n) for n in fig_nums]
       for fig in figs:
          fig.savefig(pp, format='pdf')
       pp.close()
       plt.close('all')
       
    def plot_raster(self):
        plt.close('all')

        
        pixel_per_bar = 10
        dpi = 100
        fig = plt.figure(figsize=(6+(200*pixel_per_bar/dpi), 10), dpi=dpi)
        # fig = plt.figure(figsize=(16,9), dpi=dpi)
        ax = fig.add_axes([0.05, 0.2, 0.9, 0.7])  # span the whole figure
        ax.imshow(self.analysis['Raster'], cmap='binary', aspect='auto',vmax=0.01)
        ax.set_xlabel('Frames')
        fig.supylabel('Cell Number')
        fig.suptitle(f'Raw Unsorted {"_".join(self.input_options)}')
        # for vector in np.where(self.analysis['Filter']['VectorID'])[0]:
        #     ax.axvspan(vector-1,vector+1, facecolor='g', alpha=0.1)
        
        

        # fig = plt.figure(figsize=(6+(200*pixel_per_bar/dpi), 10), dpi=dpi)
        fig = plt.figure(figsize=(16,9), dpi=dpi)
        ax = fig.add_axes([0.05, 0.2, 0.9, 0.7])  # span the whole figure
        ax.imshow(self.analysis['Filter']['RasterFiltered'], cmap='binary', aspect='auto',vmax=0.01)
        ax.set_xlabel('Frames')
        fig.supylabel('Cell Number')
        fig.suptitle(f'Filtered Unsorted {"_".join(self.input_options)}')
 
        

        # fig = plt.figure(figsize=(6+(200*pixel_per_bar/dpi), 10), dpi=dpi)
        fig = plt.figure(figsize=(16,9), dpi=dpi)
        ax = fig.add_axes([0.05, 0.2, 0.9, 0.7])  # span the whole figure
        ax.imshow(self.analysis['Filter']['RasterFiltered'], cmap='binary', aspect='auto',vmax=0.01)
        ax.set_xlabel('Frames')
        fig.supylabel('Cell Number')
        fig.suptitle(f'Filtered Unsorted Vector Overlay {"_".join(self.input_options)}')
        for vector in np.where(self.analysis['Filter']['VectorID'])[0]:
            ax.axvline(x=vector, linewidth=1, color='r', alpha=0.1)


   
        # fig = plt.figure(figsize=(6+(200*pixel_per_bar/dpi), 10), dpi=dpi)
        fig = plt.figure(figsize=(16,9), dpi=dpi)
        ax = fig.add_axes([0.05, 0.2, 0.9, 0.7])  # span the whole figure
        ax.imshow(self.analysis['Filter']['RasterVectors'].T, cmap='binary', aspect='auto',vmax=0.01)
        ax.set_xlabel('Multi Activity Vectors')
        fig.supylabel('Cell Number')
        fig.suptitle(f'RasterVectors {"_".join(self.input_options)}')
        
        
        filename = os.path.join(os.path.split(self.analysis_path)[0],f'{"_".join(self.input_options)}_{self.timestr}_rasters.pdf')
        self.save_multi_image(filename)
        
    def plot_sorted_rasters(self):
        plt.close('all')

        

        
        neuron_id=self.analysis['Ensembles']['NeuronID']
        raster=self.analysis['Raster']
        vector_id=self.analysis['Ensembles']['VectorID'].astype('uint64')
        neuronsorted=raster[neuron_id,:][::-1]
        vectorsorted=raster[:,vector_id]
        neurvectorsorted=neuronsorted[:,vector_id]

        pixel_per_bar = 10
        dpi = 100
        # fig = plt.figure(figsize=(6+(200*pixel_per_bar/dpi), 10), dpi=dpi)
        fig,ax=plt.subplots(figsize=(6+(200*pixel_per_bar/dpi), 10),dpi=dpi)
        # ax.imshow(neurvectorsorted, cmap='binary', aspect='auto',
        #     interpolation='nearest', norm=mpl.colors.Normalize(0, 1))
        ax.imshow(neurvectorsorted, cmap='binary', aspect='auto',vmax=0.05)
        ax.set_xlabel('Frames')
        fig.supylabel('Cell Number')
        fig.suptitle(f'Neuron Vectors Sorted {"_".join(self.input_options)}')
        fig.set_tight_layout(True)

     
        # fig = plt.figure(figsize=(6+(200*pixel_per_bar/dpi), 10), dpi=dpi)
        fig,ax=plt.subplots(figsize=(6+(200*pixel_per_bar/dpi), 10),dpi=dpi)
        ax.imshow(neuronsorted, cmap='binary', aspect='auto',vmax=0.05)
        ax.set_xlabel('Frames')
        fig.supylabel('Cell Number')
        fig.suptitle(f'Neuron Sorted {"_".join(self.input_options)}')
        fig.set_tight_layout(True)

        
        # fig = plt.figure(figsize=(6+(200*pixel_per_bar/dpi), 10), dpi=dpi)
        fig,ax=plt.subplots(figsize=(6+(200*pixel_per_bar/dpi), 10),dpi=dpi)
        ax.imshow(vectorsorted, cmap='binary', aspect='auto',vmax=0.05)
        ax.set_xlabel('Frames')
        fig.supylabel('Cell Number')
        fig.suptitle(f'Vectors Sorted {"_".join(self.input_options)}')
        fig.set_tight_layout(True)

        
        filename = os.path.join(os.path.split(self.analysis_path)[0],f'{"_".join(self.input_options)}_{self.timestr}_sorted_rasters.pdf')
        self.save_multi_image(filename)
        plt.close('all')

     
    def plot_networks(self):  
        plt.close('all')

        
        network=self.analysis['Significant Network']
        adjacency=self.analysis['Cell Adjacency Matrix']

        pixel_per_bar = 4
        dpi = 100
        
        # fig = plt.figure(figsize=(6+(200*pixel_per_bar/dpi), 10), dpi=dpi)
        fig,ax=plt.subplots(figsize=(16,9))
        # im=ax.imshow(network, cmap='viridis', aspect='auto', vmax=np.max(network),
        #     interpolation='nearest', norm=mpl.colors.Normalize(0, 1))
        im=ax.imshow(network, cmap='viridis', aspect='auto', vmax=np.max(network))
        fig.colorbar(im, ax=ax, orientation='vertical', fraction=.1)
        ax.set_xlabel('Cell Number')
        fig.supylabel('Cell Number')
        fig.suptitle(f'Significant Network {"_".join(self.input_options)}')
        fig.set_tight_layout(True)

        

        fig,ax=plt.subplots(figsize=(16,9))
        im=ax.imshow(adjacency,  cmap='viridis', aspect='auto',vmax=np.max(adjacency),
            interpolation='nearest', norm=mpl.colors.Normalize(0, 1))
        fig.colorbar(im, ax=ax, orientation='vertical', fraction=.1)
        ax.set_xlabel('Cell Number')
        fig.supylabel('Cell Number')
        fig.suptitle(f'Adjacency Network {"_".join(self.input_options)}')
        fig.set_tight_layout(True)

        
        filename = os.path.join(os.path.split(self.analysis_path)[0],f'{"_".join(self.input_options)}_{self.timestr}_adjacencies.pdf')
        self.save_multi_image(filename)
        
        
    def plot_vector_clustering(self):
        plt.close('all')

        similarity=self.analysis['Clustering']['Similarity']
        N=self.analysis['Filter']['RasterVectors'].shape[0]
        res_ord = seriation(self.analysis['Clustering']['Tree'], N, N+N-2)
        similarity=self.analysis['Clustering']['Similarity']
        t=similarity[res_ord,: ]
        t2=t[:, res_ord]

        
        
        
        fig,ax=plt.subplots(figsize=(16,9))
        im=ax.imshow(similarity,cmap='viridis', aspect='auto',vmax=np.max(similarity),
            interpolation='nearest', norm=mpl.colors.Normalize(0, 1))
        fig.colorbar(im, ax=ax, orientation='vertical', fraction=.1)
        ax.set_xlabel('Ensemble Vector')
        fig.supylabel('Ensemble Vector')
        fig.suptitle('Vector Similarity')
        fig.set_tight_layout(True)
        

        fig,ax=plt.subplots(figsize=(16,9))
        im=ax.imshow(t2,cmap='viridis', aspect='auto', vmax=np.max(similarity),
            interpolation='nearest', norm=mpl.colors.Normalize(0, 1))
        fig.colorbar(im, ax=ax, orientation='vertical', fraction=.1)
        ax.set_xlabel('Ensemble Vector')
        fig.supylabel('Ensemble Vector')
        fig.suptitle('Sorted Vector Similarity')
        fig.set_tight_layout(True)



        fig = plt.figure(figsize=(10, 6))
        dn = dendrogram(self.analysis['Clustering']['Tree'])
        fig.set_tight_layout(True)

        
        
        filename = os.path.join(os.path.split(self.analysis_path)[0],f'{"_".join(self.input_options)}_{self.timestr}_vector_clustering.pdf')
        self.save_multi_image(filename)
        
      
  
        
        
    def plot_all_ensembles_rasters(self):
        
        for ensemble_index  in range(len(self.analysis['Ensembles']['EnsembleNeurons'])):
            self.plot_activity_cell_ensemble(ensemble_index)
            
        
    def plot_activity_cell_ensemble(self, ensemble_index):
        
        
        # first get cell indexes
        
        cells=self.analysis['Ensembles']['EnsembleNeurons'][ensemble_index]
        # cells=self.analysis['NonEnsembles']['EnsembleNeurons'][ensemble_index]

        cells.sort()
        #second get activity
        activity=self.analysis['Raster'][cells]
        # get ensmeble presentations
        ensembleframes=np.where(self.analysis['Ensembles']['ActivationSequence']==ensemble_index+1)[0]
        # nonensembleframes=np.where(self.analysis['NonEnsembles']['ActivationSequence']==ensemble_index+1)[0]

        
        
        #plot raster
        # pixel_per_bar = 4
        # dpi = 100
        # fig = plt.figure(figsize=(16,9),dpi=dpi)
        # ax = fig.add_axes([0.05, 0.2, 0.9, 0.7])  # span the whole figure
        # ax.imshow(activity,cmap='binary', aspect='auto')
        # ax.set_xlabel('Time(s)')
        # fig.supylabel('Ensemble Cell')
        # fig.suptitle('Ensemble {} Activity'.format(ensemble_index+1) )
        
        #plot traces
        pixel_per_bar = 4
        dpi = 100
        fig = plt.figure(figsize=(16,9),dpi=dpi)
        ax = fig.add_axes([0.05, 0.2, 0.9, 0.7])  #
        
        for i,cell in enumerate(cells):
            ax.vlines(np.where(activity[i,:]), ymin=i+0, ymax=i+1)
            self.finali=i

        for j in   ensembleframes : 
            ax.axline((j, 0), (j, self.finali+1), linewidth=0.1, color='g', alpha=0.2)
 
        ax.set_xlabel('Time(s)')
        fig.supylabel('Ensemble Cell')
        fig.suptitle('Ensemble {} Activity'.format(ensemble_index+1))
        
       
        
        
        
        
        
    # def plotting_ensembles(self):
    #     self.analysis['Ensembles']['ActivationSequence']
        
        
        
        
        
    #     pixel_per_bar = 4
    #     dpi = 100
        
    #     fig = plt.figure(figsize=(16,9),dpi=dpi)
    #     ax = fig.add_axes([0.05, 0.2, 0.9, 0.7])  # span the whole figure
    #     ax.imshow(self.analysis['Ensembles']['Activity'],cmap='binary', aspect='auto')
    #     ax.set_xlabel('Time(s)')
    #     fig.supylabel('Ensemble')
    #     fig.suptitle('Ensemble Activity')
        
             
    #     fig = plt.figure(figsize=(16,9),dpi=dpi)
    #     ax = fig.add_axes([0.05, 0.2, 0.9, 0.7])  # span the whole figure
    #     ax.imshow(self.analysis['Ensembles']['ActivityBinary'],cmap='binary', aspect='auto')
    #     ax.set_xlabel('Time(s)')
    #     fig.supylabel('Ensemble')
    #     fig.suptitle('Ensemble Activity Binary')
        
        
              
    #     fig = plt.figure(figsize=(16,9),dpi=dpi)
    #     ax = fig.add_axes([0.05, 0.2, 0.9, 0.7])  # span the whole figure
    #     ax.imshow(self.analysis['Ensembles']['AllEnsembleNetwork'],cmap='binary', aspect='auto')
    #     ax.set_xlabel('Cell')
    #     fig.supylabel('Cell')
    #     fig.suptitle('Ensemble Network')
        
    #     fig = plt.figure(figsize=(16,9),dpi=dpi)
    #     ax = fig.add_axes([0.05, 0.2, 0.9, 0.7])  # span the whole figure
    #     ax.imshow(self.analysis['Ensembles']['Structure'].T,cmap='binary', aspect='auto')
    #     ax.set_xlabel('Ensemble')
    #     fig.supylabel('Cell')
    #     fig.suptitle('Ensemble Network')
        
    #     fig = plt.figure(figsize=(16,9),dpi=dpi)
    #     ax = fig.add_axes([0.05, 0.2, 0.9, 0.7])  # span the whole figure
    #     ax.imshow(self.analysis['Ensembles']['StructureSorted'].T,cmap='binary', aspect='auto')
    #     ax.set_xlabel('Ensemble')
    #     fig.supylabel('Cell')
    #     fig.suptitle('Ensemble Network')
        
      
        
    #     ensemble_1_neurons=sorted(self.analysis['Ensembles']['EnsembleNeurons'][0])
    #     ensemble_1_activations=sorted(self.analysis['Ensembles']['Indices'][0])

    #     self.full_data=self.analysis_object.full_data
    #     self.stimulus_table=self.analysis_object.stimulus_table
    #     activity_traces=self.analysis_object.full_data['imaging_data']['Plane1']['Traces']['dfdt_binary']
    #     driftinfo=self.analysis_object.full_data['visstim_info']['Drifting_Gratings']
        
  
        
    #     for j ,i in enumerate(self.analysis['Ensembles']['Networks']):
    #        fig = plt.figure(figsize=(16,9),dpi=dpi)
    #        ax = fig.add_axes([0.05, 0.2, 0.9, 0.7])  # span the whole figure
    #        ax.imshow(i,cmap='binary', aspect='auto')
    #        ax.set_xlabel('Cell')
    #        fig.supylabel('Cell')
    #        fig.suptitle('Ensemble '+str(j+1)+' Network')
           
           
    #     colors_ = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(n)))
    #     colors = colors_(len(self.analysis['Ensembles']['Vectors']))   
    #     for j ,i in enumerate(self.analysis['Ensembles']['Vectors']):
    #        fig = plt.figure(figsize=(16,9),dpi=dpi)
    #        ax = fig.add_axes([0.05, 0.2, 0.9, 0.7])  # span the whole figure
    #        ax.imshow(i,cmap='binary', aspect='auto')
    #        ax.set_xlabel('Vectors')
    #        fig.supylabel('Cell')
    #        fig.suptitle('Ensemble '+str(j+1)+' Network')
           
           
    #        ensemble_neurons=sorted(self.analysis['Ensembles']['EnsembleNeurons'][j])
    #        for ne in ensemble_neurons:
    #            ax.axhspan(ne-0.5, ne+0.5, facecolor=colors[j], alpha=0.3)
           
            
         
    #     ensemble_vis_stim_info=[]   
         
    #     for l, ensemble in enumerate(self.analysis['Ensembles']['Indices']):
        
    #         stim_indexes = 1000*np.ones((1,len(ensemble)))
            
    #         for k, activation in enumerate(ensemble):
    #             for i, row in  self.stimulus_table['drifting_gratings'].iterrows():
    #                 range_to_test=np.arange(row.start, row.end)
                    
    #                 if activation in range_to_test:
    #                     stim_indexes[0,k]=np.int32(row.name)

    #         active_stim=np.unique(stim_indexes)     
                
    #         active_ensemble_trials=self.stimulus_table['drifting_gratings'].iloc[active_stim[:-1]]
            
    #         ensemble_vis_stim_info.append((stim_indexes,active_ensemble_trials))
            
            
    #     ensemble_vis_stim_info_corrected=copy.copy(ensemble_vis_stim_info)    
        
    #     for j, ensemb in enumerate(ensemble_vis_stim_info_corrected):
    #         for k, idx in enumerate(ensemb[0][0]):
    #             if idx!=1000:
    #                 ensemble_vis_stim_info_corrected[j][0][0][k]=ensemble_vis_stim_info[j][1].loc[int(idx)].orientation
            
            
    #     fig,ax=plt.subplots(4,2, figsize=(16,9))
    #     for mm, ensemble in enumerate(ensemble_vis_stim_info_corrected):
    #         # plt.hist(ensemble_vis_stim_info_corrected[0][0][0][ensemble_vis_stim_info_corrected[0][0][0]!=1000])
    #         no1000=ensemble[0][0][ensemble[0][0]!=1000]
    #         nonan=no1000[np.isnan(no1000)]=360
    #         # _ = ax[int(mm/2),mm%2].hist(no1000, bins=np.linspace(0,360,9))
    #         hist, bin_edges = np.histogram(no1000, bins=np.linspace(0,360,9))
    #         ax[int(mm/2),mm%2].pie(hist, labels=bin_edges[:-1], autopct='%1.1f%%',
    #         shadow=True, startangle=90)# arguments are passed to np.histogram


    #     plt.pie(hist, labels=bin_edges[:-1], autopct='%1.1f%%',
    #     shadow=True, startangle=90)
           

    


        
        
    def plotting_summary(self):


        neuron_id=self.analysis['Ensembles']['NeuronID']
        raster=self.analysis['Raster']
        vector_id=self.analysis['Ensembles']['VectorID'].astype('uint64')

        
        ensemble_structure=np.vstack((self.analysis['Ensembles']['StructureWeightsSignificant'][:,neuron_id], self.analysis['NonEnsembles']['StructureWeightsSignificant'][:,neuron_id]))
        structure=ensemble_structure.T
        [n_neurons,n_ensembles] = structure.shape
        
        neuronsorted=raster[neuron_id,:][::-1]
        vectorsorted=raster[:,vector_id]
        neurvectorsorted=neuronsorted[:,vector_id]
        
        pixel_per_bar = 4
        dpi = 100
        

          
        fig = plt.figure(figsize=(16,9))
        ax = fig.add_axes([0.05, 0.2, 0.9, 0.7])  # span th
        ax.imshow(raster[neuron_id,:], aspect='auto')
        ax.set_xlabel('Frame')
        fig.supylabel('Ensemble One Cells')
        fig.suptitle('Ensemble 1 Activity')
        plt.show()

        
        pathtosav=r'D:\Projects\LabNY\Full_Mice_Pre_Processed_Data\Mice_Projects\Interneuron_Imaging\G2C\Ai14\SPKG\data\JesusRuns'

        filename = os.path.join(pathtosav,"raster.pdf")
        self.save_multi_image(filename)
    
        self.plot_networks()
        filename = os.path.join(pathtosav,"networks.pdf")
        self.save_multi_image(filename)
        # self.plot_vector_clustering()
        # filename = os.path.join(pathtosav,"vectoclustering.pdf")
        # self.save_multi_image(filename)
        # self.plotting_ensembles()
        # filename = os.path.join(pathtosav,"ensembles.pdf")
        # self.save_multi_image(filename)



    
if __name__ == "__main__":
    
    jesus_run_file=r'D:\Projects\LabNY\Full_Mice_Pre_Processed_Data\Mice_Projects\Interneuron_Imaging\G2C\Ai14\SPHV\data\JesusRuns\210615_SPHV_FOV1_10minspont_50024_narrored_920_with-000_20220510-132101_MCMC_Plane1_Full_All_jesus_results.pkl'
    
    res=JesusEnsemblesResults(analysis_path=jesus_run_file)
    self=res

#%% rerun for tesitng
    # raster=res.analysis['Raster']
    # res=JesusEnsemblesResults(input_raster=raster)
    # self=res

    
    
   