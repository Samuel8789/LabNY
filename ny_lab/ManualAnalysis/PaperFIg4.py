# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 07:38:20 2024

@author: sp3660
"""
#FRTOM STEP BY STEP PROCESSING
# CHNAGE CELL TYPE AND RUN ALL CELLS AGAINE WHEN CHANGING CELL TYPES

# Figure 3 Pnael B
# Mult traces of 8 chandelier from same FOV

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy import stats
from matplotlib.colors import Normalize
import os
from scipy.stats import wilcoxon, ttest_ind
from sys import platform
from pathlib import Path

mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['svg.fonttype'] = 'none'

if platform == "linux" or platform == "linux2":
    fig_four_basepath=Path(r'/home/samuel/Dropbox/Projects/LabNY/ChandPaper/Fig4')
elif platform == "win32":
    fig_four_basepath=Path(r'C:\Users\sp3660\Desktop\ChandPaper\Fig4')
    


#%%
trial_activity=cell_act_full
timewindow=xwindow
sorting_peaks=[np.flip(np.argsort(cell_mean_peaks['opto_blank'])),np.flip(np.argsort(cell_mean_peaks['control_blank'])),np.flip(np.argsort(cell_mean_peaks['opto_grating'])),np.flip(np.argsort(cell_mean_peaks['control_grating']))]

optoactivity=trial_activity['opto_blank'][sorting_peaks[0],:,:]/max(trial_activity['opto_blank'].max(),trial_activity['opto_grating'].max(),trial_activity['control_blank'].max(),trial_activity['control_grating'].max())*100
controlactivity=trial_activity['control_blank'][sorting_peaks[1],:,:]/max(trial_activity['opto_blank'].max(),trial_activity['opto_grating'].max(),trial_activity['control_blank'].max(),trial_activity['control_grating'].max())*100
optoactivity_grating=trial_activity['opto_grating'][sorting_peaks[0],:,:]/max(trial_activity['opto_blank'].max(),trial_activity['opto_grating'].max(),trial_activity['control_blank'].max(),trial_activity['control_grating'].max())*100
controlactivity_grating=trial_activity['control_grating'][sorting_peaks[1],:,:]/max(trial_activity['opto_blank'].max(),trial_activity['opto_grating'].max(),trial_activity['control_blank'].max(),trial_activity['control_grating'].max())*100

optoactivity=trial_activity['opto_blank'][sorting_peaks[0],:,:]
controlactivity=trial_activity['control_blank'][sorting_peaks[1],:,:]
optoactivity_grating=trial_activity['opto_grating'][sorting_peaks[0],:,:]
controlactivity_grating=trial_activity['control_grating'][sorting_peaks[1],:,:]

# for pyramidal cell tuning and cell rewposnivenes
# Define the orientation pairs to combine
orientation_mapping = {
    '0': ['0', '180'],
    '45': ['45', '225'],
    '90': ['90', '270'],
    '135': ['135', '315']
}
# orientation_mapping = {
#     '0': ['0'],
#     '180': ['180'],
#     '45': ['45'],
#     '225': [ '225'],
#     '90': ['90'],
#     '270': ['270'],
#     '135': ['135'],
#     '315': ['315']
# }
# Create an empty list to store the combined data
combined_data = []

# Loop through the mapping and combine arrays for each orientation
for new_orientation, old_orientations in orientation_mapping.items():
    combined_array = np.concatenate([trial_activity[orientation] for orientation in old_orientations], axis=1)
    num_cells, num_trials, num_frames = combined_array.shape
    
    # Flatten the combined array to create a DataFrame
    for cell in range(num_cells):
        for trial in range(num_trials):
            for frame in range(num_frames):
                combined_data.append([cell, trial, timewindow[frame], combined_array[cell, trial, frame], new_orientation])

# Create the DataFrame
df_combined = pd.DataFrame(combined_data, columns=['Cell', 'Trial', 'Time', 'Value', 'Orientation'])
# Print a snippet of the DataFrame
print(df_combined.head())



data_list = []

for cell in range(optoactivity.shape[0]):
    for trial in range(optoactivity.shape[1]):
        for frame in range(optoactivity.shape[2]):
            data_list.append([cell, trial, timewindow[frame], optoactivity[cell, trial, frame], 'Opto','Blank'])
            data_list.append([cell, trial, timewindow[frame], controlactivity[cell, trial, frame], 'Control','Blank'])
            
    for trial in range(optoactivity_grating.shape[1]):
        for frame in range(optoactivity_grating.shape[2]):
            data_list.append([cell, trial, timewindow[frame], optoactivity_grating[cell, trial, frame], 'Opto','Grating'])
            data_list.append([cell, trial, timewindow[frame], controlactivity_grating[cell, trial, frame], 'Control','Grating'])


df = pd.DataFrame(data_list, columns=['Cell', 'Trial', 'Time', 'Value', 'Treatment','Stimuli'])
control_df=df[(df['Treatment'] == 'Control') & (df['Stimuli'] == 'Blank' )]


#%% SINGLE CHANDELIER TRACES 4 CONDITIONS

cell_data = df[df['Cell'] == 0]

# Set up the matplotlib figure size (convert from millimeters to inches)

for stimul in ('Blank','Grating'):
    fig, ax = plt.subplots(figsize=(600/25.4, 500/25.4))  # 30x25 mm to inches
    
    
    # Plot individual trials for Control activity with reduced line widt
    sns.lineplot(data=cell_data[(cell_data['Treatment'] == 'Control') & (cell_data['Stimuli'] == stimul )], 
                 x='Time', y='Value', 
                 hue='Trial', 
                 palette='gray', 
                 legend=False,   # Turn off legend for this plot
                 alpha=0.8, 
                 
                 linewidth=0.5, 
                 ax=ax)
    
    # Plot average trial for Control activity
    control_avg = cell_data[(cell_data['Treatment'] == 'Control') & (cell_data['Stimuli'] == stimul )].groupby('Time')['Value'].mean().reset_index()
    sns.lineplot(data=control_avg, 
                 x='Time', y='Value', 
                 color='black', 
                 linewidth=2.5, 
                 ax=ax, 
                 legend=False)  # Turn off legend for this plot
    
    # Plot individual trials for Optoactivity with reduced line width
    sns.lineplot(data=cell_data[(cell_data['Treatment'] == 'Opto') & (cell_data['Stimuli'] == stimul )], 
                 x='Time', y='Value', 
                 hue='Trial', 
                 palette='Blues', 
                 legend=False,   # Turn off legend for this plot
                 alpha=0.8, 
                 linewidth=0.5, 
                 ax=ax)
    
    # Plot average trial for Optoactivity
    opto_avg = cell_data[(cell_data['Treatment'] == 'Opto') & (cell_data['Stimuli'] == stimul )].groupby('Time')['Value'].mean().reset_index()
    sns.lineplot(data=opto_avg, 
                 x='Time', y='Value', 
                 color='blue', 
                 linewidth=2.5, 
                 ax=ax, 
                 legend=False)  # Turn off legend for this plot
    
    # Set plot title and labels
    ax.set_xlabel('Time (s)', fontsize=6)          # X-axis label with font size 6
    ax.set_ylabel('df/f (%)', fontsize=6)          # Y-axis label with font size 6
    
    # Add vertical dashed lines at time 0 and time 1 with black color
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
    ax.axvline(x=1, color='black', linestyle='--', linewidth=1.5)
    
    # Remove the top and right spines
    sns.despine(ax=ax)
    
    # Set axis line width
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    plt.grid(False)  # Remove grid

    # Remove x-axis margins
    ax.margins(x=0)
    ax.set_ylim(-0.5,0.6 )  # Adjust for extra padding

    # Customize ticks and tick labels
    ax.tick_params(axis='both', which='major', labelsize=6, width=1)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot as an SVG file
    # plt.savefig(fr'C:\Users\sp3660\Desktop\ChandPaper\Fig4\activity_plot_cell_0_grating_{stimul}.svg', format='svg', bbox_inches='tight')
    
    # Show the plot
    plt.show()




#%% ALL CHANDELIER TIRAL AVERAGED TRACES ALL CONDITIONS
# Plotting settings
figsize_inches = (40 / 25.4, 40 / 25.4)  # Figure size in inches
fontsize = 5  # Font size for ticks

# Function to plot trial-averaged activity for each cell individually using OOP style
def plot_trial_averaged_activity_individual(df, stimuli_type):
    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=figsize_inches)

    # Filter dataframe by stimuli type
    df_stimuli = df[df['Stimuli'] == stimuli_type]

    # Get unique cell ids
    unique_cells = df_stimuli['Cell'].unique()

    # Loop through each cell
    for cell in unique_cells:
        cell_data = df_stimuli[df_stimuli['Cell'] == cell]

        # Calculate trial-averaged activity for Opto and Control
        mean_opto = cell_data[cell_data['Treatment'] == 'Opto'].groupby('Time')['Value'].mean()
        mean_control = cell_data[cell_data['Treatment'] == 'Control'].groupby('Time')['Value'].mean()

        # Plot trial-averaged activity for each cell
        ax.plot(mean_opto.index, mean_opto.values, color='blue', alpha=0.3, linewidth=0.5)
        ax.plot(mean_control.index, mean_control.values, color='black', alpha=0.3, linewidth=0.5)

    # Styling the plot
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.1)
    ax.spines['bottom'].set_linewidth(0.1)
    ax.grid(False)  # Remove grid
    ax.set_ylim([-0.5, 1])
    fig.tight_layout()
    ax.margins(x=0)

    # plt.savefig(fr'C:\Users\sp3660\Desktop\ChandPaper\Fig4\trial_averaged_all_chandeliers_{stimuli_type}.svg', format='svg', bbox_inches='tight')

    # Show the plot
    plt.show()

# Plot for Blank stimuli
plot_trial_averaged_activity_individual(df, 'Blank')
# Plot for Grating stimuli
plot_trial_averaged_activity_individual(df, 'Grating')
#%% FIRST DO ORIANTETAION TUNING ANALYSIS OF PYRAMIDAL CELLS WITH BOOSTRAYTPINON FULL GRAITNG DATASET
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

# Ensure Seaborn styles are used
sns.set(style="whitegrid")

# Assuming df_combined is already defined and contains the relevant data
# Convert 'Orientation' column to numeric if it's not already
df_combined['Orientation'] = pd.to_numeric(df_combined['Orientation'], errors='coerce')

# Define the orientations
orientations = df_combined['Orientation'].unique()
orientations.sort()  # Sort orientations for consistent plotting
orientations_rad = np.deg2rad(orientations)  # Convert to radians

# Initialize lists to store results
real_stats = {}
bootstrap_stats = {}

# Set random seed for reproducibility
np.random.seed(42)

# Perform calculations for each cell
for cell in df_combined['Cell'].unique():
    cell_data = df_combined[df_combined['Cell'] == cell]
    time_window_data = cell_data[(cell_data['Time'] >= 0) & (cell_data['Time'] <= 1)]
    real_stats[str(cell)]={}
    bootstrap_stats[str(cell)]={}
    
    # Calculate real statistics per orientation
    for orientation in orientations:
        real_stats[str(cell)][str(orientation)]={}
        orientation_data = time_window_data[time_window_data['Orientation'] == orientation]
        
        # Trial-averaged signal
        trial_averaged_signal = orientation_data.groupby('Time')['Value'].mean()
        
        # Calculate real statistics from the trial-averaged signal
        real_max = trial_averaged_signal.max()
        real_mean = trial_averaged_signal.mean()
        real_median = trial_averaged_signal.median()
        
        real_stats[str(cell)][str(orientation)]['mean']=np.array(real_mean)
        real_stats[str(cell)][str(orientation)]['max']=np.array(real_max)
        real_stats[str(cell)][str(orientation)]['median']=np.array(real_median)
    
    # Parameters
    num_bootstrap_iterations = 100
    num_samples_per_iteration = 80
    time_window_start = 0
    time_window_end = 1
 
    # Initialize lists to store bootstrap results
    bootstrap_means = []
    bootstrap_maxs = []
    bootstrap_medians = []
 
    # Set random seed for reproducibility
    np.random.seed(42)
 
    
    df_flattened = cell_data.drop(columns=['Cell'])
    
    # Create a unique identifier for each combination of 'Trial' and 'Orientation'
    # This will give us a unique trial number from 0 to 319
    df_flattened['Trial_Number'] = pd.factorize(df_flattened['Trial'].astype(str) + '_' + df_flattened['Orientation'].astype(str))[0]


    # List of unique trials
    unique_trials = df_flattened['Trial_Number'].unique()
 
    # Bootstrap sampling
    for _ in range(num_bootstrap_iterations):
        # Sample with replacement, ensuring 80 unique trials
        sampled_trials = np.random.choice(unique_trials, size=num_samples_per_iteration, replace=True)
            
 
        # Initialize DataFrame to collect sampled data
        sampled_data = pd.DataFrame(columns=df_flattened.columns)
        
        # Collect data for the sampled trials, including duplicates
        for i,trial in enumerate(sampled_trials):
            trialdf=df_flattened[df_flattened['Trial_Number'] == trial]
            
            trialdf['Trial_Number']=i
            
            sampled_data = pd.concat([sampled_data, trialdf], ignore_index=True)
        
        # Verify that the sampled data contains exactly 80 trials
        assert sampled_data['Trial_Number'].nunique() == num_samples_per_iteration, \
            f"Expected {num_samples_per_iteration} unique trials, got {sampled_data['Trial'].nunique()}"
 
        # Compute trial-averaged signal for each time point
        trial_averaged_signal = sampled_data.groupby('Time')['Value'].mean()
        
        # Filter data within the time window of interest (0 to 1 second)
        time_window_data = trial_averaged_signal[(trial_averaged_signal.index >= time_window_start) & 
                                                 (trial_averaged_signal.index <= time_window_end)]
        
        # Calculate statistics
        bootstrap_means.append(time_window_data.mean())
        bootstrap_maxs.append(time_window_data.max())
        bootstrap_medians.append(time_window_data.median())
 
    # Convert results to numpy arrays for easy analysis
    bootstrap_stats[str(cell)]['mean']=np.array(bootstrap_means)
    bootstrap_stats[str(cell)]['max']=np.array(bootstrap_maxs)
    bootstrap_stats[str(cell)]['median']=np.array(bootstrap_medians)
    


# Ensure Seaborn styles are used
sns.set(style="whitegrid")

# Initialize lists to store results for plotting
polar_data = {}
histograms = {}
prefered_or={}
# Confidence level
confidence_level = 0.95
plt.close('all')
# Perform statistical comparison and plotting for each cell
for cell in df_combined['Cell'].unique():
    cell_real_stats = real_stats[str(cell)]
    cell_bootstrap_stats = bootstrap_stats[str(cell)]
    histograms[str(cell)]={}
    # Initialize polar plot data
    angles_rad = np.deg2rad([0, 45, 90, 135])
    polar_data[str(cell)]={}
    # Prepare data for the polar plot
    for orientation in orientations:
        real_mean = cell_real_stats[str(orientation)]['mean']
        real_max = cell_real_stats[str(orientation)]['max']
        real_median = cell_real_stats[str(orientation)]['median']
        
        # Bootstrap statistics
        bootstrap_mean = np.mean(cell_bootstrap_stats['mean'])
        bootstrap_max = np.mean(cell_bootstrap_stats['max'])
        bootstrap_median = np.mean(cell_bootstrap_stats['median'])
        
        # Append to polar data
        polar_data[str(cell)][str(orientation)]={
            'Real_Mean': real_mean,
            'Bootstrap_Mean': bootstrap_mean,
            'Real_Max': real_max,
            'Bootstrap_Max': bootstrap_max,
            'Real_Median': real_median,
            'Bootstrap_Median': bootstrap_median
        }
        
        # Histograms
        histograms[str(cell)]['mean']=(cell_bootstrap_stats['mean'])
        histograms[str(cell)]['max']=(cell_bootstrap_stats['max'])
        histograms[str(cell)]['median']=(cell_bootstrap_stats['median'])
    
    # Convert polar_data to DataFrame for plotting
    df_polar = pd.DataFrame(polar_data[str(cell)])
    # Plot histograms
    # for stat in ['mean', 'max', 'median']:
    #     plt.figure(figsize=(12, 4))
    #     sns.histplot(histograms[str(cell)][stat], bins=30, kde=True)
    #     plt.title(f'Bootstrap Distribution of {stat.capitalize()}')
    #     plt.xlabel(stat.capitalize())
    #     plt.ylabel('Frequency')
    #     plt.savefig(fr'C:\Users\sp3660\Desktop\ChandPaper\Fig4\histogram_{cell}_{stat}.svg')  # Save histogram as SVG
    #     plt.close()
        
    CIs={}
    # Calculate confidence intervals
    for stat in ['mean', 'max', 'median']:
        data = np.array(histograms[str(cell)][stat])
        CIs[stat] = np.percentile(data, [(1-confidence_level)/2 * 100, (1+confidence_level)/2 * 100])


    grouped_angles_half = [0, 45, 90, 135, 180]  # Add 180 to complete the half-circle
    grouped_activity_half = [df_polar['0'].loc['Real_Max'],df_polar['45'].loc['Real_Max'],df_polar['90'].loc['Real_Max'], df_polar['135'].loc['Real_Max'], df_polar['0'].loc['Real_Max']]
 
    # Convert grouped angles to radians for polar plotting
    grouped_angles_half_rad = np.deg2rad(grouped_angles_half)
    
    
       # Plot shaded area
    if any(grouped_activity_half> CIs['max'][1]):
        
      
        prefered_or[str(cell)]=  grouped_angles_half[np.argwhere(grouped_activity_half> CIs['max'][1])[0][np.argmax(np.array(grouped_activity_half)[np.argwhere(grouped_activity_half> CIs['max'][1])[0]])]]
    
        plt.figure(figsize=(5, 5))
        ax = plt.subplot(111, polar=True)
        ax.plot(grouped_angles_half_rad, grouped_activity_half, 'o-', label='Activity at grouped angles', color='r')
        ax.fill_between(grouped_angles_half_rad, CIs['max'][0], CIs['max'][1], alpha=0.3, label=f'95 % CI')
    
    
        # Set to half-circle (0 to 180 degrees)
        ax.set_theta_offset(np.pi / 2)  # Start at 90 degrees (top)
        ax.set_theta_direction(-1)      # Clockwise direction
        ax.set_thetamax(180)            # Limit to 180 degrees
    
        # Set custom ticks for the half-circle
        ax.set_thetagrids(range(0, 181, 45), labels=grouped_angles_half)
        ax.set_rmin(-5)  # Minimum radial limit
        ax.set_rmax(10)    # Maximum radial limit
    
        plt.legend()
        plt.savefig(fr'C:\Users\sp3660\Desktop\ChandPaper\Fig4\polar_plot_{cell}.svg')  # Save polar plot as SVG
        plt.close()
    
        [str(cell)]
        f,ax=plt.subplots(2,2)    
        mins=[]
        maxs=[]
        for i,a in enumerate(ax.flatten()):
            if grouped_angles_half[i]!=prefered_or[str(cell)]:
                a.plot(df_combined[(df_combined['Cell']==cell) & (df_combined['Orientation']==grouped_angles_half[i])].groupby('Time')['Value'].mean(),color='k')
            else:
                a.plot(df_combined[(df_combined['Cell']==cell) & (df_combined['Orientation']==grouped_angles_half[i])].groupby('Time')['Value'].mean(), color='r')
            mins.append(min(df_combined[(df_combined['Cell']==cell) & (df_combined['Orientation']==grouped_angles_half[i])].groupby('Time')['Value'].mean()))
            maxs.append(max(df_combined[(df_combined['Cell']==cell) & (df_combined['Orientation']==grouped_angles_half[i])].groupby('Time')['Value'].mean()))

        for i,a in enumerate(ax.flatten()):
            a.set_ylim([min(mins), max(maxs)+1])
            
        plt.tight_layout()
        plt.savefig(fr'C:\Users\sp3660\Desktop\ChandPaper\Fig4\individual_traces_{cell}.svg')  # Save individual traces as SVG
        plt.close()
           
        
   

#%% PYRAMIDAL CELL COMARISON RESPONSIVE VS NON RESPONSIVE CELLS NO
# active_cells = np.array([int(cell) for cell in prefered_or.keys()]).astype(np.int64)
# inactive_cells = df['Cell'].unique()[~np.isin( df['Cell'].unique(),active_cells)]
# cell_subsets={'active_cells':active_cells,'inactive_cells':inactive_cells}
# cell_subsets={'active_cells':active_cells}

# save_path=r'C:\Users\sp3660\Desktop\ChandPaper\Fig4'
# dataf=df
# plt.close('all')
# def analyze_and_plot_pie(subset, stimulus_type, save_name, stdf):
#     results = []
#     increased_cells = []
#     decreased_cells = []
#     unchanged_cells = []
    
#     # Number of cells to calculate Bonferroni corrected p-value
#     num_comparisons = len(subset)
#     correction_factor = num_comparisons
#     # Function to plot value traces for each trial
#    # Function to plot trial-averaged data and single-trial data
#     def plot_traces_with_sem(stdf):
#         f,axs=plt.subplots(2)

#         for i, data in enumerate(('Blank', 'Grating')):
#             stimdf=stdf[stdf['Stimuli'] == data]
#             # Define colors
#             opto_color = 'blue'
#             control_color = 'black'
            
#             # Get unique trials
#             trials = stimdf['Trial'].unique()
            
#             # Plotting trial-averaged data with SEM
#             for treatment, color in [('Opto', opto_color), ('Control', control_color)]:
#                 # Filter data for the current treatment
#                 treatment_data = stimdf[stimdf['Treatment'] == treatment]
                
#                 # Calculate trial-averaged data and SEM
#                 trial_avg = treatment_data.groupby('Time')['Value'].mean()
#                 trial_sem = treatment_data.groupby('Time')['Value'].sem()
                
#                 # Plot trial-averaged data
#                 axs[i].plot(trial_avg.index, trial_avg.values, color=color, label=f'{treatment} - Mean')
#                 axs[i].fill_between(trial_avg.index, trial_avg - trial_sem, trial_avg + trial_sem, color=color, alpha=0.3, label=f'{treatment} - SEM')
                
#                 # Plot single-trial data
#                 for trial in trials:
#                     single_trial_data = treatment_data[treatment_data['Trial'] == trial]
#                     axs[i].plot(single_trial_data['Time'], single_trial_data['Value'], color=color, alpha=0.1)
        
#             # Add labels and title
#             plt.xlabel('Time')
#             plt.ylabel('Value')
#             plt.title('Trial-Averaged and Single-Trial Value Traces')
#             # plt.legend(loc='upper right')
#             plt.tight_layout()
            
#             # Save and show the plot
#             plt.savefig(r'C:\Users\sp3660\Desktop\ChandPaper\Fig4\Trial_Averaged_Opto_Control_Trace.svg', format='svg')
#             plt.show()
#     # Loop over each cell to perform the test
#     for cell in subset:
#         # Filter data for the current cell and stimulus type
#         cell_data = stdf[(stdf['Cell'] == cell) & (stdf['Stimuli'] == stimulus_type)]
        
   
#         if stimulus_type=='Blank':
#             # Call the function with your DataFrame
#             plot_traces_with_sem(dataf[dataf['Cell'] == cell])

        
        
        
#         window_df = cell_data[(cell_data['Time'] >= 0) & (cell_data['Time'] <= 1.5)]

        
#         # Get mean values for 'Control' and 'Opto' treatments during the window
#         control_values = window_df[window_df['Treatment'] == 'Control']['Value'].groupby(window_df['Trial']).mean()
#         opto_values = window_df[window_df['Treatment'] == 'Opto']['Value'].groupby(window_df['Trial']).mean()
        
        
        
#         # Perform Wilcoxon signed-rank test
#         if len(control_values) > 0 and len(opto_values) > 0:  # Ensure there are values to compare
#             stat, p_value = wilcoxon(control_values, opto_values)
#         else:
#             continue  # Skip this cell if there are no valid comparisons
        
#         corrected_p_value = p_value * correction_factor
    
#         # Determine if there's a significant increase or decrease
#         if corrected_p_value < 0.05:  # Using corrected p-value
#             control_mean = control_values.mean()
#             opto_mean = opto_values.mean()
#             if opto_mean > control_mean:
#                 increased_cells.append(cell)
#             elif opto_mean < control_mean:
#                 decreased_cells.append(cell)
                
#         else:
#             unchanged_cells.append(cell)
        
#         # Store the results
#         results.append({
#             'Cell': cell,
#             'Control Mean': control_values.mean(),
#             'Opto Mean': opto_values.mean(),
#             'Statistic': stat,
#             'P-Value': p_value
#         })
    
#     # Create a DataFrame to display the results
#     results_df = pd.DataFrame(results)
    
#     # Plot pie chart with custom colors
#     plt.figure(figsize=(6, 6))
#     proportions = [len(increased_cells), len(decreased_cells), len(unchanged_cells)]
#     labels = ['Increased', 'Decreased', 'Unchanged']
#     colors = ['yellow', 'cyan', 'gray']
#     plt.pie(proportions, labels=labels, autopct='%1.1f%%', colors=colors)
#     plt.title(f'Proportion of {save_name} in {stimulus_type}  with Significant Changes')

#     plt.savefig(os.path.join(save_path, f'{save_name}_{stimulus_type}_PieChart.svg'), format='svg')
#     plt.show()
    
# plt.close('all')  # Close the figure after saving to avoid displaying it
# # Loop over each subset (active and inactive cells) and both stimulus types
# for keysub, subset in cell_subsets.items():
#     for stimulus in ['Blank', 'Grating']:
#         analyze_and_plot_pie(subset, stimulus, keysub,dataf)





#%% CLCULATE METRISC FOR DIFFERNET TIME WINDOWS
import pandas as pd
import numpy as np

time_windows = [(-1, 0), (0, 1), (0, 2)]

# Initialize dictionary to store results
results = {
    'Cell': [], 
    'Stimulus': [], 'Treatment': [],
    'Mean_-1_to_0': [], 'Mean_0_to_1': [], 'Mean_0_to_2': [], 
    'Median_-1_to_0': [], 'Median_0_to_1': [], 'Median_0_to_2': [], 
    'Max_-1_to_0': [], 'Max_0_to_1': [], 'Max_0_to_2': [], 
    'Mean_Difference': [], 'Median_Difference': [], 'Max_Difference': []
}

# Get unique cells
cells = df['Cell'].unique()

for cell in cells:
    cell_data = df[df['Cell'] == cell]
    
    # Process each combination of stimulus and treatment
    for stimulus in ['Blank', 'Grating']:
        for treatment in ['Control', 'Opto']:
            subset = cell_data[(cell_data['Stimuli'] == stimulus) & (cell_data['Treatment'] == treatment)]
            
            # Initialize dictionaries for storing metrics
            mean_values = {}
            median_values = {}
            max_values = {}
            
            # Calculate mean, median, and max for each time window
            for start, end in time_windows:
                window_data = subset[(subset['Time'] >= start) & (subset['Time'] <= end)]
                mean_values[f'Mean_{start}_to_{end}'] = window_data['Value'].mean()
                median_values[f'Median_{start}_to_{end}'] = window_data['Value'].median()
                max_values[f'Max_{start}_to_{end}'] = window_data['Value'].max()
            
            # Append results to the dictionary
            results['Cell'].append(cell)
            results['Stimulus'].append(stimulus)
            results['Treatment'].append(treatment)
            results['Mean_-1_to_0'].append(mean_values.get('Mean_-1_to_0', np.nan))
            results['Mean_0_to_1'].append(mean_values.get('Mean_0_to_1', np.nan))
            results['Mean_0_to_2'].append(mean_values.get('Mean_0_to_2', np.nan))
            results['Median_-1_to_0'].append(median_values.get('Median_-1_to_0', np.nan))
            results['Median_0_to_1'].append(median_values.get('Median_0_to_1', np.nan))
            results['Median_0_to_2'].append(median_values.get('Median_0_to_2', np.nan))
            results['Max_-1_to_0'].append(max_values.get('Max_-1_to_0', np.nan))
            results['Max_0_to_1'].append(max_values.get('Max_0_to_1', np.nan))
            results['Max_0_to_2'].append(max_values.get('Max_0_to_2', np.nan))
            
            # Calculate differences for the Control Grating condition
            mean_0_to_1 = mean_values.get('Mean_0_to_2', np.nan)
            mean_neg1_to_0 = mean_values.get('Mean_-1_to_0', np.nan)
            median_0_to_1 = median_values.get('Median_0_to_2', np.nan)
            median_neg1_to_0 = median_values.get('Median_-1_to_0', np.nan)
            max_0_to_1 = max_values.get('Max_0_to_2', np.nan)
            max_neg1_to_0 = max_values.get('Max_-1_to_0', np.nan)
                
            results['Mean_Difference'].append(mean_0_to_1 - mean_neg1_to_0)
            results['Median_Difference'].append(median_0_to_1 - median_neg1_to_0)
            results['Max_Difference'].append(max_0_to_1 - max_neg1_to_0)
          
# Create a DataFrame with results
results_df = pd.DataFrame(results)
# Now calculate the ratios of the means in the 0 to 2 window across all four combinations of stimulus and treatment
# Data structure to store the results
# Data structure to store the results
# Now calculate the ratios of the means in the 0 to 2 window across all four combinations of stimulus and treatment
ratios = []

# Group by cell
for cell in cells:
    cell_results = results_df[results_df['Cell'] == cell]
    
    # Get the mean values for each combination
    mean_blank_control = cell_results.loc[(cell_results['Stimulus'] == 'Blank') & (cell_results['Treatment'] == 'Control'), 'Mean_0_to_2'].values[0]
    mean_blank_opto = cell_results.loc[(cell_results['Stimulus'] == 'Blank') & (cell_results['Treatment'] == 'Opto'), 'Mean_0_to_2'].values[0]
    mean_grating_control = cell_results.loc[(cell_results['Stimulus'] == 'Grating') & (cell_results['Treatment'] == 'Control'), 'Mean_0_to_2'].values[0]
    mean_grating_opto = cell_results.loc[(cell_results['Stimulus'] == 'Grating') & (cell_results['Treatment'] == 'Opto'), 'Mean_0_to_2'].values[0]
    
    # Calculate ratios
    ratio_opto_vs_control = mean_blank_opto - mean_blank_control
    ratio_grating_vs_control = mean_grating_control - mean_blank_control 
    ratio_opto_grating_vs_control = mean_grating_opto - mean_blank_control 
    ratio_opto_grating_vs_opto_blank = mean_grating_opto - mean_blank_opto 
    ratio_control_grating_vs_opto_grating = mean_grating_opto - mean_grating_control 

    
    # Append to the list
    ratios.append({
        'Cell': cell,
        'ratio_opto_vs_control': ratio_opto_vs_control,
        'ratio_grating_vs_control': ratio_grating_vs_control,
        'ratio_opto_grating_vs_control': ratio_opto_grating_vs_control,
        'ratio_opto_grating_vs_opto_blank': ratio_opto_grating_vs_opto_blank,
        'ratio_control_grating_vs_opto_grating': ratio_control_grating_vs_opto_grating

    })

# Create a DataFrame for the ratios
ratios_df = pd.DataFrame(ratios)

# Merge the ratios with the original results

# Merge the ratios with the original results
final_results_df = pd.merge(results_df, ratios_df, on='Cell')
final_results_df.to_csv(str(fig_four_basepath / 'metrics.csv'))


#%% SAME DETECTION OF SIGFNIFICANTLY MODIFIED CELLS AS IN FIG 3
# Step 1: Filter the dataframe for Time between 0 and 1 second
df_filtered = df[(df['Time'] >= 0) & (df['Time'] <= 1)]

# Step 2: Calculate mean values for each combination of Cell, Trial, Treatment, and Stimuli across time
mean_values = df_filtered.groupby(['Cell', 'Trial', 'Treatment', 'Stimuli']).agg({'Value': 'max'}).reset_index()
significance=0.05
# Step 3: Initialize results storage
significant_grating_control = []
significant_grating_opto = []
significant_grating_opto_only=[]
# Unique cells in the mean_values dataframe
unique_cells = mean_values['Cell'].unique()

# Step 4: Loop through each unique cell
for cell in unique_cells:
    # Filter for the current cell
    cell_data = mean_values[mean_values['Cell'] == cell]
    
    # Extract values for the relevant groups
    blank_control = cell_data[(cell_data['Stimuli'] == 'Blank') & (cell_data['Treatment'] == 'Control')]['Value']
    grating_control = cell_data[(cell_data['Stimuli'] == 'Grating') & (cell_data['Treatment'] == 'Control')]['Value']
    grating_opto = cell_data[(cell_data['Stimuli'] == 'Grating') & (cell_data['Treatment'] == 'Opto')]['Value']
    blank_opto = cell_data[(cell_data['Stimuli'] == 'Blank') & (cell_data['Treatment'] == 'Opto')]['Value']

    
    # Calculate mean values for comparison
    mean_blank_control = blank_control.mean() if not blank_control.empty else np.nan
    mean_grating_control = grating_control.mean() if not grating_control.empty else np.nan
    mean_grating_opto = grating_opto.mean() if not grating_opto.empty else np.nan
    mean_blank_opto = blank_opto.mean() if not blank_opto.empty else np.nan


    # Perform Wilcoxon tests
    if len(blank_control) > len(grating_control):
        blank_control_sample = np.random.choice(blank_control, size=len(grating_control), replace=False)
    else:
        blank_control_sample = blank_control
        
    if len(blank_control) > len(grating_opto):
        blank_control_sample_opto = np.random.choice(blank_control, size=len(grating_opto), replace=False)
    else:
        blank_control_sample_opto = blank_control
        
    if len(blank_control) > len(blank_opto):
        blank_control_sample_opto_only = np.random.choice(blank_control, size=len(blank_opto), replace=False)
    else:
        blank_control_sample_opto_only = blank_control
        


    # Test for Grating Control vs Blank Control
    if len(blank_control_sample) > 0 and len(grating_control) > 0:
        stat_gc, p_gc = wilcoxon(blank_control_sample, grating_control)
        # Check for significance and mean increase
        if p_gc < significance and mean_grating_control > mean_blank_control:
            significant_grating_control.append(cell)

    # Test for Grating Opto vs Blank Control
    if len(blank_control_sample_opto) > 0 and len(grating_opto) > 0:
        stat_go, p_go = wilcoxon(blank_control_sample_opto, grating_opto)
        # Check for significance and mean increase
        if p_go < significance and mean_grating_opto > mean_blank_control:
            significant_grating_opto.append(cell)
            

    # Test for Grating Opto vs Blank Opto
    if len(blank_control_sample_opto_only) > 0 and len(blank_opto) > 0:
        stat_go, p_go = wilcoxon(blank_control_sample_opto_only, blank_opto)
        # Check for significance and mean increase
        if p_go < significance and mean_blank_opto > mean_blank_control:
            significant_grating_opto_only.append(cell)
            
            
 
# Step 5: Create DataFrames for significant cells
significant_gc_df = pd.DataFrame(significant_grating_control, columns=['Significant Grating Control Cells'])
significant_go_df = pd.DataFrame(significant_grating_opto, columns=['Significant Grating Opto Cells'])
significant_bo_df = pd.DataFrame(significant_grating_opto_only, columns=['Significant Blank Opto Cells'])
print(len(significant_gc_df)*100/len(unique_cells))
print(len(significant_go_df)*100/len(unique_cells))



# Display the significant cells
print("Cells with Significant Increase in Grating Control:")
print(significant_gc_df)

print("\nCells with Significant Increase in Grating Opto:")
print(significant_go_df)
#%% SELECT CELLS BASED ON SOME ARBIOTRY THREHOLD THAT ARE VISUAL STIMULS RESPONSIVE


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


all_metrics={}
significance=0.05
# Assuming results_df is the DataFrame containing the difference calculations
for stimulus in ['Blank', 'Grating']:
    all_metrics[stimulus]={}
    for treatment in ['Control', 'Opto']:
        all_metrics[stimulus][treatment]={}

        filtdf_stimres=results_df[(results_df['Treatment']==treatment) & (results_df['Stimulus']==stimulus)]
        # Calculate standard deviations for each type of difference
        mean_diff_std = filtdf_stimres['Mean_Difference'].dropna().std()
        median_diff_std = filtdf_stimres['Median_Difference'].dropna().std()
        max_diff_std = filtdf_stimres['Max_Difference'].dropna().std()
        thresh=2
        # Define thresholds as 2 standard deviations
        mean_threshold = thresh * mean_diff_std
        median_threshold = thresh * median_diff_std
        max_threshold = thresh * max_diff_std

        # Set up the figure and axes for the histograms
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # Plot histogram for Mean Differences
        axs[0].hist(filtdf_stimres['Mean_Difference'].dropna(), bins=30, color='skyblue', edgecolor='black')
        axs[0].axvline(mean_threshold, color='red', linestyle='--', label=f'2 SD Threshold ({mean_threshold:.2f})')
        axs[0].set_title('Histogram of Mean Differences')
        axs[0].set_xlabel('Mean Difference')
        axs[0].set_ylabel('Frequency')
        axs[0].legend()

        # Plot histogram for Median Differences
        axs[1].hist(filtdf_stimres['Median_Difference'].dropna(), bins=30, color='lightgreen', edgecolor='black')
        axs[1].axvline(median_threshold, color='red', linestyle='--', label=f'2 SD Threshold ({median_threshold:.2f})')
        axs[1].set_title('Histogram of Median Differences')
        axs[1].set_xlabel('Median Difference')
        axs[1].set_ylabel('Frequency')
        axs[1].legend()

        # Plot histogram for Max Differences
        axs[2].hist(filtdf_stimres['Max_Difference'].dropna(), bins=30, color='salmon', edgecolor='black')
        axs[2].axvline(max_threshold, color='red', linestyle='--', label=f'2 SD Threshold ({max_threshold:.2f})')
        axs[2].set_title('Histogram of Max Differences')
        axs[2].set_xlabel('Max Difference')
        axs[2].set_ylabel('Frequency')
        axs[2].legend()

        # Adjust layout
        plt.tight_layout()

        # Save and show the figure
        plt.savefig('Differences_Histograms_with_Thresholds.svg', format='svg')
        plt.show()

        # Create subsets based on the thresholds
        mean_subset_stim = filtdf_stimres[filtdf_stimres['Mean_Difference'] > mean_threshold]['Cell']
        median_subset_stim = filtdf_stimres[filtdf_stimres['Median_Difference'] > median_threshold]['Cell']
        max_subset_stim = filtdf_stimres[filtdf_stimres['Max_Difference'] > max_threshold]['Cell']


        mean_subset_stim_un = filtdf_stimres[filtdf_stimres['Mean_Difference']< significance]['Cell']
        median_subset_stim_un  = filtdf_stimres[filtdf_stimres['Median_Difference'] < significance]['Cell']
        max_subset_stim_un  = filtdf_stimres[filtdf_stimres['Max_Difference']< significance]['Cell']


        # Example: Printing the number of cells in each subset
        print(f'{stimulus} {treatment} Number of cells with mean difference > {thresh} SD: {len(mean_subset_stim)}')
        print(f'{stimulus} {treatment} Number of cells with median difference > {thresh} SD: {len(median_subset_stim)}')
        print(f'{stimulus} {treatment} Number of cells with max difference > {thresh} SD: {len(max_subset_stim)}')
        all_metrics[stimulus][treatment]=[mean_subset_stim,median_subset_stim,max_subset_stim,mean_subset_stim_un,median_subset_stim_un,max_subset_stim_un]
        
        






#%% PLOT CELL ACTIVITY BASED ON BOTTSRAPING ANALYSIS OF TUNING

# Function to plot trial-averaged activity with individual cell traces
# Function to plot trial-averaged activity with individual cell traces
def plot_trial_averaged_activity(cells, stimulus_type, title):
    plt.figure(figsize=(8, 6))
    
    for treatment in ['Control', 'Opto']:
        subset = df[(df['Cell'].isin(cells)) & (df['Stimuli'] == stimulus_type) & (df['Treatment'] == treatment)]
        
        # Determine color scheme
        if treatment == 'Opto':
            grand_avg_color = 'darkblue'
            single_cell_colors = sns.light_palette("blue", n_colors=len(cells))
        else:
            grand_avg_color = 'black'
            single_cell_colors = sns.light_palette("black", n_colors=len(cells))
        
        # Plot individual cell traces
        for i, cell in enumerate(cells):
            cell_subset = subset[subset['Cell'] == cell]
            trial_avg = cell_subset.groupby(['Time'])['Value'].mean()
            sns.lineplot(x=trial_avg.index, y=trial_avg.values, color=single_cell_colors[i], alpha=0.3)
        
        # Plot the overall trial-averaged activity for the group of cells
        trial_avg = subset.groupby(['Time'])['Value'].mean()
        sns.lineplot(x=trial_avg.index, y=trial_avg.values, label=treatment, color=grand_avg_color)
    
    plt.title(f'{title} - {stimulus_type}')
    plt.xlabel('Time')
    plt.ylabel('Activity')
    plt.legend()
    plt.tight_layout()
    # plt.savefig(fr'C:\Users\sp3660\Desktop\ChandPaper\Fig4\{title}_{stimulus_type}_Activity.svg')


visually_active_cells=all_metrics['Grating']['Control'][2]
both_active_cells=all_metrics['Grating']['Opto'][2]
opto_only_cells=all_metrics['Blank']['Opto'][2]


visually_active_cells=significant_gc_df['Significant Grating Control Cells']
both_active_cells=significant_go_df['Significant Grating Opto Cells']
opto_only_cells=significant_bo_df['Significant Blank Opto Cells']

# Plot for active cells with Blank stimulus
plot_trial_averaged_activity(visually_active_cells, 'Blank', 'Active Cells')

# Plot for active cells with Grating stimulus
plot_trial_averaged_activity(visually_active_cells, 'Grating', 'Active Cells')

# Plot for inactive cells with Blank stimulus
plot_trial_averaged_activity(both_active_cells, 'Blank', 'Inactive Cells')

# Plot for inactive cells with Grating stimulus
plot_trial_averaged_activity(both_active_cells, 'Grating', 'Inactive Cells')




#%% PLOT STRIP PLOTS OF MEAN EVOKED ACTIVITY HERE IT CAN BE DONE WITH 0 to 2 or to 1 window and with 2 subset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# List of selected cells
for k, subset in {'visually_active':visually_active_cells,'opto+grating active': both_active_cells,'opto+blank active': all_metrics['Blank']['Opto'][0]}.items():
    for window in (2,):
        for metric in ('Mean',):#,'Median', 'Max'):
            # Extract mean values for the selected cells
            met_data = []
            
            for cell in subset:
                cell_results = results_df[results_df['Cell'] == cell]
                
                # Collect mean values
                met_wind_blank_control = cell_results.loc[(cell_results['Stimulus'] == 'Blank') & (cell_results['Treatment'] == 'Control'), f'{metric}_0_to_{window}'].values
                met_wind_blank_opto = cell_results.loc[(cell_results['Stimulus'] == 'Blank') & (cell_results['Treatment'] == 'Opto'), f'{metric}_0_to_{window}'].values
                met_wind_grating_control = cell_results.loc[(cell_results['Stimulus'] == 'Grating') & (cell_results['Treatment'] == 'Control'), f'{metric}_0_to_{window}'].values
                met_wind_grating_opto = cell_results.loc[(cell_results['Stimulus'] == 'Grating') & (cell_results['Treatment'] == 'Opto'), f'{metric}_0_to_{window}'].values
            
            
                met_data.append({'Cell': cell, 'Condition': 'CB', f'{metric}': met_wind_blank_control[0]})
                met_data.append({'Cell': cell, 'Condition': 'OB', f'{metric}': met_wind_blank_opto[0]})
                met_data.append({'Cell': cell, 'Condition': 'CG', f'{metric}': met_wind_grating_control[0]})
                met_data.append({'Cell': cell, 'Condition': 'OG', f'{metric}': met_wind_grating_opto[0]})
            
            
            # Create DataFrame for plotting
            mean_df = pd.DataFrame(met_data)
 
            # Calculate and print mean and standard deviation
            mean_stats = mean_df.groupby('Condition').agg({f'{metric}': ['mean', 'std']})
            print(mean_stats)
            
            # Ensure consistent order of conditions
            # condition_order = [ 'CB', 'OB','CG','OG']
            condition_order = [ 'CB','CG','OG']
            
            # condition_order = [ 'CB', 'OB']
            
            mean_df['Condition'] = pd.Categorical(mean_df['Condition'], categories=condition_order, ordered=True)
            
            # Set up the plot
            f,ax=plt.subplots(figsize=(50 / 25.4, 50 / 25.4))
            
            # Create the strip plot
            # sns.stripplot(x='Condition', y='Mean', data=mean_df, jitter=True, dodge=True, palette='Set1', alpha=0.7)
            
            # Create line plot connecting means
            mean_df_sorted = mean_df.sort_values(by=['Cell', 'Condition'])
            for cell in subset:
                cell_data = mean_df_sorted[mean_df_sorted['Cell'] == cell]
                sns.lineplot(x='Condition', y=f'{metric}', data=cell_data, marker='o', color='black', linewidth=0.5,ax=ax,markersize=4)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            # Add labels and title
            plt.xlabel('Condition')
            plt.ylabel(f'{metric}  Value')
            # ax.set_ylim([-0.1,0.2])
            plt.grid(False)  # Remove grid
            plt.title(f'Comparison of {metric} Values for {k} cells for the window 0 to {window}')
            # plt.savefig(fr'C:\Users\sp3660\Desktop\ChandPaper\Fig4\{k}_{window}_{metric}_comparison.svg')
            
            # Show the plot
            plt.show()


#%% PLOT TRIAL AVERAGED TRACSED FOR GROUPS OF CELLS FOR ALL CONDITIONS

def plot_single_cell_activity(ax, df, cell_id, stimulus, treatment):
    # Filter data for the given cell, stimulus, and treatment
    subset = df[(df['Cell'] == cell_id) & (df['Stimuli'] == stimulus) & (df['Treatment'] == treatment)]
    
    # Ensure we have trials and time points
    if subset.empty:
        return
    
    # Calculate trial-averaged activity
    trial_avg = subset.groupby('Time')['Value'].mean()
    trial_sem = subset.groupby('Time')['Value'].sem()  # Standard error of the mean
    
    # Plot individual trials in light grey
    for trial in subset['Trial'].unique():
        trial_data = subset[subset['Trial'] == trial]
        ax.plot(trial_data['Time'], trial_data['Value'], color='lightgrey', alpha=0.5,linewidth=0.5)

    # Plot the trial-averaged activity with SEM
    ax.plot(trial_avg.index, trial_avg.values, color='blue' if treatment == 'Opto' else 'black',linewidth=1)
    ax.fill_between(trial_avg.index, trial_avg - trial_sem, trial_avg + trial_sem, 
                    color='blue' if treatment == 'Opto' else 'black', alpha=0.2)
    
    # Add two vertical lines at time points 0 and 1
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.1)
    ax.axvline(x=1, color='black', linestyle='--', linewidth=0.1)
    
    # Remove all spines except for the bottom and left
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.margins(x=0)

    # ax.set_ylim([-10,40])



plt.close('all')
selected_cells=[visually_active_cells.iloc[0],both_active_cells.iloc[8]]
# Plot for each selected cell
for cell_id in selected_cells:

    # Create a figure with 2x2 subplots, with the required figure size
  
    # Plot for each condition in the subplots
    conditions = [
        ('Blank', 'Control'),
        ('Blank', 'Opto'),
        ('Grating', 'Control'),
        ('Grating', 'Opto')
    ]
    
    for stimulus, treatment in conditions:
        fig, ax = plt.subplots(1, figsize=(60 / 25.4, 30 / 25.4), sharex=True, sharey=True)

        plot_single_cell_activity(ax, df, cell_id, stimulus, treatment)
        plt.grid(False)  # Remove grid
        # Set tight layout and ensure figure size is appropriate
        plt.tight_layout()
        
        # Save and show the combined figure
        plt.savefig(str(fig_four_basepath / f'Cell_{cell_id}_{stimulus}_{treatment}.svg'), format='svg')
        plt.show()
#%% GROUP BY MOUSE
import statsmodels.api as sm
import scikit_posthocs as sp
plt.close('all')
# Define cell subsets
subset1_cells = visually_active_cells
  # Replace with actual list of cells
subset2_cells = both_active_cells

extended_results_df1, differences_df1 =extended_results_df1, differences_df1 
extended_results_df2, differences_df2 =extended_results_df2, differences_df2 
extended_results_df1['Group'] = extended_results_df1['Stimulus'] + ' ' + extended_results_df1['Treatment']
extended_results_df2['Group'] = extended_results_df2['Stimulus'] + ' ' + extended_results_df2['Treatment']
extended_results_df1_filtered = extended_results_df1[extended_results_df1['Group'] != 'Blank Opto']
extended_results_df2_filtered = extended_results_df2[extended_results_df2['Group'] != 'Blank Opto']



for cell_group in [extended_results_df1,extended_results_df2]:
    bc=cell_group[cell_group['Group']=='Blank Control']['Mean']
    gc=cell_group[cell_group['Group']=='Grating Control']['Mean']
    go=cell_group[cell_group['Group']=='Grating Opto']['Mean']
    means = {
        'Blank Control': (bc.mean(), bc.std()),
        'Grating Control': (gc.mean(), gc.std()),
        'Grating Opto': (go.mean(), go.std())
    }
    print("Means and Standard Deviations:")
    for condition, (mean, std) in means.items():
        print(f"{condition} - Mean: {mean:.3f}, Std: {std:.3f}")
    normality_results = {
        'Blank Control': stats.shapiro(bc),
        'Grating Control': stats.shapiro(gc),
        'Grating Opto': stats.shapiro(go)
    }
    
    print(
        f'Change in activity for gropued cells '
        f'for blank control {round(bc.mean(), 3):.3f} ± {round(bc.std(), 3):.3f} '
        f'and Grating Control {round(gc.mean(), 3):.3f} ± {round(gc.std(), 3):.3f}'
        f'and Grating Opto {round(go.mean(), 3):.3f} ± {round(go.std(), 3):.3f}'

    )
    
    
    
    for condition, result in normality_results.items():
        print(f"{condition} - W-statistic: {result[0]}, p-value: {result[1]}")
    
    # Homogeneity of variances
    levene_stat, levene_p = stats.levene(bc, gc, go)
    print(f"Levene's Test - Statistic: {levene_stat}, p-value: {levene_p}")
    
    # Step 2: Perform ANOVA or Kruskal-Wallis test based on assumptions
    if all(result[1] > 0.05 for result in normality_results.values()) and levene_p > 0.05:
        # Assumptions hold; perform ANOVA
        f_stat, p_value = stats.f_oneway(bc, gc, go)
        print(f"ANOVA F-statistic: {f_stat}, p-value: {p_value}")
    
        # Post-hoc analysis if significant
        if p_value < 0.05:
            combined_data = np.concatenate([bc, gc, go])
            labels = ['bc'] * len(bc) + ['gc'] * len(gc) + ['go'] * len(go)
            df_test = pd.DataFrame({'correlation': combined_data, 'condition': labels})
            
            tukey = sm.stats.multicomp.pairwise_tukeyhsd(df_test['correlation'], df_test['condition'])
            print(tukey)
    
    else:
        # Assumptions do not hold; perform Kruskal-Wallis test
        h_stat, p_value_kw = stats.kruskal(bc, gc, go)
        print(f"Kruskal-Wallis H-statistic: {h_stat}, p-value: {p_value_kw}")
        if p_value_kw < 0.05:
               combined_data = np.concatenate([bc, gc, go])
               labels = ['bc'] * len(bc) + ['gc'] * len(gc) + ['go'] * len(go)
               df_test = pd.DataFrame({'correlation': combined_data, 'condition': labels})
       
               # Perform Dunn's test
               dunn_result = sp.posthoc_dunn(df_test, val_col='correlation', group_col='condition', p_adjust='bonferroni')
               print(dunn_result)
    
    


#%%
def plot_mice(diff_df, color, title_suffix=''):
    ylimits=[-0.08, 0.18]
    sns.set_style("white")
    # Filter the differences DataFrame for the selected comparisons
    df_filtered = diff_df[~((diff_df['Stimulus'] == 'Blank') & (diff_df['Treatment'] == 'Opto'))]
    df_filtered['Comparison'] = df_filtered['Stimulus'] + '_' + df_filtered['Treatment']
    comparisons_of_interest = [
        'Blank_Control',
        'Grating_Control',
        'Grating_Opto'
    ]

    f,ax=plt.subplots(figsize=(1.97, 1.97))
    sns.boxplot(data=df_filtered, x='Comparison', y='Mean', fliersize=3,color=color,ax=ax)
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=True)
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_ylim(ylimits)
    sns.despine(ax=ax)
    # Set axis line width
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    plt.tight_layout()
    plt.savefig(str(fig_four_basepath / f'global_mice_recordings_{title_suffix}.svg'), format='svg')
    plt.show()

    # Plot 2: Lineplot of differences across sets for selected comparisons
    f,ax=plt.subplots(figsize=(1.97, 1.97))
    # Create a list to store the plot data
    plot_data = []
    for comparison in comparisons_of_interest:
        # Filter for the current comparison
        comparison_data = df_filtered[df_filtered['Comparison'] == comparison]
        
        # Append the data for plotting
        plot_data.append(comparison_data)

    # Combine all plot data into a single DataFrame for easy plotting
    plot_df = pd.concat(plot_data)
    # Create the lineplot
    sns.lineplot(data=plot_df, x='Comparison', y='Mean', hue='Set', marker='o', markersize=3,palette=[color]*len(plot_df['Set'].unique()), legend=None,ax=ax)
    ax.set_ylabel('')
    ax.set_xlabel('')
    sns.despine(ax=ax)

    # Set axis line width
    ax.spines['bottom'].set_linewidth(0.2)
    ax.spines['left'].set_linewidth(0.2)
    ax.set_ylim(ylimits)
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=True, right=False, labelbottom=False, labelleft=True)
    plt.tight_layout()
    # plt.savefig(str(fig_four_basepath / f'Lineplot_of_for_Each_Recording_Across_Comparisons_{title_suffix}.svg'), format='svg')
    plt.show()
    
    


plot_mice(extended_results_df1, color='grey', title_suffix='(Subset 1)')

# Example usage for subset2_cells (purple)
plot_mice(extended_results_df2, color='purple', title_suffix='(Subset 2)')

#%%

 # Set global parameters
plt.rcParams.update({
    'font.size': 5,  # Set font size to 5pt
    'axes.titlesize': 5,
    'axes.labelsize': 5,
    'xtick.labelsize': 5,
    'ytick.labelsize': 5,
    'lines.linewidth': 1,  # Set line thickness to 1 pt
    'figure.figsize': (1.97, 1.97)  # Set figure size to 50x50 mm
})

def plot_differences(diff_df, color, title_suffix=''):
    ylimits=[-0.08, 0.18]
    # Filter for the three comparisons involving Blank/Control
    comparisons_of_interest = [
        'Blank/Opto - Blank/Control',
        'Grating/Control - Blank/Control',
        'Grating/Opto - Blank/Control'
    ]
    
    # Filter the differences DataFrame for the selected comparisons
    diff_filtered = diff_df[diff_df['Comparison'].isin(comparisons_of_interest)]

    # Plot 1: Boxplot of differences for selected comparisons
    f,ax=plt.subplots(figsize=(1.97, 1.97))
    sns.boxplot(data=diff_filtered, x='Comparison', y='Difference', fliersize=3,color=color,ax=ax)
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=True)
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_ylim(ylimits)
    sns.despine(ax=ax)
    # Set axis line width
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    plt.tight_layout()
    plt.savefig(str(fig_four_basepath / f'Difference_control_recordings_{title_suffix}.svg'), format='svg')
    plt.show()
    
    # Plot 2: Lineplot of differences across sets for selected comparisons
    f,ax=plt.subplots(figsize=(1.97, 1.97))
    # Create a list to store the plot data
    plot_data = []
    for comparison in comparisons_of_interest:
        # Filter for the current comparison
        comparison_data = diff_df[diff_df['Comparison'] == comparison]
        
        # Append the data for plotting
        plot_data.append(comparison_data)
    
    # Combine all plot data into a single DataFrame for easy plotting
    plot_df = pd.concat(plot_data)
    # Create the lineplot
    sns.lineplot(data=plot_df, x='Comparison', y='Difference', hue='Set', marker='o', markersize=3,palette=[color]*len(plot_df['Set'].unique()), legend=None,ax=ax)
    ax.set_ylabel('')
    ax.set_xlabel('')
    sns.despine(ax=ax)

    # Set axis line width
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.set_ylim(ylimits)
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=True, right=False, labelbottom=False, labelleft=True)
    plt.tight_layout()
    # plt.savefig(str(fig_four_basepath / f'Lineplot_of_Differences_for_Each_Recording_Across_Comparisons_{title_suffix}.svg'), format='svg')
    plt.show()


# Example usage for subset1_cells (black)
plot_differences(differences_df1, color='grey', title_suffix='(Subset 1)')

# Example usage for subset2_cells (purple)
plot_differences(differences_df2, color='purple', title_suffix='(Subset 2)')

def prepare_comparison_data(df, subset_name, color):
    # Filter for the comparisons of interest
    comparisons_of_interest = [
        'Grating/Opto - Blank/Opto',
        'Grating/Opto - Grating/Control'
    ]
    
    # Filter the differences DataFrame for the selected comparisons
    diff_filtered = df[df['Comparison'].isin(comparisons_of_interest)]
    
    # Add a column to identify the subset
    diff_filtered['Subset'] = subset_name
    
    return diff_filtered

def plot_combined_boxplots(diff_df1, diff_df2):
    ylimits=[-0.08, 0.18]

    # Prepare data for both subsets
    diff_filtered1 = prepare_comparison_data(diff_df1, 'Subset 1', 'black')
    diff_filtered2 = prepare_comparison_data(diff_df2, 'Subset 2', 'purple')
    
    # Combine the data for both subsets
    combined_diff = pd.concat([diff_filtered1, diff_filtered2], ignore_index=True)
    
    # Plot: Boxplots of differences for all comparisons in a single plot
    f,ax=plt.subplots(figsize=(1.97, 1.97))
    sns.boxplot(data=combined_diff, x='Comparison', y='Difference', hue='Subset', palette={'Subset 1': 'grey', 'Subset 2': 'purple'},fliersize=3, legend=None,ax=ax)
    ax.set_ylabel('')
    ax.set_xlabel('')
    sns.despine(ax=ax)

    # Set axis line width
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.set_ylim(ylimits)
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=True, right=False, labelbottom=False, labelleft=True)
    plt.tight_layout()
    # plt.savefig(str(fig_four_basepath / 'Combined_Boxplots_Differences.svg'), format='svg')
    plt.show()

# Example usage
plot_combined_boxplots(differences_df1, differences_df2)

#%% GET TRHE RASTER AND TRY TO CALCULATE THE SUM OF ACTIVITIES AS TZI TZI

results_df['Mean_Difference']


even_cells = np.arange(0, df['Cell'].nunique(),2)  
odd_cells = np.arange(1, df['Cell'].nunique(),2)  



non_opto = results_df[ (results_df['Stimulus'] == 'Grating') & (results_df['Treatment'] == 'Control')][['Cell','Mean_Difference']]
non_opto_e = non_opto[non_opto['Cell'].isin(even_cells)]
non_opto_o = non_opto[non_opto['Cell'].isin(odd_cells)]

non_opto_e_vis_res=non_opto_e[non_opto_e['Cell'].isin(visually_active_cells)]
non_opto_o_vis_res=non_opto_o[non_opto_o['Cell'].isin(visually_active_cells)]



opto = results_df[ (results_df['Stimulus'] == 'Grating') & (results_df['Treatment'] == 'Opto')][['Cell','Mean_Difference']]
opto_e = opto[opto['Cell'].isin(even_cells)]
opto_o = opto[opto['Cell'].isin(odd_cells)]

opto_e_vis_res=opto_e[opto_e['Cell'].isin(visually_active_cells)]
opto_o_vis_res=opto_o[opto_o['Cell'].isin(visually_active_cells)]

