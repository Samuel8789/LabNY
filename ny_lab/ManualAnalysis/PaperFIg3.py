# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 15:49:24 2024

@author: sp3660
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy import stats
from matplotlib.colors import Normalize
import os
from sys import platform
from pathlib import Path

if platform == "linux" or platform == "linux2":
    fig_three_basepath=Path(r'/home/samuel/Dropbox/Projects/LabNY/ChandPaper/Fig3')
elif platform == "win32":
    fig_three_basepath=Path(r'C:\Users\sp3660\Desktop\ChandPaper\Fig3')
    

timestr=time.strftime("%Y%m%d-%H%M%S")


# Figure 3 Pnael B
# Mult traces of 8 chandelier from same FOV
trial_activity=cell_act_full
trial_averaged_activity=cell_act
xwindow
sorting_peaks=[np.flip(np.argsort(cell_mean_peaks['opto_blank'])),np.flip(np.argsort(cell_mean_peaks['control_blank'])),np.flip(np.argsort(cell_mean_peaks['opto_grating'])),np.flip(np.argsort(cell_mean_peaks['control_grating']))]

optoactivity=trial_activity['opto_blank'][sorting_peaks[0],:,:]/max(trial_activity['opto_blank'].max(),trial_activity['control_blank'].max())*100
controlactivity=trial_activity['control_blank'][sorting_peaks[1],:,:]/max(trial_activity['opto_blank'].max(),trial_activity['control_blank'].max())*100
optoactivity=trial_activity['opto_blank'][sorting_peaks[0],:,:]
controlactivity=trial_activity['control_blank'][sorting_peaks[1],:,:]

#%% FIgure matplotlib basics
mpl.rcParams['font.family'] = 'Arial'
plt.rcParams['svg.fonttype'] = 'none'


#%% REORGANIZE DATA IN DATAFRAME
# Select 8 cells to plot


# Create a DataFrame for plotting
data_list = []

for cell in range(optoactivity.shape[0]):
    for trial in range(optoactivity.shape[1]):
        for frame in range(optoactivity.shape[2]):
            data_list.append([cell, trial, xwindow[frame], optoactivity[cell, trial, frame], 'Opto'])
            data_list.append([cell, trial, xwindow[frame], controlactivity[cell, trial, frame], 'Control'])

df = pd.DataFrame(data_list, columns=['Cell', 'Trial', 'Time', 'Value', 'Treatment'])

#%% SINGLE CELL OPTO EFFECT
# Filter the DataFrame for cell 0
cell_data = df[df['Cell'] == 1]

# Set up the matplotlib figure size (convert from millimeters to inches)
fig, ax = plt.subplots(figsize=(60/25.4, 50/25.4))  # 30x25 mm to inches

# Plot individual trials for Control activity with reduced line width
sns.lineplot(data=cell_data[cell_data['Treatment'] == 'Control'], 
             x='Time', y='Value', 
             hue='Trial', 
             palette='gray', 
             legend=False,   # Turn off legend for this plot
             alpha=0.3, 
             linewidth=0.5, 
             ax=ax)

# Plot average trial for Control activity
control_avg = cell_data[cell_data['Treatment'] == 'Control'].groupby('Time')['Value'].mean().reset_index()
sns.lineplot(data=control_avg, 
             x='Time', y='Value', 
             color='black', 
             linewidth=2.5, 
             ax=ax, 
             legend=False)  # Turn off legend for this plot

# Plot individual trials for Optoactivity with reduced line width
sns.lineplot(data=cell_data[cell_data['Treatment'] == 'Opto'], 
             x='Time', y='Value', 
             hue='Trial', 
             palette='Blues', 
             legend=False,   # Turn off legend for this plot
             alpha=0.3, 
             linewidth=0.5, 
             ax=ax)

# Plot average trial for Optoactivity
opto_avg = cell_data[cell_data['Treatment'] == 'Opto'].groupby('Time')['Value'].mean().reset_index()
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

# Remove x-axis margins
ax.margins(x=0)

# Customize ticks and tick labels
ax.tick_params(axis='both', which='major', labelsize=6, width=1)

# Adjust layout
plt.tight_layout()

# Save the plot as an SVG file
plt.savefig(str(fig_three_basepath / f'activity_plot_cell_0_{timestr}.svg'), format='svg', bbox_inches='tight')

# Show the plot
plt.show()
#%%
# Filter the DataFrame for cells to plot
cells_to_plot = np.arange(8)

# Set up the matplotlib figure size (convert from millimeters to inches)
fig, ax = plt.subplots(figsize=(600/25.4, 500/25.4))  # Increase height for vertical stretch

# Define colors
control_color = 'black'
opto_color = 'blue'

# Set y-offset for each cell with large spacing
y_offsets = np.arange(len(cells_to_plot)) * 80  # Large spacing between cells

# Plot average activity for each cell
for cell, y_offset in zip(cells_to_plot, y_offsets):
    cell_data = df[df['Cell'] == cell]
    
    # Plot average trial for Control activity
    control_avg = cell_data[cell_data['Treatment'] == 'Control'].groupby('Time')['Value'].mean().reset_index()
    control_avg['Value'] += y_offset  # Apply y-offset
    sns.lineplot(data=control_avg, 
                 x='Time', y='Value',  # Plot with y-offset applied
                 color=control_color, 
                 linewidth=2.5, 
                 ax=ax, 
                 legend=False)
    
    # Plot average trial for Optoactivity
    opto_avg = cell_data[cell_data['Treatment'] == 'Opto'].groupby('Time')['Value'].mean().reset_index()
    opto_avg['Value'] += y_offset  # Apply y-offset
    sns.lineplot(data=opto_avg, 
                 x='Time', y='Value',  # Plot with y-offset applied
                 color=opto_color, 
                 linewidth=2.5, 
                 ax=ax, 
                 legend=False)
    
    # Add a horizontal line to separate plots
    ax.axhline(y=y_offset - 20, color='gray', linestyle='--', linewidth=1)  # Adjust y for horizontal line

# Set plot title and labels
# ax.set_title('Activity Plot for 8 Cells', fontsize=6)
ax.set_xlabel('Time (s)', fontsize=1)          # X-axis label with font size 6
ax.set_ylabel('df/f (%)', fontsize=1)          # Y-axis label with font size 6

# Add vertical dashed lines at time 0 and time 1 with black color
ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
ax.axvline(x=1, color='black', linestyle='--', linewidth=1.5)

# Remove the top and right spines
sns.despine(ax=ax)

# Set axis line width
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)

# Customize ticks and tick labels
ax.tick_params(axis='both', which='major', labelsize=6, width=1)

# Set y-axis limits to fit all cells with extra padding for stretching
ax.set_ylim(-10, y_offsets[-1] + 80)  # Adjust for extra padding


# Adjust layout
plt.tight_layout()

# Save the plot as an SVG file
plt.savefig(str(fig_three_basepath / f'activity_plot_8_cells_{timestr}_.svg'), format='svg', bbox_inches='tight')

# Show the plot
plt.show()

#%%


# Assuming df is your DataFrame

# Step 1: Filter for time range between 0 and 1 second
filtered_df = df[(df['Time'] >= 0) & (df['Time'] <= 1)]

# Step 2: Calculate the max value during the time range for each cell and treatment
max_values = (
    filtered_df.groupby(['Cell', 'Treatment'])['Value']
    .max()
    .reset_index()
    .rename(columns={'Value': 'MaxValue'})
)

# Perform a statistical test to compare means of Control vs. Opto
control_max_values = max_values[max_values['Treatment'] == 'Control']['MaxValue']
opto_max_values = max_values[max_values['Treatment'] == 'Opto']['MaxValue']
t_stat, p_value = stats.ttest_ind(control_max_values, opto_max_values)

print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

# Step 3: Create a box plot comparing max values of Control vs. Opto
fig, ax = plt.subplots(figsize=(25/25.4, 60/25.4))  # Convert mm to inches

# Box plot with transparency and grey outlines
sns.boxplot(data=max_values, x='Treatment', y='MaxValue', 
            palette={'Control': 'lightgrey', 'Opto': 'lightgrey'},
            linewidth=1.5, 
            boxprops=dict(facecolor='none', edgecolor='grey', alpha=0.5), 
            whiskerprops=dict(color='grey', linewidth=1.5),
            capprops=dict(color='grey', linewidth=1.5),
            medianprops=dict(color='black', linewidth=1.5),
            ax=ax)

# Add distribution of max points as a swarm plot with smaller points
# sns.swarmplot(data=max_values, x='Treatment', y='MaxValue', color='k', alpha=0.5, size=2, ax=ax)

# Remove all titles and labels
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_title('')

# Remove spines
sns.despine(ax=ax)

# Set axis line width
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)

# Customize ticks and tick labels
ax.tick_params(axis='both', which='major', labelsize=6, width=1)

# Save the plot as an SVG file
plt.tight_layout()
plt.savefig(str(fig_three_basepath / f'max_value_comparison_boxplot_transparent_no_labels_{timestr}_.svg'), format='svg', bbox_inches='tight')

# Show the plot
plt.show()

#%% NON CHANDELIER CELL RASTER

xwindow=all_exp_data[k][-1][0][0]['sliced_time_vector']
labels=np.arange(-aq_all_info['pre_time_df']/1000,aq_all_info['post_time_df']/1000+0.5,0.5)
opto_cross_sorted
optoactivity=opto_cross_sorted[0]/max(opto_cross_sorted[0].max(),opto_cross_sorted[1].max())*100
controlactivity=opto_cross_sorted[1]/max(opto_cross_sorted[0].max(),opto_cross_sorted[1].max())*100
optoactivity=opto_cross_sorted[0]
controlactivity=opto_cross_sorted[1]

#%% REORGANIZE DATA IN DATAFRAME
# Select 8 cells to plot


# Create a DataFrame for plotting
data_list = []

for cell in range(optoactivity.shape[0]):
    for trial in range(optoactivity.shape[1]):
        for frame in range(optoactivity.shape[2]):
            data_list.append([cell, trial, xwindow[frame], optoactivity[cell, trial,frame], 'Opto'])
            data_list.append([cell, trial,  xwindow[frame], controlactivity[cell,trial, frame], 'Control'])

df = pd.DataFrame(data_list, columns=['Cell', 'Trial' ,'Time', 'Value', 'Treatment'])

#%% SVAE ACTIVITY DF FORT TZI ZTI RASTER
# Eliminate every other cell
filtered_cells = np.arange(0, df['Cell'].nunique(),2)  # Adjust according to your data
filtered_df = df[df['Cell'].isin(filtered_cells)]

from scipy.io import savemat

# Assuming your DataFrame is called `df`

# Step 1: Pivot the DataFrame to have Treatment as columns and Time as rows
pivot_df = filtered_df.pivot_table(index=['Cell', 'Time'], columns='Treatment', values='Value').reset_index()

# Step 2: Sort the values by 'Cell' and 'Time' to maintain the proper order
pivot_df = pivot_df.sort_values(by=['Cell', 'Time'])

# Step 3: Extract the NumPy array
# The shape will be (cells, time points, 2 treatments - 'Control' and 'Opto')
cells = pivot_df['Cell'].nunique()
time_points = pivot_df['Time'].nunique()

# Reshape into (cells, time, treatment) 
value_array = pivot_df[['Control', 'Opto']].values.reshape(cells, time_points, 2)

# Step 4: Save the array as a .mat file
savemat(str(fig_three_basepath / f'cell_time_treatment_data_no_baseline_substarct.mat', {'data': value_array})




#%%

plt.rcParams['svg.fonttype'] = 'none'
# Eliminate every other cell
filtered_cells = np.arange(0, df['Cell'].nunique(),2)  # Adjust according to your data
filtered_df = df[df['Cell'].isin(filtered_cells)]

avg_df = df.groupby(['Cell', 'Time', 'Treatment'])['Value'].mean().reset_index()

# Separate the data into 'Opto' and 'Control' for heatmap plotting
control_array = avg_df[avg_df['Treatment'] == 'Control'].pivot(index='Cell', columns='Time', values='Value').values
opto_array = avg_df[avg_df['Treatment'] == 'Opto'].pivot(index='Cell', columns='Time', values='Value').values


# Get unique time values for x-axis labels
time_values = filtered_df['Time'].unique()
# Convert mm to inches
height_inch = 80 / 25.4
width_inch = 40 / 25.4
# Adjust tick intervals for y-axis
def update_y_axis_ticks(ax, data_array, interval=10):
    num_rows = data_array.shape[0]
    y_ticks = np.arange(0, num_rows, interval)
    y_tick_labels = [str(cell) for cell in y_ticks]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels, fontsize=10)

# Increase contrast by adjusting normalization range
def adjust_contrast(vmin, vmax):
    # Example: expanding the range slightly to increase contrast
    return Normalize(vmin=vmin * 0.5, vmax=vmax * 0.8)

# Create control raster image
fig1, ax1 = plt.subplots(figsize=(width_inch, height_inch))
control_norm = adjust_contrast(np.min(avg_df['Value']), np.max(avg_df['Value']))
sns.heatmap(control_array, cmap='viridis', ax=ax1,
            norm=control_norm,
            linewidths=0, linecolor='None',rasterized=True)

# Define tick positions and labels
tick_interval = 1
rounded_time_points = np.round(xwindow, 2)
tick_positions = np.arange(xwindow[0], xwindow[-1] + tick_interval, tick_interval)
tick_indices = np.searchsorted(rounded_time_points, tick_positions)

ax1.set_xticks(tick_indices)  # Set tick positions
ax1.set_xticklabels(tick_positions, fontsize=2)  # Set tick labels

# Update y-axis ticks
update_y_axis_ticks(ax1, control_array)

# Add vertical lines at times close to 0 and 1
ax1.axvline(x=np.searchsorted(rounded_time_points, 0), color='white', linestyle='--', linewidth=1)
ax1.axvline(x=np.searchsorted(rounded_time_points, 1), color='white', linestyle='--', linewidth=1)

# Remove spines and adjust plot
sns.despine(ax=ax1, left=True, bottom=True)
ax1.tick_params(axis='both', which='both', labelsize=2, width=0.5)
ax1.set_xlabel('')
ax1.set_ylabel('')

plt.tight_layout()
# plt.savefig(str(fig_three_basepath / f'control_raster_image_{timestr}_.svg'), format='svg', bbox_inches='tight')
plt.show()

# Create opto raster image
fig2, ax2 = plt.subplots(figsize=(width_inch, height_inch))
opto_norm = adjust_contrast(np.min(avg_df['Value']), np.max(avg_df['Value']))
sns.heatmap(opto_array, cmap='viridis', ax=ax2,
            norm=opto_norm,
            linewidths=0,  # No lines between cells
            linecolor=None,rasterized=True)  # No line color
# Define tick positions and labels
ax2.set_xticks(tick_indices)  # Set tick positions
ax2.set_xticklabels(tick_positions, fontsize=2)  # Set tick labels

# Update y-axis ticks
update_y_axis_ticks(ax2, opto_array)

# Add vertical lines at times close to 0 and 1
ax2.axvline(x=np.searchsorted(rounded_time_points, 0), color='white', linestyle='--', linewidth=1)
ax2.axvline(x=np.searchsorted(rounded_time_points, 1), color='white', linestyle='--', linewidth=1)

# Remove spines and adjust plot
sns.despine(ax=ax2, left=True, bottom=True)
ax2.tick_params(axis='both', which='both', labelsize=2, width=0.5)
ax2.set_xlabel('')
ax2.set_ylabel('')

plt.tight_layout()
# plt.savefig(str(fig_three_basepath / f'opto_raster_image_{timestr}_.svg'), format='svg', bbox_inches='tight')
plt.show()

#%% SIGNIFICANTLY ACTIVATED AND INHIBITED CELLS
import pandas as pd
from scipy.stats import wilcoxon, ttest_ind
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['svg.fonttype'] = 'none'

# Assuming df, controlactivity, optoactivity, xwindow, etc. are already defined

# Filter the DataFrame to the window of 0 to 1 second
window_df = df[(df['Time'] >= 0) & (df['Time'] <= 1.5)]

# Initialize lists to store results
results = []
increased_cells = []
decreased_cells = []
unchanged_cells = []

# Number of cells to calculate Bonferroni corrected p-value
num_comparisons = window_df['Cell'].nunique()
correction_factor = num_comparisons
# Loop over each cell to perform the test
for cell in window_df['Cell'].unique():
    # Filter data for the current cell
    cell_data = window_df[window_df['Cell'] == cell]
    
    # Get mean values for 'Control' and 'Opto' treatments during the window
    control_values = cell_data[cell_data['Treatment'] == 'Control']['Value'].groupby(cell_data['Trial']).mean()
    opto_values = cell_data[cell_data['Treatment'] == 'Opto']['Value'].groupby(cell_data['Trial']).mean()
    
    # Perform Wilcoxon signed-rank test
    if len(control_values) > 0 and len(opto_values) > 0:  # Ensure there are values to compare
        stat, p_value = wilcoxon(control_values, opto_values)
    else:
        continue  # Skip this cell if there are no valid comparisons
    corrected_p_value = p_value * correction_factor

    # Determine if there's a significant increase or decrease
    if p_value < 0.01:  # Using corrected p-value
        control_mean = control_values.mean()
        opto_mean = opto_values.mean()
        if opto_mean > control_mean:
            increased_cells.append(cell)
        elif opto_mean < control_mean:
            decreased_cells.append(cell)
    else:
        unchanged_cells.append(cell)
    
    # Store the results
    results.append({
        'Cell': cell,
        'Control Mean': control_values.mean(),
        'Opto Mean': opto_values.mean(),
        'Statistic': stat,
        'P-Value': p_value
    })

# Create a DataFrame to display the results
results_df = pd.DataFrame(results)

# Display the results
print(results_df)

plt.rcParams['svg.fonttype'] = 'none'
# Convert mm to inches
figsize_inches = (100 / 25.4, 100 / 25.4)
plt.close('all')

# Plot cells with significant increases
if increased_cells:
    increased_df = avg_df[avg_df['Cell'].isin(increased_cells)]

    plt.figure(figsize=figsize_inches)
    # Plot individual cell traces
    for cell in increased_cells:
        cell_data = increased_df[increased_df['Cell'] == cell]
        sns.lineplot(data=cell_data, x='Time', y='Value', hue='Treatment', 
                     palette={'Control': 'gray', 'Opto': 'blue'}, alpha=0.3, legend=False)
    
    # Calculate and plot mean with SEM
    for treatment in ['Control', 'Opto']:
        treatment_data = increased_df[increased_df['Treatment'] == treatment]
        mean_values = treatment_data.groupby('Time')['Value'].mean()
        sem_values = treatment_data.groupby('Time')['Value'].sem()
        
        plt.plot(mean_values.index, mean_values.values, label=f'{treatment} Mean',
                 color='black' if treatment == 'Control' else 'blue', linewidth=1.5)
        plt.fill_between(mean_values.index, mean_values - sem_values, mean_values + sem_values,
                         color='gray' if treatment == 'Control' else 'blue', alpha=0.3)
    
    # Remove margins
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Add vertical lines
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    plt.axvline(x=1, color='black', linestyle='--', linewidth=0.5)
    
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    plt.xlabel('')
    plt.ylabel('')
    plt.margins(x=0)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(0.5)
    plt.gca().spines['bottom'].set_linewidth(0.5)
    plt.grid(False)  # Remove grid
    plt.tight_layout()
    plt.savefig(str(fig_three_basepath /  f'Cells_with_Significant_Increase_{timestr}.svg'), format='svg')
    plt.show()

# Plot cells with significant decreases
if decreased_cells:
    decreased_df = avg_df[avg_df['Cell'].isin(decreased_cells)]
    plt.figure(figsize=figsize_inches)
    # Plot individual cell traces
    for cell in decreased_cells:
        cell_data = decreased_df[decreased_df['Cell'] == cell]
        sns.lineplot(data=cell_data, x='Time', y='Value', hue='Treatment', 
                     palette={'Control': 'gray', 'Opto': 'blue'}, alpha=0.3, legend=False)
    
    # Calculate and plot mean with SEM
    for treatment in ['Control', 'Opto']:
        treatment_data = decreased_df[decreased_df['Treatment'] == treatment]
        mean_values = treatment_data.groupby('Time')['Value'].mean()
        sem_values = treatment_data.groupby('Time')['Value'].sem()
        
        plt.plot(mean_values.index, mean_values.values, label=f'{treatment} Mean',
                 color='black' if treatment == 'Control' else 'blue', linewidth=1.5)
        plt.fill_between(mean_values.index, mean_values - sem_values, mean_values + sem_values,
                         color='gray' if treatment == 'Control' else 'blue', alpha=0.3)
    
    # Remove margins
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Add vertical lines
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    plt.axvline(x=1, color='black', linestyle='--', linewidth=0.5)
    plt.margins(x=0)

    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    plt.xlabel('')
    plt.ylabel('')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(0.5)
    plt.gca().spines['bottom'].set_linewidth(0.5)
    plt.grid(False)  # Remove grid
    plt.tight_layout()
    plt.savefig(str(fig_three_basepath /  f'Cells_with_Significant_Decrease_{timestr}.svg'), format='svg')
    plt.show()

# Plot pie chart with custom colors
plt.figure(figsize=figsize_inches)
proportions = [len(increased_cells), len(decreased_cells), len(unchanged_cells)]
labels = ['Increased', 'Decreased', 'Unchanged']
colors = ['yellow', 'cyan', 'gray']
plt.pie(proportions, labels=labels, autopct='%1.1f%%', colors=colors)
# plt.title('Proportion of Cells with Significant Changes', fontsize=5)
# plt.savefig(str(fig_three_basepath /  'PieChart_{timestr}.svg'), format='svg')
plt.show()


# Prepare data for plotting
mean_activities = pd.DataFrame({
    'Cell': [],
    'Condition': [],
    'Mean Activity': []
})

for cell in increased_cells + decreased_cells:
    cell_data = window_df[window_df['Cell'] == cell]
    mean_activity_control = cell_data[cell_data['Treatment'] == 'Control']['Value'].mean()
    mean_activity_opto = cell_data[cell_data['Treatment'] == 'Opto']['Value'].mean()
    
    mean_activities = mean_activities.append({
        'Cell': cell,
        'Condition': 'Control',
        'Mean Activity': mean_activity_control
    }, ignore_index=True)
    
    mean_activities = mean_activities.append({
        'Cell': cell,
        'Condition': 'Opto',
        'Mean Activity': mean_activity_opto
    }, ignore_index=True)

# Add labels for increased and decreased cells
mean_activities['Group'] = mean_activities['Cell'].apply(
    lambda x: 'Increased' if x in increased_cells else 'Decreased'
)


mean_activities['Group_ID'] = None  # Create an empty column for group IDs

# Group by 'Condition' and 'Group'
grouped = mean_activities.groupby(['Condition', 'Group'])

# For each combination of 'Condition' and 'Group', split the cells into 5 groups
for (condition, group), sub_mean_activities in grouped:
    num_cells = len(sub_mean_activities)
    group_labels = np.array_split(np.arange(num_cells), 5)  # Split into 5 groups
    for i, indices in enumerate(group_labels):
        mean_activities.loc[sub_mean_activities.index[indices], 'Group_ID'] = i + 1  # Assign group labels



# Plot boxplots with customizations
plt.figure(figsize=figsize_inches)

# Plot the boxplot with transparent boxes and gray outlines
sns.boxplot(data=mean_activities, x='Group', y='Mean Activity', hue='Condition', palette={'Control': 'gray', 'Opto': 'blue'},
            boxprops=dict(facecolor='none', edgecolor='gray', linewidth=0.5), linewidth=0.5)

# Add individual data points using stripplot with manual jitter
# ax = sns.stripplot(data=mean_activities, x='Group', y='Mean Activity', hue='Condition', palette={'Control': 'gray', 'Opto': 'blue'},
#                    dodge=True, jitter=True, size=2, edgecolor='none', linewidth=0.5)

plt.legend(fontsize=5)
# plt.title('Mean Activity of Cells with Significant Changes', fontsize=5)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.xlabel('')
plt.ylabel('')
plt.legend().remove()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(0.5)
plt.gca().spines['bottom'].set_linewidth(0.5)
plt.grid(False)  # Remove grid
plt.tight_layout()
plt.savefig(str(fig_three_basepath /  f'Mean_Activity_of_Cells_with_Significant_Changes_{timestr}.svg'), format='svg')
plt.show()

# Statistical comparison between increased and decreased cells
increased_activity = mean_activities[mean_activities['Group'] == 'Increased']
decreased_activity = mean_activities[mean_activities['Group'] == 'Decreased']

# Perform t-test for the Control condition
control_increased = increased_activity[increased_activity['Condition'] == 'Control']['Mean Activity']
control_decreased = decreased_activity[decreased_activity['Condition'] == 'Control']['Mean Activity']

# Perform t-test for the Opto condition
opto_increased = increased_activity[increased_activity['Condition'] == 'Opto']['Mean Activity']
opto_decreased = decreased_activity[decreased_activity['Condition'] == 'Opto']['Mean Activity']
stat, p_value = ttest_ind(control_increased, opto_increased)
print(f'Test Statistic (Stimulated): {stat}, P-Value: {p_value}')
stat, p_value = ttest_ind(control_decreased, opto_decreased)
print(f'Test Statistic (Inhibited): {stat}, P-Value: {p_value}')

# make tzi tzi lie plot

#%% Comparing experiment anfd ploting like tzi tzi
# Convert figure size from mm to inches
figsize_mm = (40, 40)
figsize_inch = (figsize_mm[0] / 25.4, figsize_mm[1] / 25.4)
group_means=group_means
# Pivot the dataframe to have 'Control' and 'Opto' as columns
pivot_df = group_means.pivot_table(index=['Group', 'Group_ID'], columns='Condition', values='Mean Activity').reset_index()

# Perform t-test and store the results
ttest_results = {}

# Plot for each group (Increased and Decreased)
for grp in pivot_df['Group'].unique():
    grp_data = pivot_df[pivot_df['Group'] == grp]  # Filter for the current group
    
    # Perform t-test comparing Control and Opto
    t_stat, p_value = stats.ttest_rel(grp_data['Control'], grp_data['Opto'])
    ttest_results[grp] = {'t_stat': t_stat, 'p_value': p_value}
    
    fig,ax=plt.subplots(figsize=figsize_inch)  # Set the figure size
    
    # Create a line plot for the current group
    for i, row in grp_data.iterrows():
        ax.plot(['Control', 'Opto'], [row['Control'], row['Opto']], marker='o', color='black', markersize=2,linewidth=0.2)
    
    # Adjust axis, font size, and line width
    # ax.xticks([0, 1], ['Control', 'Opto'], fontsize=5)
    # ax.yticks(fontsize=5)
    # plt.gca().tick_params(width=0.2)
    
    # Set axis labels and thickness
    # ax.xlabel('Condition', fontsize=5)
    # ax.ylabel('Mean Activity', fontsize=5)
    # Set axis line thickness
    for spine in plt.gca().spines.values():
        spine.set_linewidth(0.2)
    ''
    # Remove legend
    plt.legend().set_visible(False)
    ax.set_ylim([group_means['Mean Activity'].min()*1.5, group_means['Mean Activity'].max()*1.2])

    plt.savefig(str(fig_three_basepath /  f'Final_mean_activity_changed_group_{grp}_{timestr}.svg'), format='svg')

    plt.show()

# Output t-test results
print("T-test Results:")
for group, results in ttest_results.items():
    print(f"{group} Group: t-statistic = {results['t_stat']:.4f}, p-value = {results['p_value']:.4e}")




#%% CORRLATION BWTEN CHANDELIERS

#%%

# Function to compute circular permutation
def circular_permute(arr, max_shift):
    shift = np.random.randint(1, max_shift)
    return np.roll(arr, shift)
exps=exps
# Analyze each group
correlation_results = []
num_permutations = 100

for group_idx, cell_group in enumerate(exps):
    group_df = df[df['Cell'].isin(cell_group)]
    
    # Compute correlations trial by trial
    for trial in group_df['Trial'].unique():
        trial_data = group_df[group_df['Trial'] == trial]
        
        for treatment in ['Control', 'Opto']:
            treatment_data = trial_data[trial_data['Treatment'] == treatment]
            cell_values = treatment_data.pivot(index='Cell', columns='Time', values='Value').values
            
            if cell_values.shape[0] > 1:  # Ensure there is more than one cell to correlate
                corr_matrix = np.corrcoef(cell_values)
                mean_corr = np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
                correlation_results.append({'Group': group_idx, 'Trial': trial, 'Treatment': treatment, 'Type': 'Normal', 'Correlation': mean_corr})
                
                # Circular permutation test
                for _ in range(num_permutations):
                    shuffled_values = np.array([circular_permute(cell_trace, cell_values.shape[1]) for cell_trace in cell_values])
                    shuffled_corr = np.corrcoef(shuffled_values)
                    shuffled_mean_corr = np.mean(shuffled_corr[np.triu_indices_from(shuffled_corr, k=1)])
                    correlation_results.append({'Group': group_idx, 'Trial': trial, 'Treatment': treatment, 'Type': 'Shuffled', 'Correlation': shuffled_mean_corr})

correlation_df = pd.DataFrame(correlation_results)

# Calculate mean correlation per group and condition for visualization
mean_correlation_df = correlation_df.groupby(['Group', 'Treatment', 'Type'])['Correlation'].mean().reset_index()

# Plotting the comparison of actual and shuffled correlations
figsize_inches = (40 / 25.4, 40 / 25.4)
plt.figure(figsize=figsize_inches)
sns.boxplot(data=mean_correlation_df, x='Treatment', y='Correlation', hue='Type',
            boxprops=dict(facecolor='none', edgecolor='gray', linewidth=0.5), linewidth=0.5)

sns.stripplot(data=mean_correlation_df, x='Treatment', y='Correlation', hue='Type',
              dodge=True, jitter=True, size=2, edgecolor='none', linewidth=0.5)

plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.xlabel('')
plt.ylabel('')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(0.5)
plt.gca().spines['bottom'].set_linewidth(0.5)
plt.grid(False)  # Remove grid
plt.legend().remove()
plt.tight_layout()
plt.savefig(str(fig_three_basepath /  'Mean_chandelier_cell_correlations_per_recording_{timestr}.svg'), format='svg')

plt.show()


# Perform t-tests for statistical comparison
for treatment in ['Control', 'Opto']:
    normal_corr = mean_correlation_df[(mean_correlation_df['Type'] == 'Normal') & (mean_correlation_df['Treatment'] == treatment)]['Correlation']
    rolled_corr = mean_correlation_df[(mean_correlation_df['Type'] == 'Shuffled') & (mean_correlation_df['Treatment'] == treatment)]['Correlation']
    stat, p_value = ttest_ind(normal_corr, rolled_corr)
    print(f'Test Statistic ({treatment}): {stat}, P-Value: {p_value}')

# Statistical comparison between unshuffled Control and Opto
control_corr = mean_correlation_df[(mean_correlation_df['Type'] == 'Normal') & (mean_correlation_df['Treatment'] == 'Control')]['Correlation']
opto_corr = mean_correlation_df[(mean_correlation_df['Type'] == 'Normal') & (mean_correlation_df['Treatment'] == 'Opto')]['Correlation']
stat, p_value = ttest_ind(control_corr, opto_corr)
print(f'Test Statistic (Control vs. Opto - Unshuffled): {stat}, P-Value: {p_value}')