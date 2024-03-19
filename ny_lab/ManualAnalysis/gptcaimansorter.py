#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 12:14:21 2024

@author: sp3660
"""

import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pathlib import Path
import os
class DataPanel(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)

        # First line
        tk.Label(self, text='Load data').grid(row=0, column=0, padx=5, pady=5)
        self.data_text = tk.Text(self, height=1, width=100, state=tk.DISABLED, wrap=tk.NONE)
        self.data_text.grid(row=0, column=1, padx=5, pady=5)
        tk.Button(self, text='Browse', command=self.browse).grid(row=0, column=2, padx=5, pady=5)
        tk.Button(self, text='Load', command=self.load_data).grid(row=0, column=3, padx=5, pady=5)
        tk.Button(self, text='Save data', command=self.save_data).grid(row=0, column=4, padx=5, pady=5)

        # Second line
        tk.Label(self, text='Log').grid(row=1, column=0, padx=5, pady=5)
        self.log_text = tk.Text(self, height=1, width=30, state=tk.DISABLED, wrap=tk.NONE)
        self.log_text.grid(row=1, column=1, padx=5, pady=5)
        tk.Button(self, text='Save ops', command=self.save_ops).grid(row=1, column=2, padx=5, pady=5)
    def browse(self):
        file_types = [
                    ("Mat files", "*.mat"),
                    ("h5", "*.hdf5"),
                ]
        file_path = filedialog.askopenfilename( initialdir ='/media/sp3660/Data Slow/Projects/LabNY/Full_Mice_Pre_Processed_Data/Mice_Projects/Chandelier_Imaging/VRC/PVF/SPRZ/imaging/20231107/data aquisitions/FOV_2/231107_SPRZ_FOV2_2z_ShortDrift_1xChand_opto_1.9_25x_920_51020_63075_with-000/planes/Plane1/Green/',
            title="Select a file",
            filetypes=file_types
            )
        if file_path:
            self.data_text.config(state=tk.NORMAL)
            self.data_text.delete(1.0, tk.END)
            self.data_text.insert(tk.END, file_path)
            self.data_text.config(state=tk.DISABLED)

    def load_data(self):
        # Implement your logic for loading data here
        self.log_message('Data loaded.')

    def save_data(self):
        # Implement your logic for saving data here
        self.log_message('Data saved.')

    def save_ops(self):
        # Implement your logic for saving operations here
        self.log_message('Operations saved.')

    def log_message(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + '\n')
        self.log_text.config(state=tk.DISABLED)
        
class ParamInfo(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.create_widgets()

    def create_widgets(self):
       # Label/Entry pair in the first and second columns
       cell_label = tk.Label(self, text="Cell")
       cell_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")

       cell_entry = tk.Entry(self, state="readonly")
       cell_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

       # Label in the third and fourth columns
       estimated_time_label = tk.Label(self, text="Estimated Time Constant AR1")
       estimated_time_label.grid(row=0, column=2, columnspan=2, padx=5, pady=5, sticky="e")

       # Entry/Label pair in the third row
       tau_decay_label = tk.Label(self, text="Tau Decay (sec)")
       tau_decay_label.grid(row=2, column=2, padx=5, pady=5, sticky="e")

       tau_decay_entry = tk.Entry(self, state="readonly")
       tau_decay_entry.grid(row=2, column=3, padx=5, pady=5, sticky="w")

       # Label in the fifth and sixth columns
       estimated_time_ar2_label = tk.Label(self, text="Estimated Time Constant AR2")
       estimated_time_ar2_label.grid(row=0, column=4, columnspan=2, padx=5, pady=5, sticky="e")

       # Entry/Label pair in the second row
       tau_rise_label = tk.Label(self, text="Tau Rise (sec)")
       tau_rise_label.grid(row=1, column=4, padx=5, pady=5, sticky="e")

       tau_rise_entry = tk.Entry(self, state="readonly")
       tau_rise_entry.grid(row=1, column=5, padx=5, pady=5, sticky="w")

       # Entry/Label pair in the third row
       tau_decay_ar2_label = tk.Label(self, text="Tau Decay (sec)")
       tau_decay_ar2_label.grid(row=2, column=4, padx=5, pady=5, sticky="e")

       tau_decay_ar2_entry = tk.Entry(self, state="readonly")
       tau_decay_ar2_entry.grid(row=2, column=5, padx=5, pady=5, sticky="w")

       # Label/Entry pair in the last two columns of the second row
       data_frame_rate_label = tk.Label(self, text="Data Frame Rate (fps)")
       data_frame_rate_label.grid(row=1, column=6, padx=5, pady=5, sticky="e")

       data_frame_rate_entry = tk.Entry(self, state="readonly")
       data_frame_rate_entry.grid(row=1, column=7, padx=5, pady=5, sticky="w")

class DeconvMethod(ttk.Notebook):
    def __init__(self, master=None):
        super().__init__(master)
        self.create_widgets()

    def create_widgets(self):
        tab1 = SmoothDFDT(self)
        tab2 = Constrained_foopsi(self)
        tab3 = MCMC(self)

        self.add(tab1, text="SmoothDFDT")
        self.add(tab2, text="Constrained_foopsi")
        self.add(tab3, text="MCMC")

class RunDeconv(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.create_widgets()

    def create_widgets(self):
        # Buttons in the first three rows
        run_current_cell_button = tk.Button(self, text="Run Current Cell")
        run_current_cell_button.grid(row=0, column=0, pady=5)

        run_accepted_cells_button = tk.Button(self, text="Run Accepted Cells")
        run_accepted_cells_button.grid(row=1, column=0, pady=5)

        run_all_cells_button = tk.Button(self, text="Run All Cells")
        run_all_cells_button.grid(row=2, column=0, pady=5)

        # Checkbox in the next row
        overwrite_checkbox = tk.Checkbutton(self, text="Overwrite")
        overwrite_checkbox.grid(row=3, column=0, pady=5)

        # Label in the next row
        plot_smooth_label = tk.Label(self, text="Plot Smooth dFoF")
        plot_smooth_label.grid(row=4, column=0, pady=5)

        # ToggleButton in the next row (custom class assumed to exist)
        toggle_button_smooth = ToggleButton(labels=['On', 'Off'], master=self)
        toggle_button_smooth.grid(row=5, column=0, pady=5)

        # Label in the next row
        plot_const_label = tk.Label(self, text="Plot Const Foopsi")
        plot_const_label.grid(row=6, column=0, pady=5)

        # ToggleButton in the next row (custom class assumed to exist)
        toggle_button_const_foopsi = ToggleButton(labels=['On', 'Off'], master=self)
        toggle_button_const_foopsi.grid(row=7, column=0, pady=5)

        # Label in the next row
        plot_mcmc_label = tk.Label(self, text="Plot MCMC")
        plot_mcmc_label.grid(row=8, column=0, pady=5)

        # ToggleButton in the next row (custom class assumed to exist)
        toggle_button_mcmc = ToggleButton(labels=['On', 'Off'], master=self)
        toggle_button_mcmc.grid(row=9, column=0, pady=5)

        # Button in the last row
        update_plots_button = tk.Button(self, text="Update Plots")
        update_plots_button.grid(row=10, column=0, pady=5)      
     
class SmoothDFDT(tk.Frame):   
    def __init__(self, master=None):
        super().__init__(master)
        self.create_widgets()

    def create_widgets(self):
        # Labels in the top row spanning two columns each
        signal_params_label = tk.Label(self, text="Signal Extraction Params", padx=5)
        signal_params_label.grid(row=0, column=0, columnspan=2, pady=5)

        plotting_params_label = tk.Label(self, text="Plotting Params", padx=5)
        plotting_params_label.grid(row=0, column=2, columnspan=2, pady=5)

        # Label in the first column, second row
        first_derivative_label = tk.Label(self, text="First Derivative")
        first_derivative_label.grid(row=1, column=0, pady=5)

        # Checkbox in the next row, spanning two columns
        convolve_checkbox = tk.Checkbutton(self, text="Convolve with Gaussian Kernel")
        convolve_checkbox.grid(row=2, column=0, columnspan=2, pady=5)

        # Entry and label pair in the fourth row
        gaussian_sigma_label = tk.Label(self, text="Gaussian Sigma (ms)")
        gaussian_sigma_label.grid(row=3, column=0, pady=5)

        gaussian_sigma_entry = tk.Entry(self)
        gaussian_sigma_entry.grid(row=3, column=1, pady=5)

        # Button in the next row
        plot_kernel_button = tk.Button(self, text="Plot Kernel")
        plot_kernel_button.grid(row=4, column=0, columnspan=2, pady=5)

        # Checkboxes in the next rows
        rectify_checkbox = tk.Checkbutton(self, text="Rectify")
        rectify_checkbox.grid(row=5, column=0, columnspan=2, pady=5)

        normalize_checkbox = tk.Checkbutton(self, text="Normalize")
        normalize_checkbox.grid(row=6, column=0, columnspan=2, pady=5)

        # Checkbox and entry label pair in the third column
        apply_threshold_checkbox = tk.Checkbutton(self, text="Apply Threshold")
        apply_threshold_checkbox.grid(row=2, column=2, pady=5)

        threshold_mag_label = tk.Label(self, text="Threshold Mag(z)")
        threshold_mag_label.grid(row=3, column=2, pady=5)

        threshold_mag_entry = tk.Entry(self)
        threshold_mag_entry.grid(row=3, column=3, pady=5)

        # Checkbox and entry label pair in the next rows
        plot_threshold_checkbox = tk.Checkbutton(self, text="Plot Threshold")
        plot_threshold_checkbox.grid(row=4, column=2, pady=5)

        scale_label = tk.Label(self, text="Scale")
        scale_label.grid(row=5, column=2, pady=5)

        scale_entry = tk.Entry(self)
        scale_entry.grid(row=5, column=3, pady=5)

        shift_label = tk.Label(self, text="Shift")
        shift_label.grid(row=6, column=2, pady=5)

        shift_entry = tk.Entry(self)
        shift_entry.grid(row=6, column=3, pady=5)
   
class Constrained_foopsi(tk.Frame):  
    def __init__(self, master=None):
        super().__init__(master)
        self.create_widgets()

    def create_widgets(self):
        # Labels in the first row
        deconv_params_label = tk.Label(self, text="Deconvolution Params", padx=5)
        deconv_params_label.grid(row=0, column=0, columnspan=2, pady=5)

        post_deconv_params_label = tk.Label(self, text="Post Deconvolution Params", padx=5)
        post_deconv_params_label.grid(row=0, column=2, columnspan=2, pady=5)

        # Labels, toggle button, and checkbox in the second row
        ar_model_label = tk.Label(self, text="AR Model")
        ar_model_label.grid(row=1, column=0, pady=5)

        toggle_button = ToggleButton(labels=['1', '2'], master=self)
        toggle_button.grid(row=1, column=1, pady=5)

        convolve_checkbox = tk.Checkbutton(self, text="Convolve with Gaussian Kernel")
        convolve_checkbox.grid(row=1, column=2, columnspan=2, pady=5)

        # Checkbox and entry label pair in the third row
        use_manual_tc_checkbox = tk.Checkbutton(self, text="Use Manual Time Constants")
        use_manual_tc_checkbox.grid(row=2, column=0, columnspan=2, pady=5)

        default_value_entry = tk.Entry(self, text="100")
        default_value_entry.grid(row=2, column=2, pady=5)

        gaussian_sigma_label = tk.Label(self, text="Gaussian Sigma (ms)")
        gaussian_sigma_label.grid(row=2, column=3, pady=5)

        # Entry label pairs in the next rows
        tau_rise_entry = tk.Entry(self)
        tau_rise_entry.grid(row=3, column=0, pady=5)

        tau_rise_label = tk.Label(self, text="Tau Rise (sec)")
        tau_rise_label.grid(row=3, column=1, pady=5)

        plot_kernel_button = tk.Button(self, text="Plot Kernel")
        plot_kernel_button.grid(row=3, column=2, columnspan=2, pady=5)

        tau_decay_entry = tk.Entry(self)
        tau_decay_entry.grid(row=4, column=0, pady=5)

        tau_decay_label = tk.Label(self, text="Tau Decay (sec)")
        tau_decay_label.grid(row=4, column=1, pady=5)

        # Labels in row 5
        plotting_params_label = tk.Label(self, text="Plotting Params", padx=5)
        plotting_params_label.grid(row=5, column=2, columnspan=2, pady=5)

        # Entry label pairs in rows 6 and 7
        scale_entry = tk.Entry(self)
        scale_entry.grid(row=6, column=2, pady=5)

        scale_label = tk.Label(self, text="Scale")
        scale_label.grid(row=6, column=3, pady=5)

        shift_entry = tk.Entry(self)
        shift_entry.grid(row=7, column=2, pady=5)

        shift_label = tk.Label(self, text="Shift")
        shift_label.grid(row=7, column=3, pady=5)

class MCMC(tk.Frame):    
    def __init__(self, master=None):
        super().__init__(master)
        self.create_widgets()

    def create_widgets(self):
        # Labels in the first row
        deconv_params_label = tk.Label(self, text="Deconvolution Params", padx=5)
        deconv_params_label.grid(row=0, column=0, columnspan=2, pady=5)
 
        post_deconv_params_label = tk.Label(self, text="Post Deconvolution Params", padx=5)
        post_deconv_params_label.grid(row=0, column=2, columnspan=2, pady=5)
 
        # Labels, toggle button, and checkbox in the second row
        ar_model_label = tk.Label(self, text="AR Model")
        ar_model_label.grid(row=1, column=0, pady=5)
 
        toggle_button = ToggleButton(labels=['1', '2'], master=self)
        toggle_button.grid(row=1, column=1, pady=5)
 
        convolve_checkbox = tk.Checkbutton(self, text="Convolve with Gaussian Kernel")
        convolve_checkbox.grid(row=1, column=2, columnspan=2, pady=5)
 
        # Checkbox and entry label pair in the third row
        use_manual_tc_checkbox = tk.Checkbutton(self, text="Use Manual Time Constants")
        use_manual_tc_checkbox.grid(row=2, column=0, columnspan=2, pady=5)
 
        default_value_entry = tk.Entry(self, text="100")
        default_value_entry.grid(row=2, column=2, pady=5)
 
        gaussian_sigma_label = tk.Label(self, text="Gaussian Sigma (ms)")
        gaussian_sigma_label.grid(row=2, column=3, pady=5)
 
        # Entry label pairs in the next rows
        tau_rise_entry = tk.Entry(self)
        tau_rise_entry.grid(row=3, column=0, pady=5)
 
        tau_rise_label = tk.Label(self, text="Tau Rise (sec)")
        tau_rise_label.grid(row=3, column=1, pady=5)
 
        plot_kernel_button = tk.Button(self, text="Plot Kernel")
        plot_kernel_button.grid(row=3, column=2, columnspan=2, pady=5)
 
        tau_decay_entry = tk.Entry(self)
        tau_decay_entry.grid(row=4, column=0, pady=5)
 
        tau_decay_label = tk.Label(self, text="Tau Decay (sec)")
        tau_decay_label.grid(row=4, column=1, pady=5)
 
        # Labels in row 5
        plotting_params_label = tk.Label(self, text="Plotting Params", padx=5)
        plotting_params_label.grid(row=5, column=2, columnspan=2, pady=5)
 
        # Entry label pairs in rows 6 and 7
        scale_entry = tk.Entry(self)
        scale_entry.grid(row=6, column=2, pady=5)
 
        scale_label = tk.Label(self, text="Scale")
        scale_label.grid(row=6, column=3, pady=5)
 
        shift_entry = tk.Entry(self)
        shift_entry.grid(row=7, column=2, pady=5)
 
        shift_label = tk.Label(self, text="Shift")
        shift_label.grid(row=7, column=3, pady=5)
        
       
        # Entry label pairs for B and N values
        b_samples_entry = tk.Entry(self)
        b_samples_entry.grid(row=5, column=0, pady=5)
        
        b_samples_label = tk.Label(self, text="B - Number of burn in samples (default 200)")
        b_samples_label.grid(row=5, column=1, pady=5)
        
        n_samples_entry = tk.Entry(self)
        n_samples_entry.grid(row=6, column=0, pady=5)
        
        n_samples_label = tk.Label(self, text="N - Number of samples after burn in (default 500)")
        n_samples_label.grid(row=6, column=1, pady=5)
        
        # Checkbox and entry label pair in the seventh row
        save_samples_checkbox = tk.Checkbutton(self, text="Save SAMPLES MCMC outputs")
        save_samples_checkbox.grid(row=7, column=0, columnspan=2, pady=5)
        
        # Inactive button in the eighth row
        plot_mcmc_button = tk.Button(self, text="Plot MCMC SAMPLES details", state=tk.DISABLED)
        plot_mcmc_button.grid(row=8, column=0, columnspan=4, pady=5)
   
class DeconvolutionTab(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.create_widgets()

    def create_widgets(self):
        # Main frame with two columns and two rows
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Creating sub-frames
        first_frame = ParamInfo(self)
        first_frame.grid(row=0, column=0, sticky="nw")

        second_frame = DeconvMethod(self)
        second_frame.grid(row=1, column=0, sticky="nsew")

        third_frame = RunDeconv(self)
        third_frame.grid(row=0, column=1, rowspan=2, sticky="e")

class BackgroundPanel(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.create_widgets()

    def create_widgets(self):
        # Top label
        label = tk.Label(self, text="Background Plot")
        label.pack(pady=5)

        # Separating line
        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=5)

        # Buttons
        self.components_button = tk.Button(self, text="Components", command=self.on_components_click, relief=tk.SUNKEN)
        self.components_button.pack(pady=5)
        self.weighted_components_button = tk.Button(self, text="Weighted Components", command=self.on_weighted_components_click, relief=tk.GROOVE)
        self.weighted_components_button.pack(pady=5)
        self.w_comp_bkg_button = tk.Button(self, text="W Comp + Bkg", command=self.on_w_comp_bkg_click, relief=tk.GROOVE)
        self.w_comp_bkg_button.pack(pady=5)


    def on_components_click(self):
        #updte buttons reliefs
        self.components_button.config(relief="sunken")
        self.weighted_components_button.config(relief="groove")
        self.w_comp_bkg_button.config(relief="groove")
        # select image to plot
        selected_image=plt.imread(Path.home() / "Desktop" / "ILTQq.png")

        # run method to plot image
        self.master.master.center_notebook_panel.tab1.update_images(selected_image)

    def on_weighted_components_click(self):
        #updte buttons reliefs
        self.components_button.config(relief="groove")
        self.weighted_components_button.config(relief="sunken")
        self.w_comp_bkg_button.config(relief="groove")
        # select image to plot
        selected_image=plt.imread(os.path.join(os.path.expanduser('~'),'Pictures/Spyder.png'))

        # run method to plot image
        self.master.master.center_notebook_panel.tab1.update_images(selected_image)

    def on_w_comp_bkg_click(self):
        #updte buttons reliefs
        self.components_button.config(relief="groove")
        self.weighted_components_button.config(relief="groove")
        self.w_comp_bkg_button.config(relief="sunken")
        # select image to plot
        selected_image=plt.imread(os.path.join(os.path.expanduser('~'),'Pictures/Spyder.png'))

        # run method to plot image
        self.master.master.center_notebook_panel.tab1.update_images(selected_image)
        
class SptialCompSorter(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.figs={'Accepted':{'fig':'',
                               'ax':'',
                               'canvas':''},
                   'Rejected':{'fig':'',
                               'ax':'',
                               'canvas':''},
                   }
        self.create_widgets()
      
        
    def create_widgets(self):
        plot_image=plt.imread(Path.home() / "Desktop" / "ILTQq.png")
        self.create_initial_figures("Accepted", "left", plot_image)
        self.create_initial_figures("Rejected", "right", plot_image)
        
        
    def create_initial_figures(self,title,side,imagetoplot)  : 
        self.figs[title]['fig'], self.figs[title]['ax'] = plt.subplots()
        self.figs[title]['ax'].imshow(imagetoplot)
        self.figs[title]['ax'].set_title('Accepted')
     
        self.figs[title]['canvas'] = FigureCanvasTkAgg( self.figs[title]['fig'], master=self)
        self.figs[title]['canvas'].get_tk_widget().pack(side=side, fill=tk.BOTH, expand=True)


    def update_images(self, new_image):
        
        for side,objects in  self.figs.items():
            objects['ax'].clear()
            objects['ax'].imshow(new_image)
            objects['canvas'].draw()

    
    

       

class NotebookPanel(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.create_widgets()

    def create_widgets(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab1 = SptialCompSorter(self)
        self.tab2 = DeconvolutionTab(self)

        self.notebook.add(self.tab1, text="Spatial Components")
        self.notebook.add(self.tab2, text="Deconvolution")
   
        
    def configure_tab1_layout(self):
        self.tab1.grid_columnconfigure(0, weight=1)
        self.tab1.grid_columnconfigure(1, weight=1)
        self.tab1.grid_rowconfigure(0, weight=1)
        self.tab1.grid_rowconfigure(1, weight=1)



class ToggleButton(tk.Button):
    def __init__(self, master=None,labels=['On', 'Off'],  **kwargs):
        # Convert all elements in labels to strings
        super().__init__(master, text=labels[1], relief="groove", command=self.toggle, **kwargs)
        self.labels = labels  # Corrected line
        self.state = False

    def toggle(self):
        self.state = not self.state
        if self.state:
            self.config(text=self.labels[0], relief="sunken")
        else:
            self.config(text=self.labels[1], relief="groove")

class ManualEditsPanel(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.toggle_state_var = tk.StringVar(value="Off")
        self.create_widgets()

    def create_widgets(self):
        # Label and button for manual edits
        label = tk.Label(self, text="Manual Edits")
        label.pack(pady=5)

        self.toggle_button = ToggleButton(labels=['On', 'Off'], master=self)
        self.toggle_button.pack()

        reset_button = tk.Button(self, text="Reset Manual Edits", command=self.reset_manual_edits, relief=tk.GROOVE)
        reset_button.pack()


    def reset_manual_edits(self):
        # Add logic for reset button click
        print("Reset Manual Edits button clicked")

class ContoursPanel(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.create_widgets()

    def create_widgets(self):
        # Label for contour limits
        label_limits = tk.Label(self, text="Contour Limits")
        label_limits.grid(row=0, column=0, columnspan=2, pady=5)

        # Labels for min and max
        min_label = tk.Label(self, text="Min")
        min_label.grid(row=1, column=0, pady=5)
        max_label = tk.Label(self, text="Max")
        max_label.grid(row=1, column=1, pady=5)

        # Entries for min and max
        min_entry = tk.Entry(self)
        min_entry.grid(row=2, column=0, pady=5)
        max_entry = tk.Entry(self)
        max_entry.grid(row=2, column=1, pady=5)

        # FigureCanvasTkAgg object
        fig, ax = plt.subplots()
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.get_tk_widget().grid(row=3, column=0, columnspan=2, pady=5)

class PlotContoursPanel(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.create_widgets()

    def create_widgets(self):
        # Top label
        label = tk.Label(self, text="Plot Contours")
        label.pack(pady=5)

        # Separating line
        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=5)

        # Buttons
        none_button = tk.Button(self, text="None", command=self.on_none_click, relief=tk.GROOVE)
        none_button.pack(pady=5)
        snr_caiman_button = tk.Button(self, text="SNR Caiman", command=self.on_snr_caiman_click, relief=tk.GROOVE)
        snr_caiman_button.pack(pady=5)
        snr2_button = tk.Button(self, text="SNR2", command=self.on_snr2_click, relief=tk.GROOVE)
        snr2_button.pack(pady=5)
        cnn_button = tk.Button(self, text="CNN", command=self.on_cnn_click, relief=tk.GROOVE)
        cnn_button.pack(pady=5)
        r_values_button = tk.Button(self, text="R Values", command=self.on_r_values_click, relief=tk.GROOVE)
        r_values_button.pack(pady=5)
        firing_stability_button = tk.Button(self, text="Firing Stability", command=self.on_firing_stability_click, relief=tk.GROOVE)
        firing_stability_button.pack(pady=5)

    def on_none_click(self):
        # Add logic for None button click
        print("None button clicked")

    def on_snr_caiman_click(self):
        # Add logic for SNR Caiman button click
        print("SNR Caiman button clicked")

    def on_snr2_click(self):
        # Add logic for SNR2 button click
        print("SNR2 button clicked")

    def on_cnn_click(self):
        # Add logic for CNN button click
        print("CNN button clicked")

    def on_r_values_click(self):
        # Add logic for R Values button click
        print("R Values button clicked")

    def on_firing_stability_click(self):
        # Add logic for Firing Stability button click
        print("Firing Stability button clicked")

class TopLeftPanel(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.create_widgets()

    def create_widgets(self):
        # Background panel
        background_panel = BackgroundPanel(self)
        background_panel.grid(row=0, column=0, sticky="nsew")

        # Manual Edits panel
        manual_edits_panel = ManualEditsPanel(self)
        manual_edits_panel.grid(row=1, column=0, sticky="nsew")

        # Plot Contours panel
        plot_contours_panel = PlotContoursPanel(self)
        plot_contours_panel.grid(row=2, column=0, sticky="nsew")

        # Contours panel
        contours_panel = ContoursPanel(self)
        contours_panel.grid(row=3, column=0, sticky="nsew")

        # Configuring row weights to allow resizing
        for i in range(4):
            self.grid_rowconfigure(i, weight=1)
            
        self.configure_highlight(background_panel)
        self.configure_highlight(manual_edits_panel)
        self.configure_highlight(plot_contours_panel)
        self.configure_highlight(contours_panel)
 
    def configure_highlight(self, widget):
        widget.config(highlightbackground="blue", highlightthickness=2)
        widget.bind("<FocusIn>", lambda event: widget.config(highlightbackground="red"))
        widget.bind("<FocusOut>", lambda event: widget.config(highlightbackground="blue"))
        
class TracePanel(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.create_widgets()

    def create_widgets(self):
        # FigureCanvasTkAgg object
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

class CustomWidget(tk.Frame):
    def __init__(self, label,master=None):
        super().__init__(master)
        self.label=label
        self.create_widgets()

    def create_widgets(self):
        label = tk.Label(self, text= self.label)
        label.grid(row=0, column=0, pady=5)

        self.entry_var = tk.StringVar(value="10")
        entry = tk.Entry(self, textvariable=self.entry_var, width=5)
        entry.grid(row=1, column=0, pady=5,rowspan=2)

        arrow_up_button = tk.Button(self, text="▲", command=self.increase_value)
        arrow_up_button.grid(row=1, column=1, padx=5, pady=(5, 0))

        arrow_down_button = tk.Button(self, text="▼", command=self.decrease_value)
        arrow_down_button.grid(row=2, column=1, padx=5, pady=(0, 5))

    def increase_value(self):
        current_value = int(self.entry_var.get())
        new_value = current_value + 1
        self.entry_var.set(str(new_value))

    def decrease_value(self):
        current_value = int(self.entry_var.get())
        new_value = max(1, current_value - 1)  # Ensure the value is not negative
        self.entry_var.set(str(new_value))

class SmoothingPanel(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.create_widgets()

    def create_widgets(self):
        smooth_raw_button = tk.Button(self, text="Smooth Raw", command=self.smooth_raw)
        smooth_raw_button.grid(row=0, column=0, rowspan=2, padx=5, pady=5)

        custom_widget = CustomWidget('window',self)
        custom_widget.grid(row=0, column=1, rowspan=2, padx=5, pady=5)

    def smooth_raw(self):
        # Add logic for smooth raw button click
        print("Smooth Raw button clicked")

class CaTracePanel(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.plot_c_var = tk.StringVar(value="Off")
        self.plot_raw_var = tk.StringVar(value="Off")
        self.create_widgets()

    def create_widgets(self):
        label_ca_trace = tk.Label(self, text="Ca trace")
        label_ca_trace.pack()

        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=5)

        label_plot_c = tk.Label(self, text="Plot C")
        label_plot_c.pack()

        self.plot_c_button = tk.Button(self, text="Off", textvariable=self.plot_c_var, command=self.plot_c_toggle)
        self.plot_c_button.pack()

        label_plot_raw = tk.Label(self, text="Plot Raw")
        label_plot_raw.pack()

        self.plot_raw_toggle_button = tk.Button(self, text="Off", textvariable=self.plot_raw_var,  command=self.plot_raw_toggle)
        self.plot_raw_toggle_button.pack()

    def plot_c_toggle(self):
        current_state = self.plot_c_var.get()
 
        if current_state == "Off":
            self.plot_c_var.set("On")
            # Change relief to sunken when state is "On"
            self.plot_c_button.config(relief=tk.SUNKEN)
        else:
            self.plot_c_var.set("Off")
            # Change relief to groove when state is "Off"
            self.plot_c_button.config(relief=tk.GROOVE)

    def plot_raw_toggle(self):
        current_state = self.plot_raw_var.get()

        if current_state == "Off":
            self.plot_raw_var.set("On")
            # Change relief to sunken when state is "On"
            self.plot_raw_toggle_button.config(relief=tk.SUNKEN)
        else:
            self.plot_raw_var.set("Off")
            # Change relief to groove when state is "Off"
            self.plot_raw_toggle_button.config(relief=tk.GROOVE)

class SpikesLastPanel(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.plot_spikes_var = tk.StringVar(value="Off")
        self.plot_lastc_var = tk.StringVar(value="Off")
        self.create_widgets()

    def create_widgets(self):
        label_ca_trace = tk.Label(self, text="Ca trace")
        label_ca_trace.pack()

        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=5)

        label_plot_c = tk.Label(self, text="Plot Spikes")
        label_plot_c.pack()

        self.plot_spikes_button = tk.Button(self, text="Off", textvariable=self.plot_spikes_var, command=self.plot_spikes_toggle)
        self.plot_spikes_button.pack()

        label_plot_raw = tk.Label(self, text="Plot Last C")
        label_plot_raw.pack()

        self.plot_lastc_button = tk.Button(self, text="Off", textvariable=self.plot_lastc_var,  command=self.plot_lastC_toggle)
        self.plot_lastc_button.pack()

    def plot_spikes_toggle(self):
        current_state = self.plot_spikes_var.get()
 
        if current_state == "Off":
            self.plot_spikes_var.set("On")
            # Change relief to sunken when state is "On"
            self.plot_spikes_button.config(relief=tk.SUNKEN)
        else:
            self.plot_spikes_var.set("Off")
            # Change relief to groove when state is "Off"
            self.plot_spikes_button.config(relief=tk.GROOVE)

    def plot_lastC_toggle(self):
        current_state = self.plot_lastc_var.get()

        if current_state == "Off":
            self.plot_lastc_var.set("On")
            # Change relief to sunken when state is "On"
            self.plot_lastc_button.config(relief=tk.SUNKEN)
        else:
            self.plot_lastc_var.set("Off")
            # Change relief to groove when state is "Off"
            self.plot_lastc_button.config(relief=tk.GROOVE)
                    
class LeftPanel2(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.create_widgets()

    def create_widgets(self):
        ca_trace_panel = CaTracePanel(self)
        ca_trace_panel.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        smoothing_panel = SmoothingPanel(self)
        smoothing_panel.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        spikelats_panel = SpikesLastPanel(self)
        spikelats_panel.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

        # Adjust row weights to allow resizing
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)
        
class CurrentComponent(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)

        self.create_plot()
        self.create_label_entries()

    def create_plot(self):
        fig, ax = plt.subplots()
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", columnspan=2)
        
    def create_label_entries(self):
        labels = [
            "SNR caiman",
            "SNR2",
            "CNN prob",
            "r_val",
            "firing stability",
            "peaks ave",
            "std noise"
        ]
    
        for i, label_text in enumerate(labels):
            label = tk.Label(self, text=label_text)
            label.grid(row=i+1, column=0, pady=5, sticky="e")
    
            entry = tk.Entry(self)
            entry.grid(row=i+1, column=1, pady=5, padx=5, sticky="w")
            
class MiniPanel(tk.Frame):
    def __init__(self,name, left_label,master=None):
        super().__init__(master)
        self.name=name
        self.left_label=left_label

        self.create_widgets()

    def create_widgets(self):
        snr_label = tk.Label(self, text=self.name)
        snr_label.grid(row=0, column=0, pady=5)

        # Separator
        ttk.Separator(self, orient="horizontal").grid(row=1, column=0, pady=5, sticky="ew")
       

        # Two CustomWidget below each label
        custom_widget_snr_thres = CustomWidget(self.left_label,self)
        custom_widget_snr_thres.grid(row=3, column=0, pady=5)

        custom_widget_lowest_thres = CustomWidget("Lowest Thres", self)
        custom_widget_lowest_thres.grid(row=3, column=1, pady=5)
        
class CaimanEvaluate(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.create_widgets()    
        
    def create_widgets(self):
  
        self.create_label_and_separator(self, "Caiman Evaluate")

        # Canvas widget (square)
        self.canvas = tk.Canvas(self, width=10, height=10, bd=0, highlightthickness=0)
        self.canvas.grid(row=1, column=0, pady=5, sticky="nsew")

        # Initial drawing
        self.canvas.create_rectangle(5, 5, 45, 45, fill="green", outline="")

        # New panel below square canvas
        snr_panel = MiniPanel('SNR',"SNR Thres", self)
        snr_panel.grid(row=2, column=0, pady=5, sticky="nsew")
        
        cnn_panel = MiniPanel('CNN prob  [0, 1]',"CNN prob thresh", self)
        cnn_panel.grid(row=3, column=0, pady=5, sticky="nsew")
        
        r_val_panel = MiniPanel('R values [0, 1]',"R valthresh", self)
        r_val_panel.grid(row=4, column=0, pady=5, sticky="nsew")
        
    def create_label_and_separator(self, parent, label_text):
        label = tk.Label(parent, text=label_text)
        label.grid(row=0, column=0, pady=5)

        ttk.Separator(parent, orient="horizontal").grid(row=1, column=0, pady=5, sticky="ew")
        
class RejectThrehold(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.create_widgets()    
        
    def create_widgets(self):
  
        self.create_label_and_separator(self, "Reject Thredhol")

        # Canvas widget (square)
        self.canvas = tk.Canvas(self, width=10, height=10, bd=0, highlightthickness=0)
        self.canvas.grid(row=1, column=0, pady=5, sticky="nsew")

        # Initial drawing
        self.canvas.create_rectangle(5, 5, 45, 45, fill="green", outline="")
        
        
        labels = ["SNR Caiman", "SNR2", "CNN Prob", "R Values", "Min Signal Frac", "Firing Stability"]
        for i, label_text in enumerate(labels):
            checkbox_var = tk.IntVar()
            checkbox = tk.Checkbutton(self, text=label_text, variable=checkbox_var)
            checkbox.grid(row=i + 2, column=0, sticky="w", padx=5, pady=5)

            entry = tk.Entry(self)
            entry.grid(row=i + 2, column=1, sticky="e", padx=5, pady=5)


        
    def create_label_and_separator(self, parent, label_text):
        label = tk.Label(parent, text=label_text)
        label.grid(row=0, column=0, pady=5)

        ttk.Separator(parent, orient="horizontal").grid(row=1, column=0, pady=5, sticky="ew")
        
class EvaluateParamsPanel(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.create_widgets()

    def create_widgets(self):
        # Top row
        self.eval_var=tk.StringVar(value="Reject")
        
        label_caiman_evaluate = tk.Label(self, text="Caiman Evaluate")
        label_caiman_evaluate.grid(row=0, column=0, pady=5)

        self.toggle_button = tk.Button(self, text="Toggle",textvariable=self.eval_var, command=self.on_toggle_click, relief=tk.GROOVE)
        self.toggle_button.grid(row=0, column=1, pady=5,columnspan=2)

        label_reject_threshold = tk.Label(self, text="Reject Threshold")
        label_reject_threshold.grid(row=0, column=3, pady=5)

        # Left panel: Caiman Evaluate
        self.caiman_evaluate_panel = CaimanEvaluate(self)
        self.caiman_evaluate_panel.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")

        # Right panel: Reject Threshold
        self.reject_threhold_panel = RejectThrehold(self)
        self.reject_threhold_panel.grid(row=1, column=2,columnspan=2, padx=5, pady=5, sticky="nsew")


        self.evaluate_button = tk.Button(self, text="Evaluate Components", command=self.evaluate, relief=tk.GROOVE)
        self.evaluate_button.grid(row=2, column=1, pady=5,columnspan=2)
        
    def evaluate(self):
        # Add logic for Firing Stability button click

        print('Evalauet components')
 
        
    def on_toggle_click(self):
        # Add logic for toggle button click
        if self.caiman_evaluate_panel.canvas.itemcget("all", "fill") == "green":
            self.reject_threhold_panel.canvas.itemconfig("all", fill="gray")
        else:
            self.caiman_evaluate_panel.canvas.itemconfig("all", fill="green")
            self.reject_threhold_panel.canvas.itemconfig("all", fill="gray")
            
    def on_toggle_click(self):
        current_state = self.eval_var.get()

        if current_state == "Reject":
            self.eval_var.set("Caiman")
            # Change relief to sunken when state is "On"
            self.toggle_button.config(relief=tk.SUNKEN)
            self.caiman_evaluate_panel.canvas.itemconfig("all", fill="green")
            self.reject_threhold_panel.canvas.itemconfig("all", fill="gray")
        else:
            self.eval_var.set("Reject")
            # Change relief to groove when state is "Off"
            self.toggle_button.config(relief=tk.GROOVE)
            self.caiman_evaluate_panel.canvas.itemconfig("all", fill="gray")
            self.reject_threhold_panel.canvas.itemconfig("all", fill="green")
            

    def create_label_and_separator(self, parent, label_text):
        label = tk.Label(parent, text=label_text)
        label.grid(row=0, column=0, pady=5)

        ttk.Separator(parent, orient="horizontal").grid(row=1, column=0, pady=5, sticky="ew")

class ParamsDistributionsTab(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.create_widgets()

    def create_widgets(self):
        # Matplotlib canvas for plotting
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Four buttons in a single column
        plot_snr_caiman_button = tk.Button(self, text="Plot SNR Caiman", command=self.plot_snr_caiman)
        plot_snr_caiman_button.grid(row=1, column=0, pady=5)

        plot_snr2_button = tk.Button(self, text="Plot SNR2", command=self.plot_snr2)
        plot_snr2_button.grid(row=2, column=0, pady=5)

        plot_cnn_button = tk.Button(self, text="Plot CNN", command=self.plot_cnn)
        plot_cnn_button.grid(row=3, column=0, pady=5)

        plot_rval_button = tk.Button(self, text="Plot Rval", command=self.plot_rval)
        plot_rval_button.grid(row=4, column=0, pady=5)

    def plot_snr_caiman(self):
        # Add logic for plotting SNR Caiman
        print("Plot SNR Caiman button clicked")

    def plot_snr2(self):
        # Add logic for plotting SNR2
        print("Plot SNR2 button clicked")

    def plot_cnn(self):
        # Add logic for plotting CNN
        print("Plot CNN button clicked")

    def plot_rval(self):
        # Add logic for plotting Rval
        print("Plot Rval button clicked")

class ParamsTab(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.create_widgets()

    def create_widgets(self):
        # Left column
        left_label_acid_params = tk.Label(self, text="OnACID Parameters")
        left_label_acid_params.grid(row=0, column=0, columnspan=2, pady=5)

        acid_params_labels = [
            "Dims", "Decay Time", "Frame Rate", "gSig", "Epochs", "p (AR model)", "ds_factor"
        ]

        for i, label_text in enumerate(acid_params_labels):
            label = tk.Label(self, text=label_text)
            label.grid(row=i+1, column=0, pady=5, sticky="e")

            entry = tk.Entry(self)
            entry.grid(row=i+1, column=1, pady=5, padx=5, sticky="w")

        # Right column
        right_label_processing_params = tk.Label(self, text="Processing Parameters")
        right_label_processing_params.grid(row=0, column=2, columnspan=2, pady=5)

        processing_params_labels = [
            "Peaks to Ave", "Peak Bin Size"
        ]

        for i, label_text in enumerate(processing_params_labels):
            label = tk.Label(self, text=label_text)
            label.grid(row=i+1, column=2, pady=5, sticky="e")

            entry = tk.Entry(self)
            entry.grid(row=i+1, column=3, pady=5, padx=5, sticky="w")

class RightPanel1(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.create_widgets()

    def create_widgets(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # First Tab
        tab1 = CurrentComponent(self.notebook)
        self.notebook.add(tab1, text="Current Component")

        # Second Tab: Evaluate Params
        tab2 = EvaluateParamsPanel(self.notebook)
        self.notebook.add(tab2, text="Evaluate Params")

        # Third Tab: Params Distributions
        tab3 = ParamsDistributionsTab(self.notebook)
        self.notebook.add(tab3, text="Params Distributions")
        
        # Fourth Tab: Params
        tab4 = ParamsTab(self.notebook)
        self.notebook.add(tab4, text="Params")
        
class RightPanel2(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master, bg='lightyellow', width=100, height=50)
        self.create_widgets()

    def create_widgets(self):
        # Label and separator
        cell_selection_label = tk.Label(self, text="Cell selection")
        cell_selection_label.grid(row=0, column=0, pady=5)

        ttk.Separator(self, orient="horizontal").grid(row=1, column=0, pady=5, sticky="ew")

        # Left pane
        left_pane = tk.Frame(self)
        left_pane.grid(row=2, column=0, padx=5, pady=5)

        cell_number_label = tk.Label(left_pane, text="Cell number")
        cell_number_label.grid(row=0, column=0, pady=5)

        # Replace "YourCustomWidget" with your actual custom widget code
        custom_widget = CustomWidget('Cell NUmber',self)
        custom_widget.grid(row=2, column=1, pady=5)

        change_cell_button = tk.Button(left_pane, text="Press up/down key to change cell num", command=self.change_cell)
        change_cell_button.grid(row=2, column=0, pady=5)

        # Right pane
        right_pane = tk.Frame(self)
        right_pane.grid(row=2, column=2, padx=5, pady=5)

        cell_category_label = tk.Label(right_pane, text="Cell category")
        cell_category_label.grid(row=0, column=0, pady=5)

        ttk.Separator(right_pane, orient="horizontal").grid(row=1, column=0, pady=5, sticky="ew")

        accepted_button = tk.Button(right_pane, text="Accepted", command=self.select_accepted_cells)
        accepted_button.grid(row=2, column=0, pady=5)

        rejected_button = tk.Button(right_pane, text="Rejected", command=self.select_rejected_cells)
        rejected_button.grid(row=3, column=0, pady=5)

        all_button = tk.Button(right_pane, text="All", command=self.select_all_cells)
        all_button.grid(row=4, column=0, pady=5)
        
    def change_cell(self):
        # Implement the logic for changing the cell number here
        pass

    def select_accepted_cells(self):
        # Implement the logic for selecting accepted cells here
        pass

    def select_rejected_cells(self):
        # Implement the logic for selecting rejected cells here
        pass

    def select_all_cells(self):
        # Implement the logic for selecting all cells here
        pass



class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Caiman Sorter")
        self.options={}
        

        # Creating and placing the DataPanel at the top, spanning all three columns
        self.data_panel = DataPanel(self)
        self.data_panel.grid(row=0, column=0, columnspan=3, padx=5, pady=5)

        # Creating and placing panels in the left column
        self.left_panel1 = TopLeftPanel(self)
        self.left_panel1.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        self.left_panel2 = LeftPanel2(self)
        self.left_panel2.grid(row=2, column=0, rowspan=2, padx=10, pady=10, sticky="nsew")

        # Creating and placing panels in the center column
        self.center_notebook_panel = NotebookPanel(self)
        self.center_notebook_panel.grid(row=1, column=1, rowspan=2, padx=10, pady=10, sticky="nsew")
        self.center_notebook_panel.configure_tab1_layout()
        
       # Adding TracePanel just below the center_notebook_panel
        self.trace_panel = TracePanel(self)
        self.trace_panel.grid(row=3, column=1, padx=10, pady=10, sticky="nsew")


        # Creating and placing panels in the right column
        self.right_panel1 = RightPanel1(self)
        self.right_panel1.grid(row=1, column=2, rowspan=2, padx=10, pady=10, sticky="nsew")

        self.right_panel2 = RightPanel2(self)
        self.right_panel2.grid(row=3, column=2, padx=10, pady=10, sticky="nsew")

        # Configuring row and column weights to allow resizing
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=2)
        self.grid_rowconfigure(3, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)

if __name__ == "__main__":
    app = MainApp()
    app.mainloop()