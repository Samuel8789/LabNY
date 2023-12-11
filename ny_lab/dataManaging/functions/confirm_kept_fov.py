# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 10:26:12 2023

@author: sp3660
"""
import tkinter as tk
from tkinter import messagebox

class confirm_kept_fov(tk.Toplevel):
    def __init__(self,  gui):
        tk.Toplevel.__init__(self, gui) #inst     
        self.keep = None

        self.title("FOV Name Confirmation")

        label = tk.Label(self, text="Do you want to keep the current FOV name?")
        label.pack(pady=10)

        yes_button = tk.Button(self, text="Yes", command=self.keep_current_name)
        yes_button.pack(side=tk.LEFT, padx=10)

        no_button = tk.Button(self, text="No", command=self.change_name)
        no_button.pack(side=tk.RIGHT, padx=10)

        
     
        
        
            
    def keep_current_name(self):
        self.keep= True
        messagebox.showinfo("Confirmation", "Current name will be kept.")
        self.destroy()
        self.update()   

    def change_name(self):
        self.keep = False
        messagebox.showinfo("Confirmation", "Name will be changed.")
        self.destroy()
        self.update()   
        
if __name__ == "__main__":
    
    root = tk.Tk()
    app = confirm_kept_fov(root)
    root.mainloop()
    get_values=app.keep