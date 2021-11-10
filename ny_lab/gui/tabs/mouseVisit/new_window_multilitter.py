# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 12:19:32 2021

@author: sp3660
"""

from tkinter import StringVar, Button, LEFT
from tkinter.ttk import Combobox
import tkinter as tk

class new_window_multilitter(tk.Toplevel):
    
    def __init__(self, app,  values, title):
        tk.Toplevel.__init__(self, app)
        self.gui=app
   
        self.values=values
        self.title(title)  
        self.geometry("300x50")# Add a title
        var = StringVar()
        self.cb=Combobox(self ,values=self.values, textvariable=var)
        self.cb.pack(side=LEFT)
        self.b=Button(self,text='ok', command=self.check_cbox)
        self.b.pack()
        self.values=''
        
    def check_cbox(self):
        # for val in self.values:
        #     if self.cb.get() == val:
                self.values = self.cb.get() # this will assign the variable c the value of cbox
                self.destroy()

        # if cb.get() == values[1]:
        #     c = cb.get() 

    

if __name__ == "__main__":
    root = tk.Tk()
    app = new_window_multilitter(root, ['Test1', 'Test2'], 'test3')
    root.mainloop()
    get_values=app.values

    
       
      
     