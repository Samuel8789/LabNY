# -*- coding: utf-8 -*-
"""
Created on Tue May 11 15:25:41 2021

@author: sp3660
"""

import tkinter as tk
from tkinter import Label, RAISED, Button

class input_genotypes(tk.Toplevel):
    def __init__(self, gui, mouse_codes, rows, genes):
        tk.Toplevel.__init__(self, gui)

        self.values=list()
        self.b = list()
        self.b.append(list())  
        self.values.append(list())
        col_labels=['Mouse']+genes
        for j in range(len(col_labels)): #Columns
           self.b[0].append( Label( self, text =col_labels[j], relief=RAISED)) # b[i][j]
           self.b[0][j].grid(row=0, column=j)
           self.values[0].append(col_labels[j])
           
        for i in range(1,rows,1): #Rows
            self.b.append(list())  
            self.values.append(list())
            for j in range(len(col_labels)): #Columns
                if j==0:
                    texts=mouse_codes[i-1]
                    self.b[i].append(Label(self, text=texts)) # b[i][j]
                    self.b[i][j].grid(row=i+1, column=j)               
                    self.values[i].append(texts)  
                elif j>0:
                    self.b[i].append(tk.ttk.Combobox(self, width=5, state='readonly'))
                    self.b[i][j]['values'] = ('+/-' , '+/+' ,'-/-')
                    self.b[i][j].current(0)
                    self.b[i][j].grid(row=i+1, column=j)
                    self.values[i].append(self.b[i][j].get())


        def retrieve_input():
            for i in range(1,rows,1): #Rows
                for j in range(1,len(col_labels)): #Columns            
                    self.values[i][j]=self.b[i][j].get()
            self.destroy()
            
            
            
        enter_button = Button(self, text="Enter", command=retrieve_input)
        enter_button.grid(row=1,column=5)

if __name__ == "__main__":
    root = tk.Tk()
    # app = input_genotypes(root, app_rows, genes_to_genotype)
    root.mainloop()
    # get_values=app.values
