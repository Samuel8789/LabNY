# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 11:02:44 2021

@author: sp3660
"""

import tkinter as tk
from tkinter import ttk
from tkinter import END, Label, RAISED, Text, WORD, StringVar, Button



class AddImagingInfo(tk.Toplevel):
    def __init__(self,  gui, imaging_name):
        tk.Toplevel.__init__(self, gui) #inst     
        
        self.geometry("+2555+0")
        row_labels=['GreenFilter','RedFilter','DichroicBeamsplitter','IsBlockingDichroic','IsGoodObjective', 'ExcitationWavelength',
                    'CoherentPower', 'Comments', 'AtlasOverlap',' AtlasZStructure', 'AtlasSequenceMode', 'DoDeepCaiman?' ]
        self.total_rows=len(row_labels)+1
        self.values=list()
        self.b = list()
        self.b.append(list())  
        self.values.append(list())      
        col_labels=['Info',imaging_name]
    
        for j in range(len(col_labels)): #Columns
           self.b[0].append( Label( self, text =col_labels[j], relief=RAISED)) # b[i][j]
           self.b[0][j].grid(row=1, column=j)
           self.values[0].append(col_labels[j])
        for i in range(1,self.total_rows,1): #Rows
            self.b.append(list())  
            self.values.append(list())
            for j in range(len(col_labels)): #Columns
                if j==0:
                    texts=row_labels[i-1]
                    self.b[i].append(Label(self, text=texts, width=30)) # b[i][j]
                    self.b[i][j].grid(row=i+1, column=j)               
                    self.values[i].append(texts)  
                    
                elif j>0:
                    if i not in [1,2,3,4,5,11,12]:
                        self.b[i].append(Text(self, height=5, width=150, wrap=WORD)) # b[i][j]
                        self.b[i][j].grid(row=i+1, column=j)                              
                        self.values[i].append(self.b[i][j].get("1.0",END))  
                        if i==6:                       
                            self.b[i][j].insert("1.0",('920'))
                        elif i==7:
                            self.b[i][j].insert("1.0",('2000'))
                        else:
                            self.b[i][j].insert("1.0",('TO DO'))
                        
                    elif i==1:
                        values=['HQ525/70 M-2P 243899','BrightLine Fluorescent Filter 500/24','S510/20 XP 269755','D510/40 M 5633'] 
                        self.b[i].append(ttk.Combobox(self, values=values, width=30)) # b[i][j]
                        self.b[i][j].grid(row=i+1, column=j)   
                        self.b[i][j].current(1)     
                        self.values[i].append(self.b[i][j].get())   
                        
                    elif i==2:
                        values=['HQ607/45 M-2P 243869','BA570-620HQ F3F0024'] 
                        self.b[i].append(ttk.Combobox(self, values=values, width=30)) # b[i][j]
                        self.b[i][j].grid(row=i+1, column=j)   
                        self.b[i][j].current(0)     
                        self.values[i].append(self.b[i][j].get())   
                    elif i==3:
                        values=['Standard',] 
                        self.b[i].append(ttk.Combobox(self, values=values, width=30)) # b[i][j]
                        self.b[i][j].grid(row=i+1, column=j)   
                        self.b[i][j].current(0)     
                        self.values[i].append(self.b[i][j].get())   
                    elif i==4:
                        values=['Yes','No'] 
                        self.b[i].append(ttk.Combobox(self, values=values, width=30)) # b[i][j]
                        self.b[i][j].grid(row=i+1, column=j)   
                        self.b[i][j].current(1)     
                        self.values[i].append(self.b[i][j].get())  
                    elif i==5:
                        values=['Yes','No'] 
                        self.b[i].append(ttk.Combobox(self, values=values, width=30)) # b[i][j]
                        self.b[i][j].grid(row=i+1, column=j)   
                        self.b[i][j].current(0)     
                        self.values[i].append(self.b[i][j].get())  
                    elif i==11:
                        values=['Snake','Left Right'] 
                        self.b[i].append(ttk.Combobox(self, values=values, width=30)) # b[i][j]
                        self.b[i][j].grid(row=i+1, column=j)   
                        self.b[i][j].current(0)     
                        self.values[i].append(self.b[i][j].get())                      
                    elif i==12:
                        values=['Yes','No'] 
                        self.b[i].append(ttk.Combobox(self, values=values, width=30)) # b[i][j]
                        self.b[i][j].grid(row=i+1, column=j)   
                        self.b[i][j].current(1)     
                        self.values[i].append(self.b[i][j].get())  
                        
        enter_button = Button(self, text="Enter", command=self.retrieve_input)
        enter_button.grid(row=1,column=26)

    def retrieve_input(self):
        for i in range(1,self.total_rows,1): #Rows
                if i not in [1,2,3,4,5,11,12]:
                    self.values[i][1]=self.b[i][1].get("1.0",END)
                    while  self.values[i][1].endswith('\n'):
                        self.values[i][1]=self.values[i][1][:-1]
                elif i in [1,2,3,4,5,11,12]:
                    self.values[i][1]=self.b[i][1].get()
        self.destroy()
        self.update() 

if __name__ == "__main__":
    
    root = tk.Tk()
    # app = add_imaging_info(root, 'Test')
    root.mainloop()
    # get_values=app.values
	