# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 10:58:51 2021

@author: sp3660
"""

import tkinter as tk
from tkinter import END, Label, RAISED, Text, WORD, Button

class AddWidefieldInfo(tk.Toplevel):
    def __init__(self,  gui, session_date, mouse_code):
        tk.Toplevel.__init__(self, gui) #i
        self.geometry("+2555+0")
        
        colname=session_date+'\\'+mouse_code
        row_labels=['WideFieldComments']
        self.total_rows=len(row_labels)+1
        self.values=list()
        self.b = list()
        self.b.append(list())  
        self.values.append(list())      
        col_labels=['Info',colname]
    
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
                    if i==1:
                        self.b[i].append(Text(self, height=5, width=150, wrap=WORD)) # b[i][j]
                        self.b[i][j].grid(row=i+1, column=j)                              
                        self.values[i].append(self.b[i][j].get("1.0",END))   
                        
    

        def retrieve_input():
            for i in range(1,self.total_rows,1): #Rows
                if i ==1:
                    self.values[i][1]=self.b[i][j].get("1.0",END)
                    while  self.values[i][1].endswith('\n'):
                        self.values[i][1]=self.values[i][1][:-1]

            self.destroy()
            self.update()           

            
            
        enter_button = Button(self, text="Enter", command=retrieve_input)
        enter_button.grid(row=1,column=26)

if __name__ == "__main__":
    
    root = tk.Tk()
    # app = add_widefield_info(root, '20210523','SPJO')
    root.mainloop()
    # get_values=app.values
