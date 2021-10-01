# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 17:49:31 2021

@author: sp3660
"""
import tkinter as tk
from tkinter import END, Label, RAISED, Text, WORD, StringVar, Button




class AddSessionInfo(tk.Toplevel):
    def __init__(self,  gui, session_date):
        tk.Toplevel.__init__(self, gui) #inst
        self.gui_ref=gui    
        self.geometry("+2555+0")
        row_labels=['Microscope','StartTime','Objectives','EndOfSessionSummary','IssuesDuringImaging']
        self.total_rows=len(row_labels)+1
        self.values=list()
        self.b = list()
        self.b.append(list())  
        self.values.append(list())     

        col_labels=['Info',session_date]
    
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
                    if i !=1:
                        self.b[i].append(Text(self, height=5, width=150, wrap=WORD)) # b[i][j]
                        self.b[i][j].grid(row=i+1, column=j)                              
                        self.values[i].append(self.b[i][j].get("1.0",END) )
                        
                    elif i==1:
                        values=['Prairie1',"Hakim's",'Prairie2'] 
                        self.b[i].append(tk.ttk.Combobox(self, values=values)) # b[i][j]
                        self.b[i][j].grid(row=i+1, column=j)   
                        self.b[i][j].current(0)     
                        self.values[i].append(self.b[i][j].get())   
    

        def retrieve_input():
            for i in range(1,self.total_rows,1): #Rows
                if i !=1:
                    self.values[i][1]=self.b[i][j].get("1.0",END)
                elif i==1:
                    self.values[i][1]=self.b[i][j].get()
            self.destroy()            
            self.update()           
            
 
        enter_button = Button(self, text="Enter", command=retrieve_input)
        enter_button.grid(row=1,column=26)

if __name__ == "__main__":
    
    root = tk.Tk()
    # app = add_session_info(root, '20210527', 6, MouseDat )
    root.mainloop()
    # get_values=app.values
