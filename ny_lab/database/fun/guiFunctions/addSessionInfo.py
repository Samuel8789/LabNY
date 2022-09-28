# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 17:49:31 2021

@author: sp3660
"""
import tkinter as tk
from tkinter import END, Label, RAISED, Text, WORD, StringVar, Button, ttk




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

        col_labels=['Info',session_date.strftime('%Y%m%d')]
    
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
                    if i not in (1,2):
                        self.b[i].append(Text(self, height=5, width=150, wrap=WORD)) # b[i][j]
                        self.b[i][j].grid(row=i+1, column=j)                              
                        self.values[i].append(self.b[i][j].get("1.0",END) )
                        self.b[i][j].insert("1.0",('TO DO'))
                        
                    elif i==1:
                        values=['Prairie1',"Hakim's",'Prairie2'] 
                        self.b[i].append(ttk.Combobox(self, values=values)) # b[i][j]
                        self.b[i][j].grid(row=i+1, column=j)   
                        self.b[i][j].current(0)     
                        self.values[i].append(self.b[i][j].get())   
                    elif i==2:
                        self.b[i].append(Text(self, height=5, width=150, wrap=WORD)) # b[i][j]
                        self.b[i][j].grid(row=i+1, column=j)                              
                        self.values[i].append(self.b[i][j].get("1.0",END) )
                        self.b[i][j].insert("1.0",('00:00'))
                            

        enter_button = Button(self, text="Enter", command=self.retrieve_input)
        enter_button.grid(row=1,column=26)
        
    def retrieve_input(self):
        for i in range(1,self.total_rows,1): #Rows
            if i !=1:
                self.values[i][1]=self.b[i][1].get("1.0",END)
                while  self.values[i][1].endswith('\n'):
                    self.values[i][1]=self.values[i][1][:-1]
                    
            elif i==1:
                self.values[i][1]=self.b[i][1].get()
        self.destroy()            
        self.update()           
           
if __name__ == "__main__":
    
    root = tk.Tk()
    app = AddSessionInfo(root, '20210527')
    root.mainloop()
    get_values=app.values
