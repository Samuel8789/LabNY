# -*- coding: utf-8 -*-
"""
Created on Thu May 13 14:45:12 2021

@author: sp3660
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 13 08:47:09 2021

@author: sp3660
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 12 16:01:28 2021

@author: sp3660
"""
from tkinter import ttk
import tkinter as tk
from tkinter import  Label, RAISED, Button, Entry

class plan_window_parameters(tk.Toplevel):
    def __init__(self,  gui, mice_codes):
        tk.Toplevel.__init__(self, gui)

        self.selected_codes=mice_codes
        self.gui=gui
        self.total_rows=len(mice_codes)

        injection_params={'Mouse_Code':'',
                          'Date':'TODO',
                          'CorticalArea':1,
                          'HeadPlateCoordinates':6,
                          'WindowType':3,
                          'CranioSize':3,
                          'CoverType':2,
                          'CoverSize':3,
                          'Durotomy':1,
                          }
        

        self.values=list()
        self.b = list()
        self.b.append(list())  
        self.values.append(list())
        col_labels=list(injection_params.keys())
        self.col_labels=col_labels
        row_defaults=list(injection_params.values())
    
        query_brain_areas_info="""SELECT * FROM Brain_Areas_table"""    
        query_coordinates_info="""SELECT * FROM Sterocoordinates_table"""  
        query_coversize_info="""SELECT * FROM CoverSize_table"""  
        query_covertype_info="""SELECT * FROM Covertype_table"""  
        query_windowtype_info="""SELECT * FROM WindowTypes_table"""  
        query_windowsize_info="""SELECT * FROM Craniosize_table"""  
        params=()
        self.brain_areas=self.gui.MouseDat.arbitrary_query_to_df(query_brain_areas_info, params).set_index('ID').T.to_dict('list')
        self.coordinates=self.gui.MouseDat.arbitrary_query_to_df(query_coordinates_info, params).set_index('ID').T.to_dict('list')
        self.coversize=self.gui.MouseDat.arbitrary_query_to_df(query_coversize_info, params).set_index('ID').T.to_dict('list')
        self.covertype=self.gui.MouseDat.arbitrary_query_to_df(query_covertype_info, params).set_index('ID').T.to_dict('list')
        self.windowtype=self.gui.MouseDat.arbitrary_query_to_df(query_windowtype_info, params).set_index('ID').T.to_dict('list')
        self.windowsize=self.gui.MouseDat.arbitrary_query_to_df(query_windowsize_info, params).set_index('ID').T.to_dict('list')


        self.good_coordinates=[str(element)[1:-1] for element in self.coordinates.values() ]
        self.good_brain_areas=[item for sublist in  list(self.brain_areas.values()) for item in sublist]
        self.good_coversize=[item for sublist in  list(self.coversize.values()) for item in sublist]
        self.good_covertype=[item for sublist in  list(self.covertype.values()) for item in sublist]
        self.good_windowtype=[item for sublist in  list(self.windowtype.values()) for item in sublist]
        self.good_windowsize=[item for sublist in  list(self.windowsize.values()) for item in sublist]


        for j in range(len(col_labels)): #Columns
           self.b[0].append( Label( self, text =col_labels[j], relief=RAISED)) # b[i][j]
           self.b[0][j].grid(row=j, column=1, sticky='e')
           self.values[0].append(col_labels[j])
        for i in range(1,self.total_rows+1,1): #Rows
            self.b.append(list())  
            self.values.append(list())
            for j in range(len(col_labels)): #Columns
                if j==0:
                    self.b[i].append(Entry(self, text="", width=5)) # b[i][j]                   
                elif j==2:    
                    self.b[i].append(ttk.Combobox( self, values=self.good_brain_areas, width=15) ) # b[i][j]
                    self.b[i][j].current(0)
                elif j==3:    
                    self.b[i].append(ttk.Combobox( self, values=self.good_coordinates, width=25) ) # b[i][j]
                    self.b[i][j].current(5)
                elif j==4:    
                    self.b[i].append(ttk.Combobox( self, values=self.good_windowtype, width=20) ) # b[i][j]
                    self.b[i][j].current(2)
                elif j==5:    
                    self.b[i].append(ttk.Combobox( self, values=self.good_windowsize, width=10) ) # b[i][j]
                    self.b[i][j].current(2)
                elif j==6:    
                    self.b[i].append(ttk.Combobox( self, values=self.good_covertype, width=20) ) # b[i][j]
                    self.b[i][j].current(1)
                elif j==7:    
                    self.b[i].append(ttk.Combobox( self, values=self.good_coversize, width=15) ) # b[i][j]
                    self.b[i][j].current(2)
                else:
                    self.b[i].append(Entry(self, text="", width=5)) # b[i][j]     
                
                self.b[i][j].grid(row=j, column=i+1, sticky='w')
                
                if j==0:
                    self.b[i][j].insert(0, mice_codes[i-1])
                elif j not in [2,3,4,5,6,7]:
                    self.b[i][j].insert(0, row_defaults[j])
                

                self.values[i].append(self.b[i][j].get())   


        enter_button = Button(self, text="Enter", command=self.retrieve_input)
        enter_button.grid(row=1,column=26)
        
        
    
    def retrieve_input(self):
        for i in range(1,self.total_rows+1): #Rows
            for j in range(len(self.col_labels)): #Columns            
                self.values[i][j]=self.b[i][j].get()
                
                
        for i in range(1,len( self.values)):
            self.values[i][2]= self.good_brain_areas.index(self.values[i][2])+1
            self.values[i][3]= self.good_coordinates.index(self.values[i][3])+1
            self.values[i][4]= self.good_windowtype.index(self.values[i][4])+1
            self.values[i][5]= self.good_windowsize.index(self.values[i][5])+1
            self.values[i][6]= self.good_covertype.index(self.values[i][6])+1
            self.values[i][7]= self.good_coversize.index(self.values[i][7])+1

        self.destroy()
        self.update()
        self.gui.MouseDat.Experimental_class.plan_new_window(self.values, codes_selected=self.selected_codes)


if __name__ == "__main__":
    
    mice_info_list=[]
    root = tk.Tk()
    app = plan_window_parameters(root, mice_info_list)
    root.mainloop()
    get_values=app.values
