# -*- coding: utf-8 -*-
"""
Created on Fri May 14 14:22:18 2021

@author: sp3660
"""

import tkinter as Tkinter
import datetime
from tkinter import ttk
from tkinter import  Label, RAISED, Button, Entry, Text, WORD, END


class update_window_params(Tkinter.Toplevel):
    def __init__(self,  gui, cage, codes_selected, date_performed, database_object=None):
        Tkinter.Toplevel.__init__(self, gui)
        self.database_object=database_object
        self.date_performed=date_performed
        self.selected_codes=codes_selected
        self.cage=cage
        self.gui=gui
        query_mice_exp_info="""
        SELECT 
            ExperimentalAnimals_table.ID,
            MICE_table.ID, 
            Lab_number,Code,
            Labels_types,
            Room,
            ExperimentalAnimals_table.Experimental_status,
            MICE_table.Experimental_status, 
            Experiment,
            round(round(julianday('now') - julianday(Injection1Date))/7) AS WeeksFromInjection,
            WindowID ,
            Windows_table.*   
        FROM ExperimentalAnimals_table 
        LEFT JOIN MICE_table ON MICE_table.ID = ExperimentalAnimals_table.Mouse_ID 
        LEFT JOIN Windows_table ON Windows_table.ID = ExperimentalAnimals_table.WindowID 
        LEFT JOIN Labels_table d ON d.ID=Label
        WHERE Cage=? AND ExperimentalAnimals_table.Experimental_status=1 AND ExperimentalAnimals_table.Experiment=4 AND Code IN (%s)""" % ','.join('?' for i in codes_selected)  
        query_mice_exp_info="""
        SELECT 
            ExperimentalAnimals_table.ID,
            MICE_table.ID, 
            Lab_number,Code,
            Label,
            Room,
            ExperimentalAnimals_table.Experimental_status,
            MICE_table.Experimental_status, 
            Experiment,
            round(round(julianday('now') - julianday(Injection1Date))/7) AS WeeksFromInjection,
            WindowID ,
            Windows_table.*   
        FROM ExperimentalAnimals_table 
        LEFT JOIN MICE_table ON MICE_table.ID = ExperimentalAnimals_table.Mouse_ID 
        LEFT JOIN Windows_table ON Windows_table.ID = ExperimentalAnimals_table.WindowID 
        LEFT JOIN Labels_table d ON d.ID=Label
        WHERE Cage=? AND Code IN (%s)""" % ','.join('?' for i in codes_selected)            
        params=(cage,)+tuple(codes_selected)
        mice_exp = database_object.arbitrary_query_to_df(query_mice_exp_info, params).values.tolist()
               
        """
        to add comboboxes for:
            Labels
            CorticalArea
            HeadPlateCoordinates
            WindowType
            CranioSize
            CoverType
            CoverSize
        """
        mixe_exp_extended={}
        self.total_rows=len(codes_selected)
        for exp in mice_exp:
            window_params={'Mouse_Code':exp[3],
                              'Date':date_performed,
                              'Labels':exp[4],
                              'CorticalArea':exp[14],
                              'HeadPlateCoordinates':exp[15],
                              'WindowType':exp[16],
                              'CranioSize':exp[17],
                              'CoverType':exp[18],
                              'CoverSize':exp[19],
                              'Durotomy':exp[20],
                              'DamagedAreas':'',
                              'Notes':'',                     
                              }
            corrected_window_params={}
            for key,value in window_params.items():
                if value==None:
                    corrected_window_params[key]=''
                else:
                    corrected_window_params[key]=window_params[key]

            mixe_exp_extended[str(exp[3])]=corrected_window_params
        
        
        query_brain_areas_info="""SELECT * FROM Brain_Areas_table"""    
        query_coordinates_info="""SELECT * FROM Sterocoordinates_table"""  
        query_coversize_info="""SELECT * FROM CoverSize_table"""  
        query_covertype_info="""SELECT * FROM Covertype_table"""  
        query_windowtype_info="""SELECT * FROM WindowTypes_table"""  
        query_windowsize_info="""SELECT * FROM Craniosize_table"""
        query_labels_info="""SELECT * FROM Labels_table"""  

        params=()
        self.brain_areas=self.gui.MouseDat.arbitrary_query_to_df(query_brain_areas_info, params).set_index('ID').T.to_dict('list')
        self.coordinates=self.gui.MouseDat.arbitrary_query_to_df(query_coordinates_info, params).set_index('ID').T.to_dict('list')
        self.coversize=self.gui.MouseDat.arbitrary_query_to_df(query_coversize_info, params).set_index('ID').T.to_dict('list')
        self.covertype=self.gui.MouseDat.arbitrary_query_to_df(query_covertype_info, params).set_index('ID').T.to_dict('list')
        self.windowtype=self.gui.MouseDat.arbitrary_query_to_df(query_windowtype_info, params).set_index('ID').T.to_dict('list')
        self.windowsize=self.gui.MouseDat.arbitrary_query_to_df(query_windowsize_info, params).set_index('ID').T.to_dict('list')
        self.labels=self.gui.MouseDat.arbitrary_query_to_df(query_labels_info, params).set_index('ID').T.to_dict('list')
        self.good_labels=[item for sublist in  list(self.labels.values()) for item in sublist]

        self.good_coordinates=[str(element)[1:-1] for element in self.coordinates.values() ]
        self.good_brain_areas=[item for sublist in  list(self.brain_areas.values()) for item in sublist]
        self.good_coversize=[item for sublist in  list(self.coversize.values()) for item in sublist]
        self.good_covertype=[item for sublist in  list(self.covertype.values()) for item in sublist]
        self.good_windowtype=[item for sublist in  list(self.windowtype.values()) for item in sublist]
        self.good_windowsize=[item for sublist in  list(self.windowsize.values()) for item in sublist]
        
        
        self.values=list()
        self.b = list()
        self.b.append(list())  
        self.values.append(list())
        col_labels=list(mixe_exp_extended[list(mixe_exp_extended.keys())[0]].keys())
        self.col_labels=col_labels
        
        for j in range(len(col_labels)): #Columns
           self.b[0].append( Label( self, text =col_labels[j], relief=RAISED)) # b[i][j]
           self.b[0][j].grid(row=j, column=1)
           self.values[0].append(col_labels[j])
        # pair_counter=0   
        for i in range(1,self.total_rows+1,1): #Rows
            self.b.append(list())  
            self.values.append(list())
            row_defaults=list(mixe_exp_extended[list(mixe_exp_extended.keys())[i-1]].values())
            for j in range(len(col_labels)): #Columns
                if j in [1]:
                    self.b[i].append(Entry(self, text="", width=10)) # b[i][j]
                elif j==2:    
                    self.b[i].append(ttk.Combobox( self, values=self.good_labels, width=10) ) # b[i][j]
                    self.b[i][j].current(mice_exp[i-1][4]-1)
                elif j==3:    
                    self.b[i].append(ttk.Combobox( self, values=self.good_brain_areas, width=20) ) # b[i][j]
                    self.b[i][j].current(mice_exp[i-1][14]-1)
                elif j==4:    
                    self.b[i].append(ttk.Combobox( self, values=self.good_coordinates, width=30) ) # b[i][j]
                    self.b[i][j].current(mice_exp[i-1][15]-1)
                elif j==5:    
                    self.b[i].append(ttk.Combobox( self, values=self.good_windowtype, width=25) ) # b[i][j]
                    self.b[i][j].current(mice_exp[i-1][16]-1)
                elif j==6:    
                    self.b[i].append(ttk.Combobox( self, values=self.good_windowsize, width=15) ) # b[i][j]
                    self.b[i][j].current(mice_exp[i-1][17]-1)
                elif j==7:    
                    self.b[i].append(ttk.Combobox( self, values=self.good_covertype, width=25) ) # b[i][j]
                    self.b[i][j].current(mice_exp[i-1][18]-1)
                elif j==8:    
                    self.b[i].append(ttk.Combobox( self, values=self.good_coversize, width=20) ) # b[i][j]
                    self.b[i][j].current(mice_exp[i-1][19]-1)
                elif j in [10,11]:
                    self.b[i].append(Text(self, height=3, width=25, wrap=WORD)) # b[i][j]
                else:
                    self.b[i].append(Entry(self, text="", width=8)) # b[i][j]
                    

                self.b[i][j].grid(row=j, column=i+1)
                
                if j not in [2,3,4,5,6,7,8,10,11]:
                    self.b[i][j].insert(0, row_defaults[j])
                
                if j in [10,11]:
                     self.values[i].append(self.b[i][j].get("1.0",END))
                else :             
                     self.values[i].append(self.b[i][j].get())  

        enter_button = Button(self, text="Enter", command=self.retrieve_input)
        enter_button.grid(row=1,column=4)
        
    def retrieve_input(self):
        for i in range(1,self.total_rows+1): #Rows
            for j in range(len(self.col_labels)): #Columns                            
                if j in [10,11]:
                    self.values[i][j]=self.b[i][j].get("1.0",END)
                else :             
                    self.values[i][j]=self.b[i][j].get()
                
        for i in range(1,len( self.values)):
            self.values[i][2]= self.good_labels.index(self.values[i][2])+1
            self.values[i][3]= self.good_brain_areas.index(self.values[i][3])+1
            self.values[i][4]= self.good_coordinates.index(self.values[i][4])+1
            self.values[i][5]= self.good_windowtype.index(self.values[i][5])+1
            self.values[i][6]= self.good_windowsize.index(self.values[i][6])+1
            self.values[i][7]= self.good_covertype.index(self.values[i][7])+1   
            self.values[i][8]= self.good_coversize.index(self.values[i][8])+1        
   
        self.destroy()
        self.update()
        self.database_object.Experimental_class.update_performed_window(self.values, self.cage, self.selected_codes, self.date_performed )
if __name__ == "__main__":
    cage=305
    selectedanimals=['SPJA','SPJC']
    date_performed=datetime.datetime(2021, 5, 25, 0, 0)
    root = Tkinter.Tk()
    # app = update_window_params(root,cage, selectedanimals, date_performed, database_object=MouseDat)
    root.mainloop()
    # get_values=app.values
