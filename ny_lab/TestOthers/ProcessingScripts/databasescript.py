# -*- coding: utf-8 -*-
"""
Created on Tue May 11 08:56:19 2021

@author: sp3660
"""
import sys
sys.path.insert(0, r'C:/Users/sp3660/Documents/Github/LabNY/AllFunctions')
sys.path.insert(0, r'C:/Users/sp3660/Documents/Github/LabNY/MainClasses/Mouse Managing')
from MouseDatabaseClass import MouseDatabase
import datetime

#%%
databasefile=r'C:\Users\sp3660\Documents\Projects\LabNY\4. Mouse Managing\MouseDatabase.db'
MouseDat=MouseDatabase(databasefile)
#%% Global Plotting
MouseDat.update_variables()

ExperimentalDat=MouseDat.Experimental_class
ImagingDat=MouseDat.ImagingDatabase_class

Database_attributes=list(vars(MouseDat).keys())
ExpDatabase_attributes=list(vars(MouseDat.Experimental_class).keys())
ImagingDatabase_attributes=list(vars(MouseDat.ImagingDatabase_class).keys())


colony_allmice=MouseDat.all_colony_mice
colony_stocks=MouseDat.stock_mice
colony_togenotype=MouseDat.mice_to_genotype
colony_breedings=MouseDat.breedings
colony_alllitters=MouseDat.all_litters
colony_currentlitter=MouseDat.get_litters()
colony_actions=MouseDat.actions

colony_gad=MouseDat.Gad
colony_Ai14=MouseDat.Ai14
colony_Ai65=MouseDat.Ai65
colony_Ai75=MouseDat.Ai75
colony_Ai80=MouseDat.Ai80
colony_slf=MouseDat.SLF
colony_pvf=MouseDat.PVF
colony_tigre=MouseDat.TIGRES

experimental_all_in_recovery=ExperimentalDat.all_exp_animals_recovery
experimental_planned=ExperimentalDat.all_mouse_planned_in_colony
experimental_todoinjection=ExperimentalDat.all_mouse_to_do_injection
experimental_toimage=ExperimentalDat.all_mouse_with_window
experimental_todowindows=ExperimentalDat.all_mouse_to_do_window
experimental_recoveryanimalsbetter=MouseDat.remove_unnecesary_virusinfo_from_df(MouseDat.remove_unnecesary_genes_from_df(ExperimentalDat.all_exp_animals_recovery))
experimental_interneuronimaging=MouseDat.remove_unnecesary_virusinfo_from_df(MouseDat.remove_unnecesary_genes_from_df(experimental_recoveryanimalsbetter[experimental_recoveryanimalsbetter['Projects']=='Interneuron_Imaging']))
experimental_interneuronopto=MouseDat.remove_unnecesary_virusinfo_from_df(MouseDat.remove_unnecesary_genes_from_df(experimental_recoveryanimalsbetter[experimental_recoveryanimalsbetter['Projects']=='Interneuron_Optogenetics']))
experimental_chandelierimaging=MouseDat.remove_unnecesary_virusinfo_from_df(MouseDat.remove_unnecesary_genes_from_df(experimental_recoveryanimalsbetter[experimental_recoveryanimalsbetter['Projects']=='Chandelier_Imaging']))
experimental_chandelieropto=MouseDat.remove_unnecesary_virusinfo_from_df(MouseDat.remove_unnecesary_genes_from_df(experimental_recoveryanimalsbetter[experimental_recoveryanimalsbetter['Projects']=='Chandelier_Optogenetics']))
experimental_tigres=MouseDat.remove_unnecesary_virusinfo_from_df(MouseDat.remove_unnecesary_genes_from_df(experimental_recoveryanimalsbetter[experimental_recoveryanimalsbetter['Projects']=='Tigre_Controls']))

allMICE=MouseDat.allMICE
allEXPERIMENTAL=MouseDat.allEXPERIMENTAL
allINJECTIONS=MouseDat.allINJECTIONS
allWINDOWS=MouseDat.allWINDOWS
allACTIONS=MouseDat.allACTIONS
allIMAGINGSESSIONS=MouseDat.allIMAGINGSESSIONS
allIMAGEDMICE= MouseDat.allIMAGEDMICE
allACQUISITIONS=MouseDat.allACQUISITIONS
allIMAGING=MouseDat.allIMAGING
allWIDEFIELD= MouseDat.allWIDEFIELD
allFACECAMERA= MouseDat.allFACECAMERA
allVISSTIMS= MouseDat.allVISSTIMS
experimental_todocarprofen=ExperimentalDat.all_mouse_to_do_postop_injection

imaging_all_sessions=ImagingDat.all_imaging_sessions
imaging_all_mice=ImagingDat.all_imaged_mice
imaging_interneuron_imaging=ImagingDat.interneuron_imaging_imaged_mice
imaging_chandelier_imaging=ImagingDat.chandelier_imaging_imaged_mice
imaging_interneuron_optogenetics=ImagingDat.interneuron_opto_imaged_mice
imaging_chandelier_optogenetics=ImagingDat.chandelier_opto_imaged_mice
imaging_tigre_controls=ImagingDat.tigres_imaged_mice
imaging_all_recordings=ImagingDat.all_acquisitions

database_structure={table[0]:[] for table in MouseDat.all_tables if table[0]!='sqlite_sequence'}
for key in database_structure.keys():
    query_build='select * from ' + key
    cursor = MouseDat.database_connection.execute(query_build)
    names = [description[0] for description in cursor.description]
    database_structure[key]=names

    

#%% standard quering
params=()
query_brains="""
SELECT*
FROM Behaviours_table
"""
zz=MouseDat.arbitrary_query_to_df(query_brains)
#%% query updating 219 220 222

query_mice_cage_update="""
                UPDATE Injections_table
                SET VirusCombination=29
                WHERE ID IN (174,175)
            """        
params=()   
MouseDat.arbitrary_updating_record(query_mice_cage_update, params, commit=True)

#%% inserting
import datetime

query_add_actions=""" 
INSERT INTO Actions_table(ID,
Date,
Action_Type,
Cage_start,
Cage_end,
Mouse_1,
Mouse_2,
Mouse_3
)
VALUES(?,date(?),?,?,?,?,?,?)
"""
params=(730,datetime.datetime.today(),2,307,307,2877,2879,2881)

MouseDat.arbitrary_inserting_record(query_add_actions, params, commit=True)

    
#%% removals
query_remove="""
DELETE FROM Injections_table
WHERE  ID IN (219,221)
"""
params=()
MouseDat.arbitrary_remove_record(query_remove, params, commit=True)


#%% adding ne columns

query_addColumn = """
            ALTER TABLE Acquisitions_table 
            ADD COLUMN  WorkingDiskPath
            
            """
params=()
MouseDat.arbitrary_add_column(query_addColumn, params, commit=True)
#%%
#renamingtable
query_rename= """
            ALTER TABLE CranectomyType_table
            RENAME TO WindowTypes_table;           
            """
params=()         
MouseDat.arbitrary_add_column(query_rename, params, commit=True)
          
            
#%%

MouseDat.close_database()
