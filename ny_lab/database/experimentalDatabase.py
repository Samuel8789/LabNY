# -*- coding: utf-8 -*-
"""
Created on Mon May 17 08:01:22 2021

@author: sp3660
"""
import math 
import numpy as np
import datetime

from .fun.databaseCodesTransformations import transform_earlabels_to_codes, get_combination_from_virus, get_next_twolettercode

class ExperimentalDatabase():
    
    
    def __init__(self, database_object):
        
        self.databse_ref=database_object
        self.database_connection = self.databse_ref.database_connection
        self.to_do_postop_injection()
        self.update_variables()
    
    def update_variables(self):
   
       self.get_max_experimental_code()
       self.get_max_injection_code()
       self.get_max_window_code()
       self.get_max_brain_code()
       self.table_all_experimental_all_info()
       self.table_all_mouse_to_do_injection()
       self.table_all_animals_in_recovery()
       self.table_all_mouse_with_window()
       self.table_all_mouse_to_do_window()
       self.table_planned_but_in_colony()   
       self.table_virus_info()
       
    def get_max_experimental_code(self):
        c=self.database_connection.cursor()
        query_max_code="SELECT ID, Code FROM ExperimentalAnimals_table WHERE ID=(select MAX(ID) FROM ExperimentalAnimals_table)"
        c.execute(query_max_code)
        max_current_code=c.fetchall()
        self.max_current_experimenatl_code=max_current_code[0]
    
    def get_max_injection_code(self):
        c=self.database_connection.cursor()
        query_max_code="SELECT MAX(ID) FROM Injections_table"
        c.execute(query_max_code)
        max_current_code=c.fetchall()
        self.max_current_injection_code=max_current_code[0][0]
                
    def get_max_window_code(self):
        c=self.database_connection.cursor()
        query_max_code="SELECT MAX(ID) FROM Windows_table"
        c.execute(query_max_code)
        max_current_code=c.fetchall()
        self.max_current_window_code=max_current_code[0][0]
        
    def get_max_brain_code(self):
        c=self.database_connection.cursor()
        query_max_code="SELECT MAX(ID) FROM BrainProcessing_table"
        c.execute(query_max_code)
        max_current_code=c.fetchall()
        self.max_current_brain_code=max_current_code[0][0]    
        

#%% experimental animal queries


    def table_all_experimental_all_info(self):
        query_all_experimental_all_info="""
        SELECT 
            a.ID,
            Room,
            Cage, 
            Code,
            Lab_number, 
            Labels_types,      
            ExperimentalStatus_table.Status AS Experimental_status,
            Experimental_table.Experimental_types AS Experiment, 
            round(round(julianday('now') - julianday(DOB))/7) AS Age,
            round(round(julianday('now') - julianday(Injection1Date))/7) AS WeeksFromInjection,
            round(round(julianday('now') - julianday(WindowDate))/7) AS WeeksFromWindow,
            d.Projects,
            Line_short,
            Combination,
            k.Sensors AS Sensors1,
            l.Optos AS Optos1,
            m.Promoters AS Promoters1,
            n.Recombinases AS Recombinases1,
            o.Sensors AS Sensors2,
            p.Promoters AS Promoters2,
            q.Recombinases AS Recombinases2,
            r.Optos AS Optos3,
            s.Promoters AS Promoters3,
            t.Recombinases AS Recombinases3,
            G2C_table.Genotypes_types AS G2C,
            Ai14_table.Genotypes_types AS Ai14,
            Ai75_table.Genotypes_types AS Ai75,
            VRC_table.Genotypes_types AS VRC,
            SLF_table.Genotypes_types AS SLF,
            PVF_table.Genotypes_types AS PVF,
            Ai65_table.Genotypes_types AS Ai65,
            Ai80_table.Genotypes_types AS Ai80,
            VGC_table.Genotypes_types AS VGC,
            Ai162_table.Genotypes_types AS Ai162,
            Ai148_table.Genotypes_types AS Ai148   ,
            e.*,
            f.*,
            Alive
            
        FROM ExperimentalAnimals_table a
        LEFT JOIN MICE_table b ON a.Mouse_ID  = b.ID        
        LEFT JOIN Lines_table c ON c.ID=b.Line
        LEFT JOIN Projects_table d ON d.ID=a.Project
        LEFT JOIN Injections_table e ON e.ExpID = a.ID
        LEFT JOIN Windows_table f ON f.ExpID = a.ID
        LEFT JOIN VirusCombinations_table g ON g.ID=e.VirusCombination
        LEFT JOIN Virus_table h ON h.ID=g.Virus1
        LEFT JOIN Virus_table i ON i.ID=g.Virus2
        LEFT JOIN Virus_table j ON j.ID=g.Virus3
        LEFT JOIN Sensors_table k ON k.ID=h.Sensor
        LEFT JOIN Optos_table l ON l.ID=h.Opto
        LEFT JOIN Promoter_table m ON m.ID=h.Promoter
        LEFT JOIN Recombinase_table n ON n.ID=h.Recombinase
        LEFT JOIN Sensors_table o ON o.ID=i.Sensor
        LEFT JOIN Promoter_table p ON p.ID=i.Promoter
        LEFT JOIN Recombinase_table q ON q.ID=i.Recombinase
        LEFT JOIN Optos_table r ON r.ID=j.Opto
        LEFT JOIN Promoter_table s ON s.ID=j.Promoter
        LEFT JOIN Recombinase_table t ON t.ID=j.Recombinase
        LEFT JOIN Genotypes_table AS G2C_table   ON b.G2C   = G2C_table.ID
        LEFT JOIN Genotypes_table AS Ai14_table   ON b.Ai14   = Ai14_table.ID
        LEFT JOIN Genotypes_table AS Ai75_table   ON b.Ai75   = Ai75_table.ID
        LEFT JOIN Genotypes_table AS VRC_table   ON b.VRC   = VRC_table.ID
        LEFT JOIN Genotypes_table AS SLF_table   ON b.SLF   = SLF_table.ID
        LEFT JOIN Genotypes_table AS PVF_table   ON b.PVF   = PVF_table.ID
        LEFT JOIN Genotypes_table AS Ai65_table   ON b.Ai65   = Ai65_table.ID
        LEFT JOIN Genotypes_table AS Ai80_table   ON b.Ai80   = Ai80_table.ID
        LEFT JOIN Genotypes_table AS VGC_table   ON b.VGC  = VGC_table.ID
        LEFT JOIN Genotypes_table AS Ai162_table   ON b.Ai162   = Ai162_table.ID
        LEFT JOIN Genotypes_table AS Ai148_table   ON b.Ai148  = Ai148_table.ID
        LEFT JOIN Labels_table ON Labels_table.ID=a.EarMark    
        LEFT JOIN Experimental_table ON Experimental_table.ID=a.Experiment           
        LEFT JOIN ExperimentalStatus_table ON ExperimentalStatus_table.ID=a.Experimental_status           
        
        """
        self.all_experimental_all_info= self.databse_ref.arbitrary_query_to_df(query_all_experimental_all_info)     
        self.all_experimental_all_info= self.all_experimental_all_info.convert_dtypes(infer_objects=True, convert_string=False, convert_integer=True, convert_boolean=False, convert_floating=False)


    def table_all_animals_in_recovery(self):
 
        expanimalbetter=self.databse_ref.remove_unnecesary_virusinfo_from_df(self.databse_ref.remove_unnecesary_genes_from_df(self.all_experimental_all_info))
        self.all_exp_animals_recovery=expanimalbetter[(expanimalbetter['Room']==3) & (expanimalbetter['Alive']==1)]
        self.all_exp_animals_recovery= self.all_exp_animals_recovery.convert_dtypes(infer_objects=True, convert_string=False, convert_integer=True, convert_boolean=False, convert_floating=False)

        
    def table_all_mouse_with_window(self):

        recoveryanimalsbetter=self.databse_ref.remove_unnecesary_virusinfo_from_df(self.databse_ref.remove_unnecesary_genes_from_df(self.all_exp_animals_recovery))

        self.all_mouse_with_window=recoveryanimalsbetter[(recoveryanimalsbetter['Experimental_status']=='Done') & (recoveryanimalsbetter['Experiment']=='Window')]

    def table_all_mouse_to_do_window(self):

        query_all_windows="""
            SELECT 
                a.ID,
                Code,
                Lab_number, 
                Sex_types AS Sex,
                Cage, 
                round(round(julianday('now') - julianday(DOB))/7) AS Age,
                round(round(julianday('now') - julianday(Injection1Date))/7) AS WeeksFromInjection,
                Line_short,
                d.Projects,
                Labels_types,           
                round(round(julianday('now') - julianday(WindowDate))/7) AS WeeksFromWindow,
                a.Experimental_status,
                Experiment, 
                Combination,
                k.Sensors AS Sensors1,
                l.Optos AS Optos1,
                m.Promoters AS Promoters1,
                n.Recombinases AS Recombinases1,
                o.Sensors AS Sensors2,
                p.Promoters AS Promoters2,
                q.Recombinases AS Recombinases2,
                r.Optos AS Optos3,
                s.Promoters AS Promoters3,
                t.Recombinases AS Recombinases3,
                G2C_table.Genotypes_types AS G2C,
                Ai14_table.Genotypes_types AS Ai14,
                Ai75_table.Genotypes_types AS Ai75,
                VRC_table.Genotypes_types AS VRC,
                SLF_table.Genotypes_types AS SLF,
                PVF_table.Genotypes_types AS PVF,
                Ai65_table.Genotypes_types AS Ai65,
                Ai80_table.Genotypes_types AS Ai80,
                VGC_table.Genotypes_types AS VGC,
                Ai162_table.Genotypes_types AS Ai162,
                Ai148_table.Genotypes_types AS Ai148               
            FROM Windows_table 
            LEFT JOIN ExperimentalAnimals_table a ON a.WindowID=Windows_table.ID
            LEFT JOIN MICE_table b ON a.Mouse_ID  = b.ID        
            LEFT JOIN Lines_table c ON c.ID=b.Line
            LEFT JOIN Projects_table d ON d.ID=a.Project
            LEFT JOIN Injections_table e ON e.ExpID = a.ID
            LEFT JOIN Windows_table f ON f.ExpID = a.ID
            LEFT JOIN Labels_table ON Labels_table.ID=a.EarMark
            LEFT JOIN Sex_table ON Sex_table.ID=b.Sex
            LEFT JOIN VirusCombinations_table g ON g.ID=e.VirusCombination
            LEFT JOIN Virus_table h ON h.ID=g.Virus1
            LEFT JOIN Virus_table i ON i.ID=g.Virus2
            LEFT JOIN Virus_table j ON j.ID=g.Virus3
            LEFT JOIN Sensors_table k ON k.ID=h.Sensor
            LEFT JOIN Optos_table l ON l.ID=h.Opto
            LEFT JOIN Promoter_table m ON m.ID=h.Promoter
            LEFT JOIN Recombinase_table n ON n.ID=h.Recombinase
            LEFT JOIN Sensors_table o ON o.ID=i.Sensor
            LEFT JOIN Promoter_table p ON p.ID=i.Promoter
            LEFT JOIN Recombinase_table q ON q.ID=i.Recombinase
            LEFT JOIN Optos_table r ON r.ID=j.Opto
            LEFT JOIN Promoter_table s ON s.ID=j.Promoter
            LEFT JOIN Recombinase_table t ON t.ID=j.Recombinase
            LEFT JOIN Genotypes_table AS G2C_table   ON b.G2C   = G2C_table.ID
            LEFT JOIN Genotypes_table AS Ai14_table   ON b.Ai14   = Ai14_table.ID
            LEFT JOIN Genotypes_table AS Ai75_table   ON b.Ai75   = Ai75_table.ID
            LEFT JOIN Genotypes_table AS VRC_table   ON b.VRC   = VRC_table.ID
            LEFT JOIN Genotypes_table AS SLF_table   ON b.SLF   = SLF_table.ID
            LEFT JOIN Genotypes_table AS PVF_table   ON b.PVF   = PVF_table.ID
            LEFT JOIN Genotypes_table AS Ai65_table   ON b.Ai65   = Ai65_table.ID
            LEFT JOIN Genotypes_table AS Ai80_table   ON b.Ai80   = Ai80_table.ID
            LEFT JOIN Genotypes_table AS VGC_table   ON b.VGC  = VGC_table.ID
            LEFT JOIN Genotypes_table AS Ai162_table   ON b.Ai162   = Ai162_table.ID
            LEFT JOIN Genotypes_table AS Ai148_table   ON b.Ai148  = Ai148_table.ID
            WHERE b.Cage IS NOT NULL AND b.Alive=1
            """
        
        self.all_windows=self.databse_ref.arbitrary_query_to_df(query_all_windows)
        all_windowssbetter=self.databse_ref.remove_unnecesary_virusinfo_from_df(self.databse_ref.remove_unnecesary_genes_from_df(self.all_windows))
        self.all_mouse_to_do_window=all_windowssbetter[(all_windowssbetter['Experimental_status']==1) & (all_windowssbetter['Experiment']==4)]

    def table_all_mouse_to_do_injection(self):
            query_all_mouse_to_do_injection="""
                    SELECT 
                        a.ID,
                        Code,
                        Lab_number, 
                        Sex_types AS Sex,
                        Cage,  
                        round(round(julianday('now') - julianday(DOB))/7) AS Age,
                        Line_short,
                        d.Projects,
                        Labels_types,
                        a.Experimental_status,
                        Experiment, 
                        e.*,
                        Combination,
                        k.Sensors AS Sensors1,
                        l.Optos AS Optos1,
                        m.Promoters AS Promoters1,
                        n.Recombinases AS Recombinases1,
                        o.Sensors AS Sensors2,
                        p.Promoters AS Promoters2,
                        q.Recombinases AS Recombinases2,
                        r.Optos AS Optos3,
                        s.Promoters AS Promoters3,
                        t.Recombinases AS Recombinases3,
                        G2C_table.Genotypes_types AS G2C,
                        Ai14_table.Genotypes_types AS Ai14,
                        Ai75_table.Genotypes_types AS Ai75,
                        VRC_table.Genotypes_types AS VRC,
                        SLF_table.Genotypes_types AS SLF,
                        PVF_table.Genotypes_types AS PVF,
                        Ai65_table.Genotypes_types AS Ai65,
                        Ai80_table.Genotypes_types AS Ai80,
                        VGC_table.Genotypes_types AS VGC,
                        Ai162_table.Genotypes_types AS Ai162,
                        Ai148_table.Genotypes_types AS Ai148                              
                    FROM ExperimentalAnimals_table a
                    LEFT JOIN MICE_table b ON a.Mouse_ID  = b.ID        
                    LEFT JOIN Lines_table c ON c.ID=b.Line
                    LEFT JOIN Projects_table d ON d.ID=a.Project
                    LEFT JOIN Injections_table e ON e.ExpID = a.ID
                    LEFT JOIN Windows_table f ON f.ExpID = a.ID
                    LEFT JOIN VirusCombinations_table g ON g.ID=e.VirusCombination
                    LEFT JOIN Virus_table h ON h.ID=g.Virus1
                    LEFT JOIN Virus_table i ON i.ID=g.Virus2
                    LEFT JOIN Virus_table j ON j.ID=g.Virus3
                    LEFT JOIN Sensors_table k ON k.ID=h.Sensor
                    LEFT JOIN Optos_table l ON l.ID=h.Opto
                    LEFT JOIN Promoter_table m ON m.ID=h.Promoter
                    LEFT JOIN Recombinase_table n ON n.ID=h.Recombinase
                    LEFT JOIN Sensors_table o ON o.ID=i.Sensor
                    LEFT JOIN Promoter_table p ON p.ID=i.Promoter
                    LEFT JOIN Recombinase_table q ON q.ID=i.Recombinase
                    LEFT JOIN Optos_table r ON r.ID=j.Opto
                    LEFT JOIN Promoter_table s ON s.ID=j.Promoter
                    LEFT JOIN Recombinase_table t ON t.ID=j.Recombinase
                    LEFT JOIN Genotypes_table AS G2C_table   ON b.G2C   = G2C_table.ID
                    LEFT JOIN Genotypes_table AS Ai14_table   ON b.Ai14   = Ai14_table.ID
                    LEFT JOIN Genotypes_table AS Ai75_table   ON b.Ai75   = Ai75_table.ID
                    LEFT JOIN Genotypes_table AS VRC_table   ON b.VRC   = VRC_table.ID
                    LEFT JOIN Genotypes_table AS SLF_table   ON b.SLF   = SLF_table.ID
                    LEFT JOIN Genotypes_table AS PVF_table   ON b.PVF   = PVF_table.ID
                    LEFT JOIN Genotypes_table AS Ai65_table   ON b.Ai65   = Ai65_table.ID
                    LEFT JOIN Genotypes_table AS Ai80_table   ON b.Ai80   = Ai80_table.ID
                    LEFT JOIN Genotypes_table AS VGC_table   ON b.VGC  = VGC_table.ID
                    LEFT JOIN Genotypes_table AS Ai162_table   ON b.Ai162   = Ai162_table.ID
                    LEFT JOIN Genotypes_table AS Ai148_table   ON b.Ai148  = Ai148_table.ID
                    LEFT JOIN Sex_table ON Sex_table.ID=b.Sex
                    LEFT JOIN Labels_table ON Labels_table.ID=a.EarMark

                    WHERE Cage IS NOT NULL AND  Alive=1 AND a.Experimental_status=1 AND a.Experiment=2
            """
            self.all_mouse_to_do_injection=self.databse_ref.arbitrary_query_to_df(query_all_mouse_to_do_injection)

    def table_planned_but_in_colony(self):
            query_all_mouse_to_do_window="""
                SELECT 
                    Cage, 
                    Code,
                    Lab_number, 
                    round(round(julianday('now') - julianday(Injection1Date))/7) AS WeeksFromInjection,
                    d.Projects,
                    Line_short,
                    Status,
                    Experimental_types,
                    G2C_table.Genotypes_types AS G2C,
                    Ai14_table.Genotypes_types AS Ai14,
                    Ai75_table.Genotypes_types AS Ai75,
                    VRC_table.Genotypes_types AS VRC,
                    SLF_table.Genotypes_types AS SLF,
                    PVF_table.Genotypes_types AS PVF,
                    Ai65_table.Genotypes_types AS Ai65,
                    Ai80_table.Genotypes_types AS Ai80,
                    VGC_table.Genotypes_types AS VGC,
                    Ai162_table.Genotypes_types AS Ai162,
                    Ai148_table.Genotypes_types AS Ai148                    
                FROM ExperimentalAnimals_table a
                LEFT JOIN MICE_table b ON a.Mouse_ID  = b.ID            
                LEFT JOIN Lines_table c ON c.ID=b.Line
                LEFT JOIN Projects_table d ON d.ID=a.Project
                LEFT JOIN Experimental_table e ON e.ID=a.Experiment
                LEFT JOIN ExperimentalStatus_table f ON f.ID=a.Experimental_status
                LEFT JOIN Genotypes_table AS G2C_table   ON b.G2C   = G2C_table.ID
                LEFT JOIN Genotypes_table AS Ai14_table   ON b.Ai14   = Ai14_table.ID
                LEFT JOIN Genotypes_table AS Ai75_table   ON b.Ai75   = Ai75_table.ID
                LEFT JOIN Genotypes_table AS VRC_table   ON b.VRC   = VRC_table.ID
                LEFT JOIN Genotypes_table AS SLF_table   ON b.SLF   = SLF_table.ID
                LEFT JOIN Genotypes_table AS PVF_table   ON b.PVF   = PVF_table.ID
                LEFT JOIN Genotypes_table AS Ai65_table   ON b.Ai65   = Ai65_table.ID
                LEFT JOIN Genotypes_table AS Ai80_table   ON b.Ai80   = Ai80_table.ID
                LEFT JOIN Genotypes_table AS VGC_table   ON b.VGC  = VGC_table.ID
                LEFT JOIN Genotypes_table AS Ai162_table   ON b.Ai162   = Ai162_table.ID
                LEFT JOIN Genotypes_table AS Ai148_table   ON b.Ai148  = Ai148_table.ID       
                WHERE Cage IS NOT NULL AND Room=2 AND Alive=1 AND a.Experimental_status=1 AND (a.Experiment=2 OR a.Experiment=3 OR a.Experiment=11 OR (a.Experiment=4 AND  a.Injection1Date IS NULL))
                """    
            self.all_mouse_planned_in_colony=self.databse_ref.arbitrary_query_to_df(query_all_mouse_to_do_window)        
   
    def to_do_postop_injection(self, date_performed=False):
        
          if date_performed:
                 corrected_date_performed=datetime.datetime.strptime(date_performed, '%Y%m%d')               
          else:     
                 corrected_date_performed=datetime.date.today()         
          query_all_mouse_to_do_postop_injection="""
                        SELECT 
                            a.ID,
                            Cage, 
                            Code,
                            Labels_types,
                            Lab_number, 
                            julianday(date(?)) - julianday(date(Injection1Date)) AS DaysFromInjection,
                            julianday(date(?)) - julianday(date(WindowDate)) AS DaysFromWindow
                                    
                        FROM ExperimentalAnimals_table a
                        LEFT JOIN MICE_table b ON a.Mouse_ID  = b.ID          
                        LEFT JOIN Injections_table c ON c.ID=a.Injection1ID
                        LEFT JOIN Windows_table d ON d.ID=a.WindowID
                        LEFT JOIN Labels_table z ON z.ID=a.EarMark

                        WHERE Cage IS NOT NULL AND  Alive=1 AND (DaysFromWindow<3 OR DaysFromInjection<3) AND (c.PostInjection2 IS NULL OR d.PostInjection2 IS NULL )
                        """     

          params=(corrected_date_performed,corrected_date_performed)
          self.all_mouse_to_do_postop_injection=self.databse_ref.arbitrary_query_to_df(query_all_mouse_to_do_postop_injection, params)
            # WHERE Cage IS NOT NULL AND  Alive=1 AND (DaysFromWindow<3 OR DaysFromInjection<3)
    def table_virus_info(self):       
            query_visuscombinationsinfo="""
            SELECT a.ID,
                Combination,
                e.Sensors,
                f.Optos,
                g.Promoters,
                h.Recombinases,
                i.Sensors,
                k.Promoters,
                l.Recombinases,
                n.Optos,
                o.Promoters,
                p.Recombinases
            FROM VirusCombinations_table a
            LEFT JOIN Virus_table b ON b.ID=a.Virus1
            LEFT JOIN Virus_table c ON c.ID=a.Virus2
            LEFT JOIN Virus_table d ON d.ID=a.Virus3
            LEFT JOIN Sensors_table e ON e.ID=b.Sensor
            LEFT JOIN Optos_table f ON f.ID=b.Opto
            LEFT JOIN Promoter_table g ON g.ID=b.Promoter
            LEFT JOIN Recombinase_table h ON h.ID=b.Recombinase
            LEFT JOIN Sensors_table i ON i.ID=c.Sensor
            LEFT JOIN Promoter_table k ON k.ID=c.Promoter
            LEFT JOIN Recombinase_table l ON l.ID=c.Recombinase
            LEFT JOIN Optos_table n ON n.ID=d.Opto
            LEFT JOIN Promoter_table o ON o.ID=d.Promoter
            LEFT JOIN Recombinase_table p ON p.ID=d.Recombinase
            
            """
            params=()
            self.virusinfo=self.databse_ref.arbitrary_query_to_df(query_visuscombinationsinfo, params)


             
#%%            
    def add_new_planned_experimental_cage(self,cage):  
        
        query_mice_exp_info="SELECT ID, Lab_number, Line, Label, Room FROM MICE_table   WHERE Cage=?"
        params=(cage,)
        mice_exp = self.databse_ref.arbitrary_query_to_df(query_mice_exp_info, params).values.tolist()

        mice_exp_IDs=[i[0] for i in mice_exp]
        mice_exp_labels=[i[3] for i in mice_exp]
        mice_exp_lines=[i[2] for i in mice_exp]
        
        project=[]
        for line in  mice_exp_lines:
            if line in [5,12,13]:
                project.append(4)
            elif line in [18,19,20,21,22,23]:
                project.append(2)               
            elif line in [16,17]:
                project.append(7)
            elif line in [26]:
                project.append(8)
                                
        query_mice_exp="""
         UPDATE MICE_table
         SET  Experimental_Status=3, Experimental_Code=?
         WHERE ID=?
         """
        query_add_exps=""" 
              INSERT INTO ExperimentalAnimals_table( ID, Code, Project, Mouse_ID, EarMark, Experimental_status, Experiment)
              VALUES(?,?,?,?,?,1,3) 
            """              
        actions_dictionary={cage:{11:((),(),())                                   
                                  }
                            }
        self.databse_ref.add_multiple_actions(actions_dictionary, commit=False)    
    
        for i, mouse in enumerate(mice_exp_IDs):
            # update MICE_table
            if i==0:
                max_code=self.max_current_experimenatl_code[0]
                next_code=max_code+1
                max_lettercode=self.max_current_experimenatl_code[1]
                nextletter_code=get_next_twolettercode(max_lettercode)
                
            else:
                next_code=next_code+1
                nextletter_code=get_next_twolettercode(nextletter_code)
                
            params=(next_code,mouse)
            self.databse_ref.arbitrary_updating_record(query_mice_exp,params)

            # add new record to experimental
            #get last code

            if mice_exp_labels[i]==1:
                EarMark=7
            else:
                EarMark=mice_exp_labels[i]
                
            params_insert=(next_code, 
                    nextletter_code,
                    project[i],
                    mouse, 
                    EarMark)    
            self.databse_ref.arbitrary_inserting_record(query_add_exps, params_insert, commit=False)          
            mice_exp[i].append(next_code)
            
            
        return mice_exp
    
    def plan_new_injection(self, values,  cage, animals_selected, injection_sites, virus_dilutions):
        
        res = list(zip(*virus_dilutions))   
        virus_tuple=res[0]
        viruscomb=get_combination_from_virus(virus_tuple, self)
        
        if isinstance(animals_selected[0], int):
            mice_exp=self.add_new_planned_experimental_cage(cage) 
            seleced_mice_exp=[mouse_exp for mouse_exp in mice_exp if mouse_exp[1] in animals_selected]
            selected_mouse_IDs=[i[0] for i in seleced_mice_exp ]
            params=tuple(selected_mouse_IDs)    
            query_mice_exp_info='SELECT ID FROM ExperimentalAnimals_table WHERE Mouse_ID IN (%s)' % ','.join('?' for i in params)   
            mice_experiments_IDs = self.databse_ref.arbitrary_query_to_df(query_mice_exp_info, params).values.tolist()
        else:    
            params=tuple(animals_selected)
            query_mice_exp_info='SELECT ID, Mouse_ID FROM ExperimentalAnimals_table WHERE Code IN (%s)' % ','.join('?' for i in params)  
            mice_experiments_IDs = self.databse_ref.arbitrary_query_to_df(query_mice_exp_info, params).values.tolist()
            selected_mouse_IDs=[i[1] for i in mice_experiments_IDs ]
            selected_mouse_exp_IDs=[i[0] for i in mice_experiments_IDs ]
            params=tuple(selected_mouse_IDs)
            query_mice_exp_info2='SELECT ID, Lab_number, Line, Label, Room FROM MICE_table   WHERE ID IN (%s)' % ','.join('?' for i in params)  
            mice_exp = self.databse_ref.arbitrary_query_to_df(query_mice_exp_info2, params).values.tolist()
            seleced_mice_exp=[mouse_exp for mouse_exp in mice_exp if mouse_exp[0] in selected_mouse_IDs]

        for mouse in  seleced_mice_exp:
            if mouse[2] in [5,12,13]:
                 if any(x in res[0] for x in ['B', 'C1V1', 'I']):
                      mouse.append(5)  
                 else:
                      mouse.append(4)
            elif mouse[2] in [18,19,20,21,22,23]:
                if any(x in res[0] for x in ['B', 'C1V1', 'I']):
                      mouse.append(3)
                else:
                      mouse.append(2)
                     
            elif mouse[2] in [16,17]:
                      mouse.append(7)
         
        query_update_exps=""" 
             UPDATE  ExperimentalAnimals_table
             SET Project=?, Experimental_status=1 , Experiment=2 , NumberOfInjections=1 , Injection1Date='TODO', Injection1ID=?
             WHERE Mouse_ID=?
             """  
             
        query_add_injections=""" 
                  INSERT INTO Injections_table(
                      ExpID,
                      InjDate,
                      VirusCombination,
                      DilutionSensor1,
                      DilutionSensor2,
                      DilutionOpto,
                      CorticalArea,
                      InjectionSites,
                      InjectionSite1Coordinates,
                      InjectionSite1volume,
                      InjectionSite1speed,
                      InjectionSite1pretime,
                      InjectionSite1posttime,
                      InjectionSite2Coordinates,
                      InjectionSite2volume,
                      InjectionSite2speed,
                      InjectionSite2pretime,
                      InjectionSite2posttime                     
                    )
                  VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) 
                """           
       
   
  
        all_inject_params=values
        datezz='TODO'
        
        for i, mouse_exp in enumerate(seleced_mice_exp):
  
            if i==0:
                next_injection_id=self.max_current_injection_code+1
            else:
                next_injection_id=next_injection_id+1
                     
            params=(mouse_exp[-1], next_injection_id, mouse_exp[0])
            self.databse_ref.arbitrary_updating_record(query_update_exps,params)
            
            # add new record to injecitons
            exp_ID=mice_experiments_IDs[i][0]
            
            params=((exp_ID, datezz,viruscomb[0][0])+ tuple( (all_inject_params[i+1][5:15])+(all_inject_params[i+1][17:22])  ))
            self.databse_ref.arbitrary_inserting_record(query_add_injections,params)
            
        self.databse_ref.independent_commit()    

    def update_performed_injections(self, all_inject_params, cage, selected_codes, raw_date_performed):
        
        date_performed=datetime.datetime.strptime(raw_date_performed, '%Y%m%d').date()
      
              
        query_mice_exp_info="SELECT ExperimentalAnimals_table.ID, MICE_table.ID, Lab_number,Code,Label,Room,ExperimentalAnimals_table.Experimental_status, MICE_table.Experimental_status, Experiment,Injection1ID FROM ExperimentalAnimals_table LEFT JOIN MICE_table ON MICE_table.ID = ExperimentalAnimals_table.Mouse_ID WHERE Cage=?"
        cage=cage       
        params=(cage,)
        mice_exp=self.databse_ref.arbitrary_query_to_df(query_mice_exp_info,params).values.tolist()  

        
        good_mice_exp=[i for i in mice_exp if i[3] in selected_codes]
         
        all_inject_params=all_inject_params
            
        
        
        labels_list=[mouse[1] for mouse in all_inject_params[1:]]
        labels_tuple=tuple(labels_list)
        labels=transform_earlabels_to_codes(labels_tuple, self)
        for i,mouse in enumerate(all_inject_params[1:]):
            mouse[1]=labels[i]
        

        room=good_mice_exp[0][5]
        # udate actions room change and experiment 
        
        
        if room==2:
            actions_dictionary={cage:{12:((),(),(date_performed)) 
                                      }
                                }
            self.databse_ref.add_multiple_actions(actions_dictionary)
            
        else:
            print('already out of colony')
        
        query_mice_cage_update="""
                UPDATE MICE_table
                SET  Room=3
                WHERE Cage=?
            """        
        params=(cage,)
        self.databse_ref.arbitrary_updating_record(query_mice_cage_update, params)
        
        # update queries  
        query_mice_injected_update="""
                UPDATE MICE_table
                SET  Room=3, Label=?, Experimental_Status=2
                WHERE ID=?
            """
        query_update_exps=""" 
             UPDATE  ExperimentalAnimals_table
             SET EarMark=?, Experimental_status=2 ,Experiment=2 , NumberOfInjections=1 , Injection1Date=date(?)
             WHERE ID=?
             """
        query_update_injections=""" 
                 UPDATE  Injections_table
                 SET 
                     InjDate=date(?),
                     VirusCombination=?,
                     DilutionSensor1=?,
                     DilutionSensor2=?,
                     DilutionOpto=?,
                     CorticalArea=?,
                     InjectionSites=?,
                     InjectionSite1Coordinates=?,
                     InjectionSite1volume=?,
                     InjectionSite1speed=?,
                     InjectionSite1pretime=?,
                     InjectionSite1posttime=?,
                     InjectionSite1goodvolume=?,
                     InjectionSite1bleeding=?,
                     InjectionSite2Coordinates=?,
                     InjectionSite2volume=?,
                     InjectionSite2speed=?,
                     InjectionSite2pretime=?,
                     InjectionSite2posttime=?,
                     InjectionSite2goodvolume=? ,                  
                     InjectionSite2bleeding=?,
                     Notes=?
                     WHERE  ID=?
                    
                """    
                

        for i, mouse in enumerate(good_mice_exp):
            noinjection=0
            if not (all_inject_params[i+1][3] or all_inject_params[i+1][4] or all_inject_params[i+1][3] ):
                for j, k in enumerate(all_inject_params[i+1][:-1]):
                    if j>5:
                        all_inject_params[i+1][j]=np.nan                       
                        noinjection=1
        # update MICE_table
            params=(all_inject_params[i+1][1], mouse[1],)
            self.databse_ref.arbitrary_updating_record(query_mice_injected_update,params)
        # update exper
  
            params=(all_inject_params[i+1][1], date_performed ,mouse[0] )
            self.databse_ref.arbitrary_updating_record(query_update_exps,params)
        # update injection
        
            virus_tuple=(all_inject_params[i+1][3],all_inject_params[i+1][4],all_inject_params[i+1][5])
            if any(virus_tuple):
                viruscomb=get_combination_from_virus(virus_tuple, self)
            else:
                viruscomb=[[28]]
            list_for_update=[all_inject_params[i+1][2],viruscomb[0][0]]+all_inject_params[i+1][6:]  
            list_for_update[0]=date_performed
            if not noinjection:
                list_for_update_integercorrected=[ int(float(j))  if i in [5,6,7,14] else j for i,j in enumerate(list_for_update)]
            else:
                list_for_update_integercorrected=list_for_update
            tuple_for_update=tuple(list_for_update_integercorrected )
            tuple_inje_id=(mouse[-1],)
            tuple_for_update=  (tuple_for_update +tuple_inje_id)
            
            params=(tuple_for_update)
            self.databse_ref.arbitrary_updating_record(query_update_injections,params)
            
        self.databse_ref.independent_commit()    

    def plan_new_window(self, all_winodws_parameters, lab_number_selected=False, codes_selected=False, ):
        
        if lab_number_selected:
            params=tuple(lab_number_selected)    
            query_mice_cage='SELECT Cage FROM MICE_table WHERE Lab_Number IN (%s)' % ','.join('?' for i in params)   
            mice_cages = self.databse_ref.arbitrary_query_to_df(query_mice_cage, params).values.tolist()
            
            codes_selected=[]
            for cage in mice_cages:
                mice_exp=self.add_new_planned_experimental_cage(cage[0])
                seleced_mice_exp=[mouse_exp for mouse_exp in mice_exp if str(mouse_exp[1]) in lab_number_selected]
                selected_mouse_IDs=[i[0] for i in seleced_mice_exp ]
                params=tuple(selected_mouse_IDs)    
                query_mice_exp_info='SELECT Code FROM ExperimentalAnimals_table WHERE Mouse_ID IN (%s)' % ','.join('?' for i in params)   
                mice_experiments_codes = self.databse_ref.arbitrary_query_to_df(query_mice_exp_info, params).values.tolist()[0]
                codes_selected.append(mice_experiments_codes)
            codes_selected = [item for sublist in codes_selected for item in sublist]

    
    
            
        params=tuple(codes_selected)
        # query_check_exp_info="SELECT * FROM ExperimentalAnimals_table  WHERE Code=?"
        query_check_exp_info='SELECT * FROM ExperimentalAnimals_table WHERE Code IN (%s)' % ','.join('?' for i in codes_selected)    
        selected_mice_exp_info=self.databse_ref.arbitrary_query_to_df(query_check_exp_info, params)

 
        all_winodws_parameters=all_winodws_parameters

        query_update_exps=""" 
             UPDATE  ExperimentalAnimals_table
             SET  Experimental_status=1, Experiment=4, WindowDate='TODO', WindowID=?
             WHERE ID=?
             """  
        query_add_windw=""" 
                    INSERT INTO Windows_table(   
                      ID,
                      ExpID,
                      WindDate,
                      CorticalArea,
                      HeadPlateCoordinates,
                      WindowType,
                      CranioSize,
                      CoverType,
                      CoverSize,
                      Durotomy            
                      )
                    VALUES(?,?,'TODO',?,?,?,?,?,?,?) 
                  """             

        mice_info_list=[]       
        mouse_dict={}
        # get animal info
        # for i, exp in enumerate(selected_mice_exp_info):
        for index, row in selected_mice_exp_info.iterrows():

            mouse_dict['Code']=row['Code']
            mouse_dict['WindowDate']='TODO'
            if index==0:
                windowcode=self.max_current_window_code+1               
            else:
                windowcode=windowcode+1
            mouse_dict['WindowID']=windowcode    
            mouse_dict['ExpID']=row['ID']
            mouse_dict['Experimental_status']=1
            mouse_dict['Experiment']=4
            mouse_dict['MouseID']=row['Mouse_ID']
            mice_info_list.append(mouse_dict)
            # update exp and main     
            
            params=( mouse_dict['WindowID'],mouse_dict['ExpID'])
            self.databse_ref.arbitrary_updating_record(query_update_exps,params)
        # add new window
            integg=[2,3,4,5,6,7,8]
            windwo_parameters=[int(param) if i in integg else param for i, param in enumerate(all_winodws_parameters[index+1])]

            tuple_for_adding_window=(mouse_dict['WindowID'],mouse_dict['ExpID'],)+tuple(windwo_parameters[2:])            
            params=tuple_for_adding_window
            self.databse_ref.arbitrary_inserting_record(query_add_windw,params)
            
        self.databse_ref.independent_commit()        
    
                       
    def update_performed_window(self,all_inject_params, cage, animals_selected, raw_date_performed=False):   
        # cage=cage[0]
        date_performed=datetime.datetime.strptime(raw_date_performed, '%Y%m%d').date()

    
        query_mice_exp_info="""
        SELECT 
            ExperimentalAnimals_table.ID,
            MICE_table.ID, 
            Lab_number,
            Code,
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
        WHERE Cage=? AND ExperimentalAnimals_table.Experimental_status=1 AND ExperimentalAnimals_table.Experiment=4 AND Code IN (%s)""" % ','.join('?' for i in animals_selected)            
        params=(cage,)+tuple(animals_selected)
        mice_exp = self.databse_ref.arbitrary_query_to_df(query_mice_exp_info, params).values.tolist()

        updated_window_values=all_inject_params
        labels_list=[mouse[2] for mouse in updated_window_values[1:]]
        labels_tuple=tuple(labels_list)
        labels=transform_earlabels_to_codes(labels_tuple, self.databse_ref)       
        # labels=transform_earlabels_to_codes(labels_tuple, MouseDat)
        for i,mouse in enumerate(updated_window_values[1:]):
            mouse[2]=labels[i]
        

        room=mice_exp[0][5]
        # udate actions room change and experiment 
        if room==2:
            actions_dictionary={cage:{12:((),(),(date_performed)) 
                                      }
                                }
            self.databse_ref.add_multiple_actions(actions_dictionary)
            
        else:
            print('already out of colony')
        
            
        # update queries  
        query_mice_window_update="""
                UPDATE MICE_table
                SET Label=?, Experimental_Status=4
                WHERE ID=?
            """
        query_update_exps=""" 
             UPDATE  ExperimentalAnimals_table
             SET EarMark=?, Experimental_status=2, Experiment=4, WindowDate=date(?)
             WHERE ID=?
             """
        query_update_windows=""" 
                 UPDATE  Windows_table
                 SET 
                    WindDate=date(?),
                    CorticalArea=?,
                    HeadPlateCoordinates=?,
                    WindowType=?,
                    CranioSize=?,
                    CoverType=?,
                    CoverSize=?,
                    Durotomy=?,
                    DamagedAreas=?,
                    Notes=?
                 WHERE  ID=?
                    
                """  
        # update rest of the cage   
        query_mice_cage_update="""
                UPDATE MICE_table
                SET  Room=3
                WHERE Cage=?
            """        
        params=(cage,)
        self.databse_ref.arbitrary_updating_record(query_mice_cage_update, params)
            
        for i, mouse in enumerate(mice_exp):
        # update MICE_table
            params=(updated_window_values[i+1][2], mouse[1])
            self.databse_ref.arbitrary_updating_record(query_mice_window_update, params)
        # update exper
            params=(updated_window_values[i+1][2], date_performed ,mouse[0] )
            self.databse_ref.arbitrary_updating_record(query_update_exps, params)

        # update injection
        
            list_for_update=[updated_window_values[i+1][1]]+updated_window_values[i+1][3:] 
            list_for_update[0]=date_performed

            list_for_update_integercorrected=[ int(j)  if i in list(range(1,8)) else j for i,j in enumerate(list_for_update)]
            
            tuple_for_update=tuple(list_for_update_integercorrected )
            tuple_inje_id=(mouse[10],)
            tuple_for_update=  (tuple_for_update +tuple_inje_id)
            
            params=(tuple_for_update )
            self.databse_ref.arbitrary_updating_record(query_update_windows, params, commit=True)
            

        
    def update_postop_injection(self, cage, animals_selected, date_performed=False, ignored=False):   
            date_performedbackup=date_performed

            for mouse in animals_selected:  
                if date_performed=='NOT DONE':
                    date_performed_corrected=date_performedbackup
                if date_performed:
                    self.to_do_postop_injection(date_performed)
                    date_performed_corrected=datetime.datetime.strptime(date_performed, '%Y%m%d')               
                else:     
                    date_performed_corrected=datetime.date.today()

                mouse_info=self.all_mouse_to_do_postop_injection[self.all_mouse_to_do_postop_injection['Code']==mouse].values.tolist()[0]
                daysfromwindows=mouse_info[-1]
                daysfrominjection=mouse_info[-2]
                if not daysfromwindows:
                    daysfromwindows=np.nan
                if not math.isnan(daysfromwindows):
                   if daysfromwindows==1:
                       query_update_postop=""" 
                            UPDATE  Windows_table
                            SET PostInjection1=date(?)           
                            WHERE  ExpID=?                           
                           """            
                   elif daysfromwindows==2:
                       query_update_postop=""" 
                            UPDATE  Windows_table
                            SET PostInjection2=date(?)
                            WHERE  ExpID=?                              
                           """       
                elif not math.isnan(daysfrominjection):                     
                    if daysfrominjection  in [0,1]:
                        query_update_postop=""" 
                            UPDATE  Injections_table
                            SET  PostInjection1=date(?)
                            WHERE  ExpID=?  
                            """      
                    elif daysfrominjection==2:
                          query_update_postop=""" 
                            UPDATE  Injections_table
                            SET PostInjection2=date(?)
                            WHERE  ExpID=?     
                               """  
                if ignored:
                   date_performed_corrected='NOT DONE'
                   query_update_postop=query_update_postop.replace('date(?)', '?')

                   
                params=(date_performed_corrected, mouse_info[0])                         
                self.databse_ref.arbitrary_updating_record(query_update_postop, params, commit=False)
                
            self.databse_ref.independent_commit()    

    def mouse_dead_during_surgery(self, mouse_code, date=False): 
        Action_Type_mouse_dead=19
        query_mouse_dead_info="SELECT Mouse_ID, Cage, Lab_number FROM ExperimentalAnimals_table LEFT JOIN MICE_table a ON a.ID=ExperimentalAnimals_table.Mouse_ID  WHERE Code=?"      
        params=(mouse_code,)
        dead_mouse_info= self.databse_ref.arbitrary_query_to_df(query_mouse_dead_info, params).values.tolist()[0]
        
        
        dead_mouse_ID=dead_mouse_info[0]
        dead_mouse_cage=dead_mouse_info[1]
        dead_mouse_lab_number=dead_mouse_info[2]
        # add date here
        actions_dictionary={dead_mouse_cage:{Action_Type_mouse_dead:((dead_mouse_lab_number,),(),(date)), 
                                  },
                        }
        self.databse_ref.add_multiple_actions(actions_dictionary)
        
        query_update_mouse_dead="""
                UPDATE MICE_table
                SET  Cage=NULL, Alive=0, Experimental_Status=9, Room=1
                WHERE ID=?
                """       
        params=(dead_mouse_ID,)
        self.databse_ref.arbitrary_updating_record(query_update_mouse_dead,params)
        
        query_update_experimental_status="""
                        UPDATE ExperimentalAnimals_table
                        SET  Experimental_status=4 , Experiment=9, Notes='Dead during surgery'
                        WHERE Mouse_ID=?           
                        """                                        
        params=(dead_mouse_ID,)
        self.databse_ref.arbitrary_updating_record(query_update_experimental_status,params, commit=True)

    def mouse_dead_postop(self, mouse_code): 
        Action_Type_mouse_dead=19
        query_mouse_dead_info="SELECT Mouse_ID, Cage, Lab_number FROM ExperimentalAnimals_table LEFT JOIN MICE_table a ON a.ID=ExperimentalAnimals_table.Mouse_ID  WHERE Code=?"      
        params=(mouse_code,)
        dead_mouse_info= self.databse_ref.arbitrary_query_to_df(query_mouse_dead_info, params).values.tolist()[0]
        
        
        dead_mouse_ID=dead_mouse_info[0]
        dead_mouse_cage=dead_mouse_info[1]
        dead_mouse_lab_number=dead_mouse_info[2]
        
        actions_dictionary={dead_mouse_cage:{Action_Type_mouse_dead:((dead_mouse_lab_number,),(),()), 
                                  },
                        }
        self.databse_ref.add_multiple_actions(actions_dictionary)
        
        query_update_mouse_dead="""
                UPDATE MICE_table
                SET  Cage=NULL, Alive=0, Experimental_Status=9, Room=1
                WHERE ID=?
                """       
        params=(dead_mouse_ID,)
        self.databse_ref.arbitrary_updating_record(query_update_mouse_dead,params)
        
        query_update_experimental_status="""
                        UPDATE ExperimentalAnimals_table
                        SET  Experimental_status=4 , Experiment=9, Notes='Dead after surgery'
                        WHERE Mouse_ID=?           
                        """                                        
        params=(dead_mouse_ID,)
        self.databse_ref.arbitrary_updating_record(query_update_experimental_status, params, commit=True)
            
        
        
    def sac_experimental_mouse(self, mouse_code, date=False):
         if date:
            corrected_date_performed=datetime.datetime.strptime(date, '%Y%m%d')               
         else:     
            corrected_date_performed=datetime.date.today()
        
         Action_Type_mouse_saced=4
         query_mouse_saced_info="SELECT Mouse_ID, Cage, Lab_number FROM ExperimentalAnimals_table LEFT JOIN MICE_table a ON a.ID=ExperimentalAnimals_table.Mouse_ID  WHERE Code=?"      
         params=(mouse_code,)
         dead_mouse_info= self.databse_ref.arbitrary_query_to_df(query_mouse_saced_info, params).values.tolist()[0]
         
         
         dead_mouse_ID=dead_mouse_info[0]
         dead_mouse_cage=dead_mouse_info[1]
         dead_mouse_lab_number=dead_mouse_info[2]
         
         actions_dictionary={dead_mouse_cage:{Action_Type_mouse_saced:((dead_mouse_lab_number,),(),(corrected_date_performed)), 
                                   },
                         }
         self.databse_ref.add_multiple_actions(actions_dictionary)
         
         query_update_mouse_dead="""
                 UPDATE MICE_table
                 SET  Cage=NULL, Alive=0, Experimental_Status=10, Room=1
                 WHERE ID=?
                 """       
         params=(dead_mouse_ID,)
         self.databse_ref.arbitrary_updating_record(query_update_mouse_dead,params)
         
         query_update_experimental_status="""
                         UPDATE ExperimentalAnimals_table
                         SET  Experimental_status=4 , Experiment=10
                         WHERE Mouse_ID=?           
                         """                                        
         params=(dead_mouse_ID,)
         self.databse_ref.arbitrary_updating_record(query_update_experimental_status,params, commit=True)
   
    def brain_fixation(self, values, mouse_codes, date_performed):
        
        if date_performed:
            corrected_date_performed=datetime.datetime.strptime(date_performed, '%Y%m%d')               
        else:     
            corrected_date_performed=datetime.date.today()
            
            
        for i, mouse in  enumerate(mouse_codes):
            params=(mouse,)
            query_mouse_brain_info="SELECT ExperimentalAnimals_table.ID, Mouse_ID, Cage, Lab_number FROM ExperimentalAnimals_table LEFT JOIN MICE_table a ON a.ID=ExperimentalAnimals_table.Mouse_ID  WHERE Code=?"     
            brain_mouse_info= self.databse_ref.arbitrary_query_to_df(query_mouse_brain_info, params).values.tolist()[0]

            dead_mouse_cage=brain_mouse_info[2]
            dead_mouse_lab_number=brain_mouse_info[3]
            Action_Type_mouse_saced=4
          
            actions_dictionary={dead_mouse_cage:{Action_Type_mouse_saced:((dead_mouse_lab_number,),(),()), 
                                       },
                             }
            self.databse_ref.add_multiple_actions(actions_dictionary)
            
    
            Brain_ID =self.max_current_brain_code+1
            Exp_Mouse_ID=brain_mouse_info[0]
            Fixation_date=corrected_date_performed
            Perfusion_solution=values[i+1][2]
            Postfixation_solution=values[i+1][3]
            Postfixation_time=values[i+1][4]
            Postfixation_temperature=values[i+1][5]
            Prehistology_storage_date=corrected_date_performed+datetime.timedelta(days=1)
            Prehistology_storage_solution=values[i+1][6]
            Prehistology_storage_location=values[i+1][7]
            Comments=values[i+1][8]
    
            query_add_brain_fixation=""" 
                  INSERT INTO BrainProcessing_table( ID, 
                                                    Exp_Mouse_ID,
                                                    Fixation_date, 
                                                    Perfusion_solution, 
                                                    Postfixation_solution,
                                                    Postfixation_time,
                                                    Postfixation_temperature,
                                                    Prehistology_storage_date,
                                                    Prehistology_storage_solution,
                                                    Prehistology_storage_location,
                                                    Comments
      
                                                    )
                  VALUES(?,?,date(?),?,?,?,?,date(?),?,?,?) 
                """              
            params=(Brain_ID, 
                    Exp_Mouse_ID, 
                    Fixation_date,
                    Perfusion_solution,
                    Postfixation_solution,
                    Postfixation_time,
                    Postfixation_temperature, 
                    Prehistology_storage_date,
                    Prehistology_storage_solution,
                    Prehistology_storage_location,
                    Comments)
            
            self.databse_ref.arbitrary_inserting_record(query_add_brain_fixation,params)
    
            
            
            query_update_exps_brain=""" 
                 UPDATE  ExperimentalAnimals_table
                 SET  Experimental_status=2, Experiment=6, BrainProcessingDate=?, BrainProcessingID=?
                 WHERE ID=?
                 """  
            params=(Fixation_date, Brain_ID, Exp_Mouse_ID)
            self.databse_ref.arbitrary_updating_record(query_update_exps_brain, params)
       
            query_update_mouse_dead="""
                    UPDATE MICE_table
                    SET  Cage=NULL, Alive=0, Experimental_Status=6, Room=1
                    WHERE ID=?
                    """       
            params=(brain_mouse_info[1],)
            self.databse_ref.arbitrary_updating_record(query_update_mouse_dead,params)
            
        self.databse_ref.independent_commit()    
