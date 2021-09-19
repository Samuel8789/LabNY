# # -*- coding: utf-8 -*-
# """
# Created on Wed Jul 21 11:34:23 2021

# @author: sp3660
# """
# import pandas as pd

# query_all_imaged_mice="""
#     SELECT  
#              c.Code,         
#              Sex_types AS Sex,
#              Line_short,
#              g.Projects,
#              f.cage,
#              bb.Labels_types,
#              aa.Genotypings_types,
#              cc.Room_types,
#              f.Alive,
#              f.Notes,
            
#              b.ImagingDate,
#              b.Objectives,
#              b.EndOfSessionSummary,  
#              ll.Microscope,
#              b.ImagingSessionRawPath,
#              b.MiceRawPath,
#              b.IssuesDuringImaging,
            
#              date(f.DOB) AS DOB,
#              date(c.Injection1Date) AS InjectionDate,
#              date(c.WindowDate) AS WindowDate,
#              c.NumberOfInjections , 
#              c.Notes,
                
#              a.EyesComments,
#              a.BehaviourComments,
#              a.FurComments,
#              a.LessionComments,
#              a.BehaviourProtocolComments,
#              a.OptogeneticsProtocolComments,
#              a.Objectives,
#              a.EndOfSessionSummary,
#              a.IssuesDuringImaging,
             
             
             
             
#              d.DamagedAreas,
#              d.Notes AS WindowNotes,
#              ff.BrainAreas,
#              gg.CoverslipSize,
#              hh.WindowType,
#              ii.CraniectomySize,
#              jj.CraniectomyType,
#              kk.X,
#              kk.Y,
#              kk.Z,
#              d.PostInjection1,
#              d.PostInjection2,
#              d.Durotomy,
             
             
            
#              e.Notes AS InjectionNotes,
#              e.PostInjection1,
#              e.PostInjection2,
#              Combination,
#              e.InjectionSites,
#              e.InjectionSite1volume,    
#              e.InjectionSite2volume,     
#              e.InjectionSite1speed,    
#              e.InjectionSite2speed,     
#              e.InjectionSite1pretime,    
#              e.InjectionSite2pretime,     
#              e.InjectionSite1posttime,    
#              e.InjectionSite2posttime,     
#              e.InjectionSite1goodvolume,    
#              e.InjectionSite2goodvolume, 
#              e.InjectionSite1bleeding,    
#              e.InjectionSite2bleeding, 
#              oo.X,
#              oo.Y,
#              oo.Z,
#              pp.X,
#              pp.Y,
#              pp.Z,


                                
#              k.Sensors AS Sensors1,
#              e.DilutionSensor1,
#              l.Optos AS Optos1,
#              m.Promoters AS Promoters1,
#              n.Recombinases AS Recombinases1,
#              qq.Serotypes AS Serotype1,
            
#              o.Sensors AS Sensors2,
#              e.DilutionSensor2,
#              p.Promoters AS Promoters2,
#              q.Recombinases AS Recombinases2,
#              rr.Serotypes AS Serotype2,
            
#              r.Optos AS Optos3,
#              e.DilutionOpto,
#              s.Promoters AS Promoters3,
#              t.Recombinases AS Recombinases3,
#              ss.Serotypes AS Serotype3,
            
            
#              G2C_table.Genotypes_types AS G2C,
#              Ai14_table.Genotypes_types AS Ai14,
#              Ai75_table.Genotypes_types AS Ai75,
#              VRC_table.Genotypes_types AS VRC,
#              SLF_table.Genotypes_types AS SLF,
#              PVF_table.Genotypes_types AS PVF,
#              Ai65_table.Genotypes_types AS Ai65,
#              Ai80_table.Genotypes_types AS Ai80,
#              VGC_table.Genotypes_types AS VGC,
#              Ai162_table.Genotypes_types AS Ai162,
#              Ai148_table.Genotypes_types AS Ai148,    
#              c.SlowStoragePath, 
#              c.WorkingStoragePath,
#              a.SlowStoragePath, 
#              a.WorkingStoragePath,
             
#              z.WideFieldImagePath,
#              z.WideFieldFileName,
#              z.WideFieldComments,
#              z.SlowStoragePath,
#              z.WorkingStoragePath
    
#     FROM ImagedMice_table                   a
#     LEFT JOIN ExperimentalAnimals_table     c ON c.ID           =   a.ExpID
#     LEFT JOIN ImagingSessions_table         b ON a.SessionID    =   b.ID
#     LEFT JOIN Windows_table                 d ON d.ID           =   c.WindowID
#     LEFT JOIN Injections_table              e ON e.ID           =   c.Injection1ID
#     LEFT JOIN MICE_table                    f ON f.ID           =   c.Mouse_ID
#     LEFT JOIN WideField_table               z ON z.ID           =   a.WideFIeldID 
     
    
#     LEFT JOIN Projects_table            g               ON g.ID                         = c.Project 
#     LEFT JOIN VirusCombinations_table                   ON VirusCombinations_table.ID   = e.VirusCombination         
#     LEFT JOIN Virus_table               h               ON h.ID                         = VirusCombinations_table.Virus1
#     LEFT JOIN Virus_table               i               ON i.ID                         = VirusCombinations_table.Virus2
#     LEFT JOIN Virus_table               j               ON j.ID                         = VirusCombinations_table.Virus3
#     LEFT JOIN Sensors_table             k               ON k.ID                         = h.Sensor
#     LEFT JOIN Optos_table               l               ON l.ID                         = h.Opto
#     LEFT JOIN Promoter_table            m               ON m.ID                         = h.Promoter
#     LEFT JOIN Recombinase_table         n               ON n.ID                         = h.Recombinase
#     LEFT JOIN Sensors_table             o               ON o.ID                         = i.Sensor
#     LEFT JOIN Promoter_table            p               ON p.ID                         = i.Promoter
#     LEFT JOIN Recombinase_table         q               ON q.ID                         = i.Recombinase
#     LEFT JOIN Optos_table               r               ON r.ID                         = j.Opto
#     LEFT JOIN Promoter_table            s               ON s.ID                         = j.Promoter
#     LEFT JOIN Recombinase_table         t               ON t.ID                         = j.Recombinase
#     LEFT JOIN Genotypes_table           AS G2C_table    ON f.G2C                        = G2C_table.ID
#     LEFT JOIN Genotypes_table           AS Ai14_table   ON f.Ai14                       = Ai14_table.ID
#     LEFT JOIN Genotypes_table           AS Ai75_table   ON f.Ai75                       = Ai75_table.ID
#     LEFT JOIN Genotypes_table           AS VRC_table    ON f.VRC                        = VRC_table.ID
#     LEFT JOIN Genotypes_table           AS SLF_table    ON f.SLF                        = SLF_table.ID
#     LEFT JOIN Genotypes_table           AS PVF_table    ON f.PVF                        = PVF_table.ID
#     LEFT JOIN Genotypes_table           AS Ai65_table   ON f.Ai65                       = Ai65_table.ID
#     LEFT JOIN Genotypes_table           AS Ai80_table   ON f.Ai80                       = Ai80_table.ID
#     LEFT JOIN Genotypes_table           AS VGC_table    ON f.VGC                        = VGC_table.ID
#     LEFT JOIN Genotypes_table           AS Ai162_table  ON f.Ai162                      = Ai162_table.ID
#     LEFT JOIN Genotypes_table           AS Ai148_table  ON f.Ai148                      = Ai148_table.ID          
#     LEFT JOIN Sex_table                                 ON Sex_table.ID                 = f.Sex
#     LEFT JOIN Lines_table               v               ON v.ID                         = f.Line 
    
    
    
#     LEFT JOIN Genotypings_table         aa ON   aa.ID   =   f.Genotyping_Status
#     LEFT JOIN Labels_table              bb ON   bb.ID   =   f.Label
#     LEFT JOIN Rooms_table               cc ON   cc.ID   =   f.Room

#     LEFT JOIN Experimental_table        dd ON   dd.ID   =   c.Experiment
#     LEFT JOIN ExperimentalStatus_table  ee ON   ee.ID   =   c.Experimental_status
    
#     LEFT JOIN Brain_Areas_table         ff ON   ff.ID   =   d.CorticalArea
#     LEFT JOIN Brain_Areas_table         nn ON   nn.ID   =   e.CorticalArea

    
#     LEFT JOIN CoverSize_table           gg ON   gg.ID   =   d.CoverSize
#     LEFT JOIN Covertype_table           hh ON   hh.ID   =   d.CoverType
#     LEFT JOIN Craniosize_table          ii ON   ii.ID   =   d.CranioSize
#     LEFT JOIN WindowTypes_table         jj ON   jj.ID   =   d.WindowType
#     LEFT JOIN Sterocoordinates_table    kk ON   kk.ID   =   d.HeadPlateCoordinates
#     LEFT JOIN Sterocoordinates_table    oo ON   oo.ID   =   e.InjectionSite1Coordinates
#     LEFT JOIN Sterocoordinates_table    pp ON   pp.ID   =   e.InjectionSite2Coordinates

  
#     LEFT JOIN Microscopes_table         ll ON   ll.ID   =   b.Microscope
    
#     LEFT JOIN Serotypes_table           qq ON   qq.ID   =   h.Serotype
#     LEFT JOIN Serotypes_table           rr ON   rr.ID   =   i.Serotype
#     LEFT JOIN Serotypes_table           ss ON   ss.ID   =   j.Serotype
    
#     WHERE c.Code = ?
#             """
# params=('SPJA',)  
# allinfo=MouseDat.arbitrary_query_to_df(query_all_imaged_mice, params)   


# age=pd.to_datetime(allinfo.ImagingDate) - pd.to_datetime(allinfo.DOB)
# wai=pd.to_datetime(allinfo.ImagingDate) - pd.to_datetime(allinfo.InjectionDate)
# waw=pd.to_datetime(allinfo.ImagingDate) - pd.to_datetime(allinfo.WindowDate)
# age=age.dt.days/7
# wai=wai.dt.days/7
# waw=waw.dt.days/7
# allinfo.insert(8, 'Age',  age.round(1))
# allinfo.insert(10, 'WAI',   wai.round(1))
# allinfo.insert(12, 'WAW',   waw.round(1))       
# allinfo.Combination.fillna(value='NoInjection', inplace=True)
# allinfo.Sensors1.fillna(value='NoInjection', inplace=True)
# allinfo.Optos1.fillna(value='NoInjection', inplace=True)
# allinfo.Sensors2.fillna(value='NoInjection', inplace=True)
# allinfo.Recombinases1.fillna(value='NoInjection', inplace=True)
# allinfo.Recombinases2.fillna(value='NoInjection', inplace=True)
# allinfo.Recombinases3.fillna(value='NoInjection', inplace=True)
# allinfo.Optos3.fillna(value='NoInjection', inplace=True) 
# all_mouse_info=allinfo.copy(0) 
# all_mouse_info.drop_duplicates(subset ="Code",keep='first', inplace=True)  
        

#         # add sessios