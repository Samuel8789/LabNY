# -*- coding: utf-8 -*-
"""
Created on Thu May 20 08:39:43 2021

@author: sp3660
"""


import pandas as pd
import numpy as np
import sqlite3
from sqlite3 import Error


excelfile=r'C:\Users\sp3660\Desktop\ImagingDatabaseCreation.xlsx'
databasefile=r'C:\Users\sp3660\Documents\Projects\LabNY\4. Mouse Managing\MouseDatabase.db'

#%%
ImagingSessions_table = pd.read_excel(
    excelfile, 
    sheet_name='ImagingSessions',
    header=0)
ImagingSessions_table.dtypes

ImagedMice_table = pd.read_excel(
    excelfile, 
    sheet_name='ImagedMice',
    header=0)
ImagedMice_table.dtypes
ImagedMice_table['OptogeneticsProtocol'] = ImagedMice_table['OptogeneticsProtocol'].astype('Int64')
ImagedMice_table['WideFieldID'] = ImagedMice_table['WideFieldID'].astype('Int64')


WideField_table = pd.read_excel(
    excelfile, 
    sheet_name='WideField',
    header=0)
WideField_table.dtypes
WideField_table['ID'] = WideField_table['ID'].astype('Int64')


Acquisitions_table = pd.read_excel(
    excelfile, 
    sheet_name='Acquisitons',
    header=0)
Acquisitions_table.dtypes
Acquisitions_table['ID'] = Acquisitions_table['ID'].astype('Int64')
Acquisitions_table['ImagedMouseID'] = Acquisitions_table['ImagedMouseID'].astype('Int64')
Acquisitions_table['SampleID'] = Acquisitions_table['SampleID'].astype('Int64')
Acquisitions_table['ImagingID'] = Acquisitions_table['ImagingID'].astype('Int64')
Acquisitions_table['VisStimulationID'] = Acquisitions_table['VisStimulationID'].astype('Int64')
Acquisitions_table['FaceCameraID'] = Acquisitions_table['FaceCameraID'].astype('Int64')
Acquisitions_table['OptogeneticsID'] = Acquisitions_table['OptogeneticsID'].astype('Int64')



Imaging_table = pd.read_excel(
    excelfile, 
    sheet_name='Imaging',
    header=0)
Imaging_table.dtypes
Imaging_table['FOVNumber'] = Imaging_table['FOVNumber'].astype('Int64')
Imaging_table['IsETLStack'] = Imaging_table['IsETLStack'].astype('Int64')
Imaging_table['IsObjectiveStack'] = Imaging_table['IsObjectiveStack'].astype('Int64')
Imaging_table['IsChannel1Red'] = Imaging_table['IsChannel1Red'].astype('Int64')
Imaging_table['IsChannel2Green'] = Imaging_table['IsChannel2Green'].astype('Int64')
Imaging_table['PlaneNumber'] = Imaging_table['PlaneNumber'].astype('Int64')
Imaging_table['CoherentPower'] = Imaging_table['CoherentPower'].astype('Int64')
Imaging_table['PowerSetting'] = Imaging_table['PowerSetting'].astype('Int64')
Imaging_table['FrameAveraging'] = Imaging_table['FrameAveraging'].astype('Int64')
Imaging_table['TotalFrames'] = Imaging_table['TotalFrames'].astype('Int64')
Imaging_table['TotalVolumes'] = Imaging_table['TotalVolumes'].astype('Int64')
Imaging_table['IsVoltageRecording'] = Imaging_table['IsVoltageRecording'].astype('Int64')
Imaging_table['VoltageRecordingFrequency'] = Imaging_table['VoltageRecordingFrequency'].astype('Int64')




VisualStimulations_table = pd.read_excel(
    excelfile, 
    sheet_name='VisualStimulations',
    header=0)
VisualStimulations_table.dtypes
VisualStimulations_table['ID'] = VisualStimulations_table['ID'].astype('Int64')
VisualStimulations_table['AcquisitionID'] = VisualStimulations_table['AcquisitionID'].astype('Int64')
VisualStimulations_table['Behaviour'] = VisualStimulations_table['Behaviour'].astype('Int64')
VisualStimulations_table['IsMATLAB'] = VisualStimulations_table['IsMATLAB'].astype('Int64')
VisualStimulations_table['IsPython'] = VisualStimulations_table['IsPython'].astype('Int64')
VisualStimulations_table['IsWithSignal'] = VisualStimulations_table['IsWithSignal'].astype('Int64')


FaceCamera_table = pd.read_excel(
    excelfile, 
    sheet_name='FaceCamera',
    header=0)
FaceCamera_table.dtypes




CalibrationSamples_table = pd.read_excel(
    excelfile, 
    sheet_name='CalibrationSamples',
    header=0)
CalibrationSamples_table.dtypes

Microscopes_table = pd.read_excel(
    excelfile, 
    sheet_name='Microscopes',
    header=0)
Microscopes_table.dtypes

Behaviours_table = pd.read_excel(
    excelfile, 
    sheet_name='Behaviours',
    header=0)
Behaviours_table.dtypes

GreenFilters_table = pd.read_excel(
    excelfile, 
    sheet_name='GreenFilters',
    header=0)
GreenFilters_table.dtypes

RedFilters_table = pd.read_excel(
    excelfile, 
    sheet_name='RedFIlters',
    header=0)
RedFilters_table.dtypes

DichroicBeamSplitters_table = pd.read_excel(
    excelfile, 
    sheet_name='DichroicBeamSplitters',
    header=0)
DichroicBeamSplitters_table.dtypes
#%%
db_conn = sqlite3.connect(databasefile)
c = db_conn.cursor()
#%%
ImagingSessions_table.dtypes

c.execute(
    """
CREATE TABLE ImagingSessions_table (
             ID INTEGER PRIMARY KEY AUTOINCREMENT,
             ImagingDate TEXT,
             StartTime TEXT,
             Microscope INTEGER,
             ImagingSessionRawPath TEXT,
             CalibrationsRawPath TEXT,
             PowerCalPath TEXT,
             MecahincalZStackPath TEXT,
             ETLCalibrationsPath TEXT,
             AlignmentCalibrationsPath TEXT,
             MiceRawPath TEXT,
             Objectives TEXT,
             EndOfSessionSummary TEXT,
             IssuesDuringImaging TEXT,
             
             FOREIGN KEY (Microscope) REFERENCES Microscopes_table (ID)
     
    );     
"""
)


ImagedMice_table.dtypes

c.execute(
    """
CREATE TABLE ImagedMice_table (
                ID   INTEGER PRIMARY KEY AUTOINCREMENT,
                SessionID        INTEGER,
                ExpID                 INTEGER,
                TimeSetOnWheel            TEXT,
                MouseRawPath            TEXT,
                EyesComments                    TEXT,
                BehaviourComments                TEXT,
                FurComments                      TEXT,
                LessionComments                  TEXT,
                DexaInjection                     INTEGER,
                BehaviourProtocol                 INTEGER,
                BehaviourProtocolComments        TEXT,
                OptogeneticsProtocol            INTEGER,
                OptogeneticsProtocolComments    TEXT,
                Objectives                       TEXT,
                EndOfSessionSummary              TEXT,
                IssuesDuringImaging             TEXT,
                IsWideFIeld                     INTEGER,
                WideFieldID                     INTEGER,
                
                FOREIGN KEY (SessionID) REFERENCES ImagingSessions_table (ID)
                FOREIGN KEY (ExpID) REFERENCES ExperimentalAnimals_table (ID)
                FOREIGN KEY (BehaviourProtocol) REFERENCES Behaviours_table (ID)
                FOREIGN KEY (OptogeneticsProtocol) REFERENCES OptogeneticsProtocols_table (ID)
                FOREIGN KEY (WideFieldID) REFERENCES WideField_table (ID)


    
    );
"""
)

WideField_table.dtypes

c.execute(
    """
CREATE TABLE WideField_table (
                ID   INTEGER PRIMARY KEY AUTOINCREMENT,
                ImagedMouseID          INTEGER,
                WideFieldImagePath    TEXT,
                WideFieldFileName     TEXT,
                WideFieldComments     TEXT,
                
                FOREIGN KEY (ImagedMouseID) REFERENCES ImagedMice_table (ID)
                
    
    );
"""
)


Acquisitions_table.dtypes

c.execute(
    """
CREATE TABLE Acquisitions_table (
                ID   INTEGER PRIMARY KEY AUTOINCREMENT,
                IsMouse                     INTEGER,
                ImagedMouseID                INTEGER,
                SampleID                     INTEGER,
                AcquisitonRawPath           TEXT,
                IsCalibration                INTEGER,
                IsTestAcquisition            INTEGER,
                IsNonImagingAcquisition      INTEGER,
                Is0CoordinateAcquisiton      INTEGER,
                IsFOVAcquisition             INTEGER,
                AcquisitonNumber             INTEGER,
                AqTime                      TEXT,
                IsImaging                    INTEGER,
                IsFaceCamera                 INTEGER,
                IsLocomotion                 INTEGER,
                IsPhotodiode                 INTEGER,
                IsVisStimSignal              INTEGER,
                IsBehaviour                  INTEGER,
                IsVisualStimulation          INTEGER,
                IsOptogenetic                INTEGER,
                ImagingID                    INTEGER,
                VisStimulationID             INTEGER,
                FaceCameraID                 INTEGER,
                OptogeneticsID               INTEGER,
                Comments                   TEXT,
                
                FOREIGN KEY (ImagedMouseID) REFERENCES ImagedMice_table (ID)
                FOREIGN KEY (SampleID) REFERENCES CalibrationSamples_table (ID)
                FOREIGN KEY (ImagingID) REFERENCES Imaging_table (ID)
                FOREIGN KEY (VisStimulationID) REFERENCES VisualStimulations_table (ID)
                FOREIGN KEY (FaceCameraID) REFERENCES FaceCamera_table (ID)
                FOREIGN KEY (OptogeneticsID) REFERENCES Optogenetics_table (ID)
                    
    );
                
"""
)

Imaging_table.dtypes

c.execute(
    """
CREATE TABLE Imaging_table (
                ID   INTEGER PRIMARY KEY AUTOINCREMENT,
                AcquisitionID                  INTEGER,
                ImagingFullFilePath           TEXT,
                ImagingFilename               TEXT,
                RedFilter                      INTEGER,
                GreenFilter                    INTEGER,
                DichroicBeamsplitter           INTEGER,
                IsBlockingDichroic             INTEGER,
                FOVNumber                      INTEGER,
                IsETLStack                     INTEGER,
                IsObjectiveStack               INTEGER,
                PlaneNumber                    INTEGER,
                Objective                    REAL,
                ObjectivePositions           REAL,
                ETLPositions                REAL,
                PMT1GainRed                  REAL,
                PMT2GainGreen                REAL,
                IsChannel1Red                  INTEGER,
                IsChannel2Green                INTEGER,
                ExcitationWavelength         REAL,
                CoherentPower                  INTEGER,
                PowerSetting                   INTEGER,
                CalculatedPower              REAL,
                IsGalvo                        INTEGER,
                IsResonant                     INTEGER,
                Resolution                   REAL,
                DwellTime                    REAL,
                Multisampling                REAL,
                BitDepth                     REAL,
                FrameAveraging                 INTEGER,
                LinePeriod                   REAL,
                FramePeriod                  REAL,
                InterFramePeriod             REAL,
                FinalVolumePeriod            REAL,
                FinalFrequency               REAL,
                FullAcquisitionTime          REAL,
                TotalFrames                    INTEGER,
                TotalVolumes                   INTEGER,
                IsVoltageRecording             INTEGER,
                VoltageRecordingChannels     REAL,
                VoltageRecordingFrequency      INTEGER,
                Comments TEXT,
                
                FOREIGN KEY (AcquisitionID) REFERENCES Acquisitions_table (ID)
                FOREIGN KEY (RedFilter) REFERENCES RedFilters_table (ID)
                FOREIGN KEY (GreenFilter) REFERENCES GreenFilters_table (ID)
                FOREIGN KEY (DichroicBeamsplitter) REFERENCES DichroicBeamSplitters_table (ID)
                
                 
    );   
"""
)

VisualStimulations_table.dtypes

c.execute(
    """
CREATE TABLE VisualStimulations_table (
                ID   INTEGER PRIMARY KEY AUTOINCREMENT,
                AcquisitionID           INTEGER,
                Behaviour               INTEGER,
                Treatment              TEXT,
                VisStimSequence        TEXT,
                IsMATLAB                INTEGER,
                IsPython                INTEGER,
                VisStimLogPath         TEXT,
                VisStimLogName         TEXT,
                IsWithSignal            INTEGER,
                Comments              TEXT,
                SynchronizeMethods     TEXT,
                
                FOREIGN KEY (AcquisitionID) REFERENCES Acquisitions_table (ID)
                FOREIGN KEY (Behaviour) REFERENCES Behaviours_table (ID)
    
    );

"""
)
FaceCamera_table.dtypes

c.execute(
    """
CREATE TABLE FaceCamera_table (
                ID   INTEGER PRIMARY KEY AUTOINCREMENT,
                AcquisitionID           INTEGER,
                VideoPath              TEXT,
                EyeCameraFilename      TEXT,
                Exposure         REAL,
                Frequency             REAL,
                Resolution            TEXT,
                BitDepth              REAL,
                IsIRlight               INTEGER,
                IRLightPosition        TEXT,
                CameraPosition         TEXT,
                SideImaged             TEXT,
                VideoFormat           TEXT,
                SynchronizeMethods     TEXT,
                Comments               TEXT,
                
                FOREIGN KEY (AcquisitionID) REFERENCES Acquisitions_table (ID)
    
    );
"""
)


c.execute(
    """
CREATE TABLE CalibrationSamples_table (
                ID   INTEGER PRIMARY KEY AUTOINCREMENT,
                CalibrationSample TEXT
           
    );        
"""
)

c.execute(
    """
CREATE TABLE Microscopes_table (
                ID   INTEGER PRIMARY KEY AUTOINCREMENT,
                Microscope TEXT
         
    );          
"""
)

c.execute(
    """
CREATE TABLE Behaviours_table (
                ID   INTEGER PRIMARY KEY AUTOINCREMENT,
                Behaviour TEXT
         
    );          
"""
)

c.execute(
    """
CREATE TABLE GreenFilters_table (
                ID   INTEGER PRIMARY KEY AUTOINCREMENT,
                FilterName TEXT,
                Center INTEGER,           
                Width	INTEGER,   
                Min	INTEGER,   
                Max INTEGER

        
    );           
"""
)

c.execute(
    """
CREATE TABLE RedFilters_table (
                ID   INTEGER PRIMARY KEY AUTOINCREMENT,
                FilterName  TEXT,
                Center	INTEGER,   
                Width	INTEGER,   
                Min	INTEGER,   
                Max     INTEGER   
                    
    );
"""
)

c.execute(
    """
CREATE TABLE DichroicBeamSplitters_table (
                ID   INTEGER PRIMARY KEY AUTOINCREMENT,
                DichroicBeamSplitter TEXT
    
    );
"""
)






#%%



ImagingSessions_table.to_sql('ImagingSessions_table', db_conn, if_exists='append', index=False)
WideField_table.to_sql('WideField_table', db_conn, if_exists='append', index=False)
Acquisitions_table.to_sql('Acquisitions_table', db_conn, if_exists='append', index=False)
ImagedMice_table.to_sql('ImagedMice_table', db_conn, if_exists='append', index=False)

Imaging_table.to_sql('Imaging_table', db_conn, if_exists='append', index=False)
VisualStimulations_table.to_sql('VisualStimulations_table', db_conn, if_exists='append', index=False)
FaceCamera_table.to_sql('FaceCamera_table', db_conn, if_exists='append', index=False)
CalibrationSamples_table.to_sql('CalibrationSamples_table', db_conn, if_exists='append', index=False)
Microscopes_table.to_sql('Microscopes_table', db_conn, if_exists='append', index=False)
Behaviours_table.to_sql('Behaviours_table', db_conn, if_exists='append', index=False)
GreenFilters_table.to_sql('GreenFilters_table', db_conn, if_exists='append', index=False)
RedFilters_table.to_sql('RedFilters_table', db_conn, if_exists='append', index=False)
DichroicBeamSplitters_table.to_sql('DichroicBeamSplitters_table', db_conn, if_exists='append', index=False)





#%%
db_conn.close()

#%%