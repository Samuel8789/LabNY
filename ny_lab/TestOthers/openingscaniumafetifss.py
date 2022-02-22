# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 18:03:19 2022

@author: sp3660
"""

from ScanImageTiffReader import ScanImageTiffReader
import json
tifpath=r'F:\Projects\LabNY\Imaging\2022\20220210 Hakim\Mice\SPIK\Widefield\20220212_SPIK_4xHakimTest_00001_00001.tif'

vol=ScanImageTiffReader(tifpath).data();
meta=ScanImageTiffReader(tifpath).metadata();
desc=ScanImageTiffReader(tifpath).description(0);

with ScanImageTiffReader(tifpath) as reader:
        o=json.loads(reader.metadata())
        print(o["RoiGroups"]["imagingRoiGroup"]["rois"]["scanfields"]["affine"])
        
        
o=json.loads(meta)

parameter='SI.hDisplay.displayRollingAverageFactor = '
paddedstring=meta[meta.find(parameter)+len(parameter):meta.find(parameter)+len(parameter)+20]
value=paddedstring[:paddedstring.find('\n')]


parameters=['SI.hBeams.powers ', 'SI.hDisplay.displayRollingAverageFactor', 'SI.hRoiManager.linesPerFrame','SI.hRoiManager.pixelsPerLine','SI.hRoiManager.scanFramePeriod',
            'SI.hRoiManager.scanFrameRate', 'SI.hRoiManager.linePeriod']