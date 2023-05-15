# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 15:14:15 2023

@author: sp3660
"""
import sys
import glob

sys.path.insert(
    0, r"C:\Users\sp3660\Documents\Github\LabNY\ny_lab\dataManaging\classes\standalone"
)
sys.path.insert(
    0, r"C:\Users\sp3660\Documents\Github\LabNY\ny_lab\dataManaging\functions"
)

from functionsDataOrganization import (
    move_all_files_from_source_to_dest,
    check_channels_and_planes,
)


sorce = r"F:\Projects\LabNY\Imaging\2022\20221021\Mice\SPNN\FOV_1\Aq_1\221021_SPNN_FOV1_AllenA_25x_940_51020_60745_with-000\Ch2Green\plane1"
dest = r"F:\Projects\LabNY\Imaging\2022\20221021\Mice\SPNN\FOV_1\Aq_1\221021_SPNN_FOV1_AllenA_25x_940_51020_60745_with-000"
move_all_files_from_source_to_dest(sorce, dest)


# %%

[
    ChannelRedExists,
    ChannelGreenExists,
    RedFrameNumber,
    RedFirstFrame,
    RedLastFrame,
    GreenFrameNumber,
    GreenFirstFrame,
    GreenLastFrame,
    Multiplane,
    RedPlaneNumber,
    GreenPlaneNumber,
    FirstRedPlane,
    LastRedPlane,
    FirstGreenPlane,
    LastGreenPlane,
    sq_type,
] = check_channels_and_planes(dest)

# %%
from pyometiff import OMETIFFReader
import pathlib

alltiffs = glob.glob(dest + "\**.tif")

img_fpath = pathlib.Path(alltiffs[0])

reader = OMETIFFReader(fpath=img_fpath)

img_array, metadata, xml_metadata = reader.read()


with open("xmltest", "w") as f:
    f.write(xml_metadata)

    metadata["SizeZ"] = 3
    metadata["SizeC"] = 1


from PIL import Image

im = Image.open(img_fpath)
imageDesc = im.tag[270][0]

import xml.etree.ElementTree as ET

root = ET.fromstring(imageDesc)

for child in root:
    print(child.attrib)
