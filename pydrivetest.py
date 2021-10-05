#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 11:13:08 2021

@author: samuel
"""

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
GoogleAuth.DEFAULT_SETTINGS['client_config_file'] = '/home/samuel/Documents/Github/LabNY/client_secret_381856385662-06l788l29tneq16ge1orf5jk1p2q4vn8.apps.googleusercontent.com.json'
gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)
file_list = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()


