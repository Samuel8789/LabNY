# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 19:37:04 2022

@author: Samuel
"""
from mendeley import Mendeley
import webbrowser
import requests
import os
import glob
import json

client_id=12539
client_secret='yY1Yzl0WMB7Ofzs3'
redirect_uri='http://localhost:8888/'


# These values should match the ones supplied when registering your application.
mendeley = Mendeley(client_id, redirect_uri=redirect_uri)

auth = mendeley.start_implicit_grant_flow()

# The user needs to visit this URL, and log in to Mendeley.
login_url = auth.get_login_url()

res = requests.post(login_url, allow_redirects = False, data = {
    'username': 'samuelpontesquero@gmail.com',
    'password': 'Angry223'
})

auth_response = res.headers['Location']

# After logging in, the user will be redirected to a URL, auth_response.
session = auth.authenticate(auth_response)
#%%
# tets=session.documents.advanced_search(title='CaI')
# paper=tets.list().items[0]
# paperfile=paper.files.list().items[0]

# webbrowser.open(paperfile.download_url)
# paperfile.download()
# firstauto=paper.authors[0]

# firstauto.last_name+' et al'
# paper.source
# paperfile.file_name
# 'Pontes-Quero et al. - 2019 - High mitogenic stimulation arrests angiogenesis'

# test1=paperfile.file_name.replace('___', ' - ')
# test2=test1.replace('_et_al', ' et al')


# pathtofile=r'C:\Users\Samuel\Documents\Mendeley Desktop'

# filepath=os.path.join(pathtofile,firstauto.last_name+' et al',paper.source)

# os.startfile(glob.glob(filepath+'\*')[0])
# os.startfile(filepath)


folder='Remarkable'
folders=[]
urll="https://api.mendeley.com/folders?limit={}"
uri=urll.format(500)
rsp=session.request("GET",uri)

for f in json.loads(rsp.content) :
    parent_id=f['parent_id'] if 'parent_id' in f else None
    if f['name']=='Remarkable':
        folders.append((f['id'],f['name'], parent_id))
    
    
documents_ids=[]
docsurl="https://api.mendeley.com/folders/{}/documents?limit={}"
uri=docsurl.format(folders[0][0],500)
rsp=session.request("GET",uri)

for f in json.loads(rsp.content) :
    documents_ids.append(f['id'])
    
paper=session.documents.get(documents_ids[0])
paperfile=paper.files.list().items[0]
test=paperfile.download(r'C:\Users\Samuel\Downloads')
os.startfile(test)




    
    