# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 13:45:32 2021

@author: sp3660
"""

import pandas as pd
import itertools

def get_combination_from_virus(virus_tuple, database_object):

    if not virus_tuple[1]:
        virus_tuple=tuple(virus_tuple[0])
    database_connection=database_object.database_connection

    query_virus="SELECT ID FROM Virus_table WHERE VirusCode=?"

    viruscodes=[pd.read_sql_query(query_virus, database_connection, params=(vir,)).ID[0] for vir in virus_tuple if vir]
    
    good_virus_codes=[int(i) for i in viruscodes]

    if len(good_virus_codes)==1:
        query_viruscomb="SELECT ID FROM VirusCombinations_table WHERE Virus1=? AND Virus2 IS NULL AND Virus3 IS NULL"
        viruscomb = pd.read_sql_query(query_viruscomb, database_connection, params=(good_virus_codes[0],)).values.tolist()
        
    if len(good_virus_codes)==2:
        if any(x in virus_tuple for x in ['B', 'C1V1', 'I']):
            query_viruscomb="SELECT ID FROM VirusCombinations_table WHERE Virus1=? AND Virus2 IS NULL AND Virus3=?"
            viruscomb = pd.read_sql_query(query_viruscomb, database_connection, params=(good_virus_codes[0],good_virus_codes[1])).values.tolist()
        else:
            query_viruscomb="SELECT ID FROM VirusCombinations_table WHERE Virus1=? AND Virus2=? AND Virus3 IS NULL"
            viruscomb = pd.read_sql_query(query_viruscomb, database_connection, params=(good_virus_codes[0],good_virus_codes[1])).values.tolist() 
    
    if len(good_virus_codes)==3:
        query_viruscomb="SELECT ID FROM VirusCombinations_table WHERE Virus1=? AND Virus2=? AND Virus3=?"
        viruscomb = pd.read_sql_query(query_viruscomb, database_connection, params=(good_virus_codes[0],good_virus_codes[1],good_virus_codes[2])).values.tolist()
        
    return viruscomb

def get_next_twolettercode(last_code):

    last=last_code[-2:]
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    test=[]
    n = 2
    for i in range(1, n+1):
        for item in itertools.product(chars, repeat=i):
            test.append("".join(item))
            
    next_code='SP'+test[test.index(last)+1]
    return next_code

def transform_earlabels_to_codes(labels_tuple, database_object):
    
    database_connection=database_object.database_connection
    query_labels="SELECT * FROM Labels_table WHERE Labels_types=?"
    labels_tuple=list(labels_tuple)
    for i, label in enumerate(labels_tuple):
            try :
                int(label)
                labels_tuple[i]=int(label)
            except:
                continue
    


    labels=[pd.read_sql_query(query_labels, database_connection,params=(label,)).values.tolist()[0 ]if isinstance(label, str) else label for label in labels_tuple ]
    test= [lable[0] if isinstance(lable,list) else lable for lable in labels]
    # label_codes=list(zip(*test)) [0]
    label_codes=test

    return label_codes

def transform_filterinfo_to_codes(filtervalues, database_object):
    
    query_red_filters="SELECT ID  FROM RedFilters_table WHERE FilterName=?"
    params1=(filtervalues[0],)
    red_filter_code=database_object.arbitrary_query_to_df(query_red_filters, params1).iloc[0]['ID']
    
    query_green_filters="SELECT ID  FROM GreenFilters_table WHERE FilterName=?"
    params2=(filtervalues[1],)
    green_filter_code=database_object.arbitrary_query_to_df(query_green_filters, params2).iloc[0]['ID']
    
    query_dichroic_filters="SELECT ID  FROM DichroicBeamSplitters_table WHERE DichroicBeamSplitter=?"
    params3=(filtervalues[2],)
    dichroic_code=database_object.arbitrary_query_to_df(query_dichroic_filters, params3).iloc[0]['ID']

 
    filter_codes=[red_filter_code,green_filter_code,dichroic_code]

    return filter_codes