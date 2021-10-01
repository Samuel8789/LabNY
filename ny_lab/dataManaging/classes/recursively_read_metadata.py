# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 10:12:02 2021

@author: sp3660
"""

def recursively_read_metadata(root):    
    metadata={}
    if list(root):
        for key in root.attrib.keys():
            metadata[key]=root.attrib[key] 
        for i, element in enumerate(root):
            if not list(element):         
                if element.tag not in metadata.keys():
                    tag_lol=element.tag
                else:
                    tag_lol= element.tag+'_'+str(i) 
                metadata[tag_lol]={}
                for key in element.attrib.keys():
                    metadata[tag_lol][key]=element.attrib[key]  
            else:
                if element.tag not in metadata.keys():
                    tag_lol=element.tag
                else:
                    tag_lol=element.tag +'_'+str(i) 
                metadata[tag_lol]={}
                for key in element.attrib.keys():
                    metadata[tag_lol][key]=element.attrib[key]   
                metadata[tag_lol]['Childs']={}
                for jj, child in enumerate(element):
                    if child.tag in  metadata[tag_lol]['Childs'].keys():
                        tag= child.tag+'_'+str(jj)    
                    else:
                        tag=child.tag        
                        # these tagas include the multiple PVstatevalues
                    metadata[tag_lol]['Childs'][tag]=recursively_read_metadata(child)
                    if not  metadata[tag_lol]['Childs'][tag]:
                        metadata[tag_lol]['Childs'][tag].pop(tag, None)
            if not  metadata[tag_lol]:
                metadata[tag_lol].pop(tag, None)

    else:      
        for key in root.attrib.keys():
            metadata[key]=root.attrib[key] 
 
    return metadata


