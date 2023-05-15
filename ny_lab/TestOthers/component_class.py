#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 20:03:34 2022

@author: samuel
"""

from __future__ import annotations
import webbrowser
from dataclasses import dataclass, field
from typing import List
import os
from sqlalchemy import Column, ForeignKey, Integer, String, create_engine, Float
from sqlalchemy.orm import registry, relationship, Session
if os.path.isfile('components.db'):
    os.remove('components.db')
mapper_registry = registry()
engine = create_engine('sqlite:///components.db', echo=True, future=True)


@mapper_registry.mapped
@dataclass
class Component:
    __tablename__ = "component"

    __sa_dataclass_metadata_key__ = "sa"
    id: int = field(init=False, metadata={"sa": Column(Integer, primary_key=True)})
    name: str = field(default=None, metadata={"sa": Column(String)})
    lotnumber: str = field(default=None, metadata={"sa": Column(String)})
    company: str = field(default=None, metadata={"sa": Column(String)})
    price: str = field(default=None, metadata={"sa": Column(Float)})
    component: str = field(default=None, metadata={"sa": Column(String)})
    comments: str = field(default=None, metadata={"sa": Column(String)})




    urls: List[Url] = field(
        default_factory=list, metadata={"sa": relationship("Url")}
    )
    
    def open_urls(self):
        for url in self.urls:
            url.open()
    

@mapper_registry.mapped
@dataclass
class Url:
    __tablename__ = "url"
    __sa_dataclass_metadata_key__ = "sa"
    id: int = field(init=False, metadata={"sa": Column(Integer, primary_key=True)})
    component_id: int = field(init=False, metadata={"sa": Column(ForeignKey("component.id"))})
    url_link: str = field(default=None, metadata={"sa": Column(String)})
    
    
    def open(self):
        if isinstance(self.url_link,str):
            webbrowser.open_new_tab(self.url_link)

mapper_registry.metadata.create_all(engine)

#%% inserting
partlist=r'C:\Users\sp3660\Documents\Projects\LabNY\6. Microscopy\Widefield Opto Amsterdam\Widefield Macroscope and Hhigh Speed Scanning Stimulation Part List.xlsx'

import pandas as pd

df = pd.read_excel (partlist)

thorlab_components=df[df['Unnamed: 4']=='Thorlabs']  

clean_df=df.dropna(subset = ['Unnamed: 4'])



with Session(engine) as session:
    
    for index, comp in clean_df.iloc[1:].iterrows():

        cage = Component(
            name=comp['Unnamed: 2'],
            lotnumber=comp['Unnamed: 3'],
            company=comp['Unnamed: 4'],
            price=comp['Unnamed: 5'],
            component=comp['Unnamed: 6'],
            comments=comp['Unnamed: 7'],
            urls=[
                Url(url_link=comp['Unnamed: 8'])
            ]
        )
        
        session.add_all([cage])
    session.commit()


#%% querying
from sqlalchemy import select


# stmt = select(Component)
# result=Session(engine).execute(stmt)
# cage=result.fetchone()[0]
# cage.open_urls()

stmt = select(Component).join(Url)
with Session(engine) as session:
    result=session.execute(stmt)
    for i,row in enumerate(result):
        row[0].urls[0].open()



