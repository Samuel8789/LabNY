# -*- coding: utf-8 -*-
"""
Created on Sun May 22 21:27:59 2022

@author: Samuel
"""
import twint
import nest_asyncio
import pandas as pd


nest_asyncio.apply()

c = twint.Config()

c.Search = ['neuromatch']       # topic
c.Limit = 500      # number of Tweets to scrape
c.Store_csv = True       # store tweets in a csv file
c.Output = "neuromatch.csv"     # path to csv file

twint.run.Search(c)

df = pd.read_csv('neuromatch.csv')

tweets=df['tweet']


c.Search = "from:@yusterafa"
c.Store_csv = True       # store tweets in a csv file
c.Output = "rafa.csv"     # path to csv file
twint.run.Search(c)

df = pd.read_csv('rafa.csv')
tweets=df['tweet']

