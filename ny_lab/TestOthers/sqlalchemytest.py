# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 18:04:33 2022

@author: sp3660
"""
import sqlalchemy
import sqlite3
from sqlite3 import Error
sqlalchemy.__version__ 
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import  declarative_base, sessionmaker



# NEW_DB_NAME = 'database_name'

# with engine.connect() as conn:
#     conn.execute("commit")
#     # Do not substitute user-supplied database names here.
#     conn.execute(f"CREATE DATABASE {NEW_DB_NAME}")



#%% creating

engine = create_engine('sqlite:///foo.db', echo=True)
Base = declarative_base()


class Laboratory(Base):
    __tablename__ = 'labs'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    institution = Column(String)
    pi_name = Column(String)

    def __repr__(self):
       return "<User(name='%s', institution='%s', pi_name='%s')>" % (
                            self.name, self.institution, self.pi_name)
   
    def create_row():
        
        
    def load_row()
Base.metadata.create_all(engine)

ed_user = Laboratory(name='Yuste Lab', institution='Columbia University', pi_name='Rafael Yuste')


Session = sessionmaker(bind=engine)
session = Session()
session.add(ed_user)

our_user = session.query(Laboratory).filter_by(name='Yuste Lab').first() 

ed_user is our_user


session.add_all([
     Laboratory(name='Losonczy Lab', institution='Columbia University', pi_name='Attila Losonczy'),
     Laboratory(name='Paninski Lab', institution='Columbia University', pi_name='Liam Paninski'),
     Laboratory(name='Shadlen Lab', institution='Columbia University', pi_name='Michael N. Shadlen')])

ed_user.pi_name = 'Rafa'
session.dirty
session.new
ed_user.pi_name = 'Rafael Yuste'

session.commit()
ed_user.id 


for instance in session.query(Laboratory).order_by(Laboratory.id):
  print(instance.name, instance.pi_name)

for name, pi_name in session.query(Laboratory.name, Laboratory.pi_name):
  print(name, pi_name)
  
  
for row in session.query(Laboratory, Laboratory.name).all():
    print(row.Laboratory, row.name)
    #%% loading
    
    engine = create_engine('sqlite:///foo.db', echo=True)
    Base = declarative_base()

    class Laboratory(Base):
        __tablename__ = 'labs'

        id = Column(Integer, primary_key=True)
        name = Column(String)
        institution = Column(String)
        pi_name = Column(String)

        def __repr__(self):
           return "<User(name='%s', institution='%s', pi_name='%s')>" % (
                                self.name, self.institution, self.pi_name)
Session = sessionmaker(bind=engine)
session = Session()

for instance in session.query(Laboratory).order_by(Laboratory.id):
  print(instance.name, instance.pi_name)
#%%
import sqlalchemy as db


dbpath=os.path.join(r'C:\Users\sp3660\Dropbox\Projects','LabNY', 'MouseDatabase.db')

engine = db.create_engine('sqlite:///'+dbpath)
connection = engine.connect()
metadata = db.MetaData()
mice_table = db.Table('MICE_table', metadata, autoload=True, autoload_with=engine)
print(mice_table.columns.keys())

print(repr(metadata.tables['MICE_table']))

query = db.select([census]) 


class Database_Mouse(Base):
    __tablename__ = 'MICE_table'

    id = Column(Integer, primary_key=True)
    Lab_Number = Column(Integer)
    Cage = Column(String)
    DOB = Column(String)

    def __repr__(self):
       return "<User(name='%s', institution='%s', pi_name='%s')>" % (
                            self.name, self.institution, self.pi_name)