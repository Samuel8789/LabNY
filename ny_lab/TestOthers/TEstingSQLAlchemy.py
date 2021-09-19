# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 11:11:32 2021

@author: sp3660
"""
from sqlalchemy import create_engine, MetaData, Table, join, text, select
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import mapper, sessionmaker
import sqlalchemy as sqla
from sqlalchemy.orm import column_property
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy.orm import aliased
# c=MouseDat.database_connection.cursor()


# c.execute(
#     """
# CREATE TABLE OptogeneticsProtocols_table (
#     ID INTEGER PRIMARY KEY AUTOINCREMENT 
#     );
# """
# )
# c.execute(
#     """
# CREATE TABLE Optogenetics_table (
#     ID INTEGER PRIMARY KEY AUTOINCREMENT 
#     );
# """
# )




#%%
databasefile=r'C:\Users\sp3660\Documents\Projects\LabNY\4. Mouse Managing\MouseDatabase.db'
dbPath = databasefile
engine = create_engine('sqlite:///%s' % dbPath, echo=True)
metadata = MetaData(engine)
MICE_table = Table('MICE_table', metadata, autoload=True)
ExperimentalAnimals_table = Table('ExperimentalAnimals_table', metadata, autoload=True)


#%% inserting
ins = MICE_table.insert()
str(ins)
ins2 =MICE_table.insert().values(ID=15, Lab_Number = 'Karan')
str(ins2)
ins2.compile().params


conn = engine.connect()
result = conn.execute(ins3)
result.inserted_primary_key
#%% selecting rows

s = MICE_table.select()
str(s)
conn = engine.connect()
result = conn.execute(s)
row = result.fetchone()
for row in result:
    print (row)

s = MICE_table.select().where(MICE_table.c.Cage!=None)
str(s)
result = conn.execute(s)
for row in result:
    print (row)
   
t = text("SELECT * FROM MICE_table")
result = conn.execute(t)
for row in result:
    print (row)
   
s = text("select MICE_table.ID MICE_table.Cage from MICE_table where MICE_table.cage between :x and :y")
conn.execute(s, x = 220, y = 230).fetchall()

s = select([ExperimentalAnimals_table, MICE_table]).where(ExperimentalAnimals_table.c.Mouse_ID == MICE_table.c.ID)
result = conn.execute(s)

for row in result:
    zzz=row
   
j = ExperimentalAnimals_table.join(MICE_table)
print(ExperimentalAnimals_table.join(MICE_table))
stmt = select([ExperimentalAnimals_table.c.ID, MICE_table]).select_from(j)
str(stmt)
result = conn.execute(stmt)
result.fetchall()
#%%
Base = declarative_base()
class Mouse(Base):
    __tablename__ = MICE_table
    #%%
Session = sessionmaker(bind = engine)
session = Session()

q = session.query(mapped class)
q = Query(mappedClass, session)
result = session.query(Customers).all()


#%%
class Mouse(Base):
    __table__ = MICE_table


Base = automap_base()
Base.prepare(engine, reflect=True)
Mouse = Base.classes.MICE_table
Session = sessionmaker(bind = engine)
session = Session()
result = session.query(Mouse).all()

result[0].print_some

#%%
Base = declarative_base()
databasefile=r'C:\Users\sp3660\Documents\Projects\LabNY\4. Mouse Managing\MouseDatabase.db'

class Mouse(Base):
    __table__ = Table('mytable', Base.metadata,
                    autoload=True, autoload_with=some_engine)
    
    
    pass
class PrairieImagingSession(object):
    pass
class MouseImagingSession(object):
    pass
class Acquisitions(object):
    pass
class Datasets(object):
    pass
class FaceVideo(object):
    pass
class FOV(object):
    pass
class WideField(object):
    pass
class VisualStimulation(object):
    pass
class ImagedMouse(object):
    pass
class ExperimentalMouse(object):
    pass
class Injections(object):
    pass
class Windows(object):
    pass
#----------------------------------------------------------------------
""""""    





    #%%
databasefile=r'C:\Users\sp3660\Documents\Projects\LabNY\4. Mouse Managing\MouseDatabase.db'
dbPath = databasefile
engine = create_engine('sqlite:///%s' % dbPath, echo=True)

metadata = MetaData(engine)
Base = declarative_base() 

MICE_table = Table('MICE_table', metadata, autoload=True)
ExperimentalAnimals_table = Table('ExperimentalAnimals_table', metadata, autoload=True)
ImagedMice_table = Table('ImagedMice_table', metadata, autoload=True)
Acquisitions_table = Table('Acquisitions_table', metadata, autoload=True)
FaceCamera_table = Table('FaceCamera_table', metadata, autoload=True)
Imaging_table = Table('Imaging_table', metadata, autoload=True)
ImagingSessions_table = Table('ImagingSessions_table', metadata, autoload=True)
Injections_table = Table('Injections_table', metadata, autoload=True)
WideField_table = Table('WideField_table', metadata, autoload=True)
Windows_table = Table('Windows_table', metadata, autoload=True)
VisualStimulations_table = Table('VisualStimulations_table', metadata, autoload=True)
Genotypes_table = Table('Genotypes_table', metadata, autoload=True)
Ai14_genotypes = aliased(Genotypes_table)
Ai65_genotypes = aliased(Genotypes_table)
Ai80_genotypes = aliased(Genotypes_table)
Ai75_genotypes = aliased(Genotypes_table)
Ai162_genotypes = aliased(Genotypes_table)
Ai148_genotypes = aliased(Genotypes_table)
SLF_genotypes = aliased(Genotypes_table)
VRC_genotypes= aliased(Genotypes_table)
PVF_genotypes = aliased(Genotypes_table)
VGC_genotypes = aliased(Genotypes_table)
G2C_genotypes = aliased(Genotypes_table)



Session = sessionmaker(bind=engine)
session = Session()
all_mice_table=sqla.join(MICE_table, Ai14_genotypes, onclause=MICE_table.c.Ai14)

class Mouse(Base):
    __table__ =Table('MICE_table', metadata, autoload=True)


AllMiceObjects = session.query(Mouse).all()
dir(AllMiceObjects[0])

len(AllMiceObjects)
z=[zz.Code for zz in AllMiceObjects]


#%%

from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

Base = automap_base()

# engine, suppose it has two tables 'user' and 'address' set up
databasefile=r'C:\Users\sp3660\Documents\Projects\LabNY\4. Mouse Managing\MouseDatabase.db'
dbPath = databasefile
engine = create_engine('sqlite:///%s' % dbPath, echo=True)

# reflect the tables
Base.prepare(engine, reflect=True)
Base.classes.keys()
# mapped classes are now created with names by default
# matching that of the table name.
Mouse = Base.classes.MICE_table


session = Session(engine)
zz= session.query(Mouse).all()
zzz=zz[2600]
