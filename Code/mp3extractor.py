#imports
import os
import sys
import time
import glob
import datetime
import numpy as np
import hdf5_getters as hdf5getter
import sqlite3
import py7digital as py7d

"""the aim of this file is to take the information from the database
and then obtain the mp3 file and genre of a given song
"""

#get the music

db = sqlite3.connect('songdb')
db.text_factory = str
cursor = db.cursor()
cursor.execute('''SELECT digitalID FROM songs''')

listOf7dIds =[r[0] for r in cursor.fetchall()]

print listOf7dIds[20]

track = py7d.Track(listOf7dIds[20]).get_preview()

print track
