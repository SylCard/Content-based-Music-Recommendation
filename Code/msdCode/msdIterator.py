#imports
import os
import sys
import time
import glob
import datetime
import numpy as np
import hdf5_getters as hdf5getter
import sqlite3

# path to million song dataset
msd_path = '../../MillionSongSubset/'
msd_data_path = os.path.join(msd_path,'data')

# this function will iterate through each file in the dataset
def apply_to_all_files(basedir,func=lambda x: x, extension='.h5'):
    count = 0
    # iterate over all files in all subdirectories of the base direcotry
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root,'*'+extension))
        # count files
        count += len(files)
        # apply function to all files
        for f in files :
            func(f)
    return count


#time to get the information we need from the database


def func_fetch_song_details(filename):
    """
    This function does the following things:
    - open the database connection
    - open the song file
    - get info we need and store it in the database
    - close the file
    - close the database connection
    """
    db = sqlite3.connect('songdb')
    db.text_factory = str
    file = hdf5getter.open_h5_file_read(filename)

    track_id = str(hdf5getter.get_track_id(file))
    artist_name = str(hdf5getter.get_artist_name(file))
    song_name = str(hdf5getter.get_title(file))
    digital_id = str(hdf5getter.get_track_7digitalid(file))
    print song_name
    cursor = db.cursor()
    cursor.execute('''INSERT INTO songs(trackID,artist,song_title,digitalID) VALUES(?,?,?,?)''', (track_id,artist_name, song_name, digital_id))
    db.commit()

    file.close()
    db.close()

apply_to_all_files(msd_data_path,func=func_fetch_song_details)
