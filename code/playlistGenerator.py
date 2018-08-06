import numpy as np
from sklearn.externals import joblib
from math import *
import operator

songLibrary = {}
counter = 0
# 1. match predictions to corresponding songs
# predictions = joblib.load('UserTestSongs.prediction')
predictions = joblib.load('predictions.data')
rockPredictions = joblib.load('extraRock.prediction')
userPredictions = joblib.load('UserChosenSongs.prediction')
with open('songTitles.txt') as f:
   for line  in f:
       songLibrary[line.strip('\n')] = predictions[counter]
       counter += 1

# add incubus + billy Talent
rockCounter = 0
with open('incubus.txt') as f:
   for line  in f:
       songLibrary[line.strip('\n')] = rockPredictions[rockCounter]
       rockCounter += 1
 # add user chosen songs
# userCounter = 0
# with open('humanTestSongOrder.txt') as f:
#     for line  in f:
#         songLibrary[line.strip('\n')] = userPredictions[userCounter]
#         userCounter += 1
# print songLibrary
# 2. choose a query song then use similarity algorithm to return top ten similar songs
querySong = "Cocoa Butter Kisses (ft Vic Mensa & Twista) (Prod by Cam for JUSTICE League & Peter Cottont (DatPiff Exclusive)"

querySongData = songLibrary[querySong]

del songLibrary[querySong]
# del songLibrary['Big Sean - How It Feel (Lyrics)']
# del songLibrary['The Game - Ali Bomaye (Explicit) ft. 2 Chainz, Rick Ross']
# del songLibrary['Kendrick Lamar - Money Trees (HD Lyrics)']
# del songLibrary['Faint (Official Video) - Linkin Park']
# del songLibrary['Wale-Miami Nights (Ambition)']
# del songLibrary['Wale - Bad Girls Club Ft. J Cole Official Video']
# 3. find top 10 closest songs
topSongs = {}

for key, value in songLibrary.iteritems():
    # calculate distance
    dist = np.linalg.norm(querySongData-songLibrary[key])
    # store in distance directory
    topSongs[key] = dist

# order top songs by distance
sortedSongs = sorted(topSongs.items(), key=operator.itemgetter(1))
# take top 10 closest songs
sortedSongs = sortedSongs[:10]

for value in sortedSongs:
    print value


# for visualisation get coordinates of top 10 songs
topSongDistances = {}
for val in sortedSongs:

    topSongDistances[val[0]] = songLibrary[val[0]]

topSongDistances[querySong] = querySongData

# print topSongDistances

# joblib.dump(topSongDistances, 'kanyeHeartless.playlist')
