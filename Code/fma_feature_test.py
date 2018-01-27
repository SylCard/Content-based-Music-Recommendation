
#the aim of this file will be to traverse my dataset and output an array containing features for each track with corresponding labels
import glob
import os
import sys
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
# np.set_printoptions(threshold='nan')

genreDict = {
    'Pop'        :   0,
    'Rock'       :   1,
    'Hip-Hop'    :   2,
    'Folk'       :   3,
}

# this function will iterate through each file in the dataset
def extract_features(basedir,trackDict,extension='.mp3'):
    features=[]
    labels=[]
    counter = 0
    # iterate over all files in all subdirectories of the base directory
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root,'*'+extension))
        # apply function to all files
        for f in files :

            # Find genre - we're going to ignore International,Expiremental,Instrumental as they're too broad
            trackID = f.split('/')[4].split('.')[0]
            #remove 0s and then find genre using id
            trackID = trackID.lstrip("0")
            genre = trackDict.get(trackID)

            if (genre == 'Hip-Hop' or genre == 'Rock' or genre == 'Pop' or genre == 'Folk'):
                        counter += 1
                        print counter,' ',trackID
                        # Extract the mel-spectrogram
                        y, sr = librosa.load(f)
                        # Let's make and display a mel-scaled power (energy-squared) spectrogram
                        mel_spec = librosa.feature.melspectrogram(y, sr=sr,n_mels=128,hop_length=1024,n_fft=2048)
                        # Convert to log scale (dB). We'll use the peak power as reference.
                        log_mel_spec = librosa.logamplitude(mel_spec, ref_power=np.max)
                        #make dimensions of the array even 128x644
                        log_mel_spec = np.resize(log_mel_spec,(128,644))
                        # print log_mel_spec.shape
                        #store into feature array
                        features.append(log_mel_spec.flatten())

                        label = genreDict.get(genre)
                        labels.append(label)

            else:
                pass

    features = np.asarray(features).reshape(len(features),82432)
    print features.shape
    return (features, one_hot_encode(labels))


def one_hot_encode(labels,num_classes=4):

    assert len(labels) > 0

    if num_classes is None:
        num_classes = np.max(labels)+1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(labels)

    result = np.zeros(shape=(len(labels), num_classes))
    result[np.arange(len(labels)), labels] = 1
    return np.array(result)

def get_fma_data():
    #use this function to return genres for the 100,000 unique song ids
    tracks = pd.read_csv('../../tracks.csv')
    d = {}
    counter = 0
    for index,row in tracks.iterrows():
            counter += 1
            if (row['subset'] == 'small'):
                d[row['track_id']] = row['genre_top']
                print row['genre_top']


    print d

    return d

if __name__ == "__main__":
    # d = get_fma_data()
    # with open('../track.data', 'w') as f:
    #         pickle.dump(d, f)
    track_data = {}
    with open('../../track.data', 'r') as f:
      track_data = pickle.load(f)

    train_data, train_labels = extract_features('../../fma_small',track_data)

    batch1, batch2 = train_test_split(train_data,test_size=0.5)
    batch1Labels, batch2Labels = train_test_split(train_labels,test_size=0.5)
    # store preprocessed data in serialised format so we can save computation time and power
    with open('../../4genreBatch1.data', 'w') as f:
        pickle.dump(batch1, f)
    with open('../../4genreBatch2.data', 'w') as f:
        pickle.dump(batch2, f)

    with open('../../4genreBatch1.labels', 'w') as f:
        pickle.dump(batch1Labels, f)
    with open('../../4genreBatch2.labels', 'w') as f:
        pickle.dump(batch2Labels, f)
