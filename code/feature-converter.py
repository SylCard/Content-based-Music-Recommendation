
#the aim of this file will be to traverse my dataset and output an array containing features for each track with corresponding labels
import glob
import os
import sys
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
# np.set_printoptions(threshold='nan')

genreDict = {
    'pop'        :   0,
    'rock'       :   1,
    'hiphop'    :   2,
    'country'       :   3,
    # 'blues'     :   0,
    # 'classical' :   1,
    # 'country'   :   2,
    # 'disco'     :   3,
    # 'hiphop'    :   4,
    # 'jazz'      :   5,
    # 'metal'     :   6,
    # 'pop'       :   7,
    # 'reggae'    :   8,
    # 'rock'      :   9,
}

# this function will iterate through each file in the dataset
def extract_features(basedir,extension='.au'):
    features=[]
    labels=[]
    # iterate over all files in all subdirectories of the base directory
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root,'*'+extension))
        # apply function to all files
        for f in files :
            genre = f.split('/')[4].split('.')[0]

            if (genre == 'hiphop' or genre == 'rock' or genre == 'pop' or genre == 'country'):
                print genre
                # Extract the mel-spectrogram
                y, sr = librosa.load(f)
                # Let's make and display a mel-scaled power (energy-squared) spectrogram
                mel_spec = librosa.feature.melspectrogram(y, sr=sr,n_mels=128,hop_length=1024,n_fft=2048)
                # Convert to log scale (dB). We'll use the peak power as reference.
                log_mel_spec = librosa.logamplitude(mel_spec, ref_power=np.max)
                #make dimensions of the array even 128x1292
                log_mel_spec = np.resize(log_mel_spec,(128,644))
                print log_mel_spec.shape
                #store into feature array
                features.append(log_mel_spec.flatten())
                # print len(np.array(log_mel_spec.T.flatten()))
                # Extract label
                label = genreDict.get(genre)
                labels.append(label)
            else:
                pass
    features = np.asarray(features).reshape(len(features),82432)
    print features.shape
    print len(labels)

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

if __name__ == "__main__":
    trainingPath = '../../gtzanDataset'
    train_data, train_labels = extract_features(trainingPath)

    # store preprocessed data in serialised format so we can save computation time and power
    with open('../../4GenreTest.data', 'w') as f:
        pickle.dump(train_data, f)

    with open('../../4GenreTest.labels', 'w') as f:
        pickle.dump(train_labels, f)
