#import important modules
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

labelsDict = {
    'RnB'     :     1,
    'Classical' :   2,
    'Country'   :   3,
    'Electronic':   4,
    'Hiphop'    :   5,
    'Jazz'      :   6,
    'Metal'     :   7,
    'Pop'       :   8,
    'Latin'    :    9,
    'Rock'      :   10,
}


#helper methods
# parse_audio_files will iterate through sub directories given a parent directory and apply a second function called extract_features
def parse_audio_files(parent_dir,sub_dirs,file_ext="*.mp3"):
    features, labels = np.empty((0,193)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            try:
              mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
            except Exception as e:
              print "Error encountered while parsing file: ", fn
              continue
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            features = np.vstack([features,ext_features])
            labels = np.append(labels, fn.split('/')[2].split('-')[1])
    return np.array(features), np.array(labels, dtype = np.int)


#sampling rate is 22050 by default 
