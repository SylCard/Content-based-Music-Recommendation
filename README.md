# Content-based-Music-Recommendation System using Deep Learning
Project seeking to bridge the semantic gap between music audio and user preferences

## Abstract
The automated curation of music playlists has become a signifcant problem in the last decade, with the colossal rise of streaming platforms e.g. Spotify.

Current state-of-the-art recommender systems depend on the **collaborative filtering** model. 
Nevertheless, these systems experience the **cold start problem**; 
they break down when no historical
data is available and as a result they cannot recommend new and unpopular songs.


In this project, the author proposes a new recommendation model that uses music
audio and deep learning to produce playlists based on a given query song. 

details explained here: https://medium.com/@silvercloud438/how-i-taught-a-neural-network-to-understand-similarities-in-music-audio-d4fca54c1aed
## Code

to use this system:

- download a dataset (FMA,GTZAN)
- feature convert data to mel-spectrograms using featureConverter script
- change file path in model script to feed data and train CNN
- keep an ordered list of songTitles in text file for library of songs
- run CNN in predict script
- run playlist generator

- output playlist :)

have fun!


