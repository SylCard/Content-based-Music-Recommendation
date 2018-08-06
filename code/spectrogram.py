# Feature extraction example
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
np.set_printoptions(threshold='nan')
# Load sound file
y, sr = librosa.load("Migos.mp3")

# Set the hop length; at 22050 Hz, 512 samples ~= 23ms
hop_length = 512
n_fft = 1024
# Let's make and display a mel-scaled power (energy-squared) spectrogram
S = librosa.feature.melspectrogram(y, sr=sr,n_mels=128,hop_length=hop_length,n_fft=n_fft)
# Convert to log scale (dB). We'll use the peak power as reference.
log_S = librosa.logamplitude(S, ref_power=np.max)

# Make a new figure
plt.figure(figsize=(12,4))

# Display the spectrogram on a mel scale
# sample rate and hop length parameters are used to render the time axis
librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

# Put a descriptive title on the plot
plt.title('mel spectrogram')

# draw a color bar
plt.colorbar(format='%+02.0f dB')

# Make the figure layout compact
plt.tight_layout()

# finally visualise the plot
plt.show()
