# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 22:06:44 2024

@author: sp3660
"""

from PIL import Image
import matplotlib.pyplot as plt
import tifffile
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import rgb_to_hsv
from scipy.ndimage import gaussian_filter
from scipy.io import loadmat
# Open the TIFF file
tiff_path = r'C:\Users\sp3660\Desktop\ChandPaper\Fig2\Movie1opticflow.tif'  # Replace with your TIFF file path

def rgb_to_angle(rgb):
    """ Convert RGB to an angle on the color wheel. """
    # Normalize RGB values to range [0, 1]
    rgb_normalized = np.array(rgb) / 255.0
    
    # Convert RGB to HSV
    hsv = rgb_to_hsv(rgb_normalized)
    
    # Extract hue (angle) from HSV
    hue = hsv[0] * 360  # Hue ranges from 0 to 360 degrees
    angle_rad = np.deg2rad(hue)
    
    # Map angle using cosine to achieve the desired scaling
    # Cosine function: cos(angle) will be 0 at 0 and 180 degrees, 
    # +1 at 90 degrees, and -1 at 270 degrees
    scaled_value = np.sin(angle_rad)
    
    return scaled_value

def load_multipage_image_as_numpy(file_path):
    """ Load a multi-page image into a NumPy array. """
    with Image.open(file_path) as img:
        frames = []
        for i in range(img.n_frames):
            img.seek(i)
            frame = img.convert('RGB')
            frame_array = np.array(frame)
            frames.append(frame_array)
        frames_array = np.stack(frames, axis=0)
    return frames_array

def plot_rgb_images(images_array):
    """ Plot RGB images from a NumPy array. """
    num_images = images_array.shape[0]
    
    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    
    if num_images == 1:
        axes = [axes]  # Ensure axes is iterable when there's only one image

    for i in range(num_images):
        ax = axes[i]
        ax.imshow(images_array[i])
        ax.axis('off')
        ax.set_title(f'Page {i + 1}')
    
    plt.tight_layout()
    plt.show()



with Image.open(tiff_path) as img:
    frame_number = 0
    avg_rgb_values = []  # To store average RGB values for each frame
    
    while True:
        try:
            # Convert to RGB if not already in RGB mode
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert image to numpy array
            img_array = np.array(img)
            
            # Calculate the average RGB values for the current frame
            avg_rgb = np.mean(img_array, axis=(0, 1))  # Mean over width and height
            
            # Append average RGB values to the list
            avg_rgb_values.append(avg_rgb)
            
            # Move to the next frame
            img.seek(img.tell() + 1)
            frame_number += 1
        except EOFError:
            # End of the multipage TIFF file
            break

# Convert list of average RGB values to a NumPy array for plotting
avg_rgb_values = np.array(avg_rgb_values)

# Plot the average RGB values over time
plt.figure(figsize=(12, 6))
plt.plot(avg_rgb_values[:, 0], label='Average Red', color='r')
plt.plot(avg_rgb_values[:, 1], label='Average Green', color='g')
plt.plot(avg_rgb_values[:, 2], label='Average Blue', color='b')
plt.xlabel('Frame Number')
plt.ylabel('Average RGB Value')
plt.title('Average RGB Values Over Time')
plt.legend()
plt.show()


#%%
angl=[rgb_to_angle(frame) for frame in avg_rgb_values]
filt=gaussian_filter(angl, sigma=2)
plt.plot(gaussian_filter(angl, sigma=1))
intensity=loadmat(r'C:\Users\sp3660\Desktop\ChandPaper\Fig2\VideoMeanIntensity.mat')['intensity'].flatten()
filt=np.insert(filt,0,0)
f,ax=plt.subplots(2,sharex=True)
ax[0].plot(filt)
ax[1].plot(np.abs(np.diff(np.abs(angl))))


#%% FREQUENCY COPMPOSITION
mo=load_multipage_image_as_numpy(r'C:\Users\sp3660\Desktop\ChandPaper\Fig2\Movie1tiff.tif')
import cv2

gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in mo]

#%%

import numpy as np

def compute_frequency_spectrum(frame):
    dft = np.fft.fft2(frame)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = np.abs(dft_shift)
    return magnitude_spectrum

frequency_spectra = [compute_frequency_spectrum(frame) for frame in gray_frames]

def calculate_power_spectrum(magnitude_spectrum):
    power_spectrum = magnitude_spectrum**2
    return np.mean(power_spectrum)

power_spectra = [calculate_power_spectrum(spectrum) for spectrum in frequency_spectra]

from scipy.stats import entropy

def calculate_entropy(magnitude_spectrum):
    flat_spectrum = magnitude_spectrum.flatten()
    norm_spectrum = flat_spectrum / np.sum(flat_spectrum)
    return entropy(norm_spectrum)

entropies = [calculate_entropy(spectrum) for spectrum in frequency_spectra]

average_entropy = np.mean(entropies)
average_power_spectrum = np.mean(power_spectra)

import matplotlib.pyplot as plt

# plt.plot(entropies)
plt.plot(power_spectra)

plt.xlabel('Frame Index')
plt.ylabel('Entropy')
plt.title('Entropy of Each Frame')
plt.show()



import matplotlib.pyplot as plt

# Compute power spectra
def compute_power_spectrum(magnitude_spectrum):
    return np.log(np.sum(magnitude_spectrum, axis=0) + 1)  # Adding 1 to avoid log(0)

# Calculate the power spectrum for each frame
power_spectra = [compute_power_spectrum(spectrum) for spectrum in frequency_spectra]

# Stack power spectra vertically to create the spectrogram
spectrogram = np.stack(power_spectra, axis=0)

# Normalize the spectrogram for better visualization
spectrogram -= np.min(spectrogram)
spectrogram /= np.max(spectrogram)


# Increase contrast using linear stretching
def enhance_contrast(image, lower_percentile=10, upper_percentile=80):
    """Enhance the contrast of an image using percentile stretching."""
    lower, upper = np.percentile(image, (lower_percentile, upper_percentile))
    return np.clip((image - lower) / (upper - lower), 0, 1)

# Apply contrast enhancement
contrast_spectrogram = enhance_contrast(spectrogram)
# Plot the spectrogram
plt.imshow(contrast_spectrogram.T, aspect='auto', cmap='inferno')
plt.colorbar(label='Power')
plt.xlabel('Frame')
plt.ylabel('Frequency Bin')
plt.title('Spectrogram of Video')
plt.show()

