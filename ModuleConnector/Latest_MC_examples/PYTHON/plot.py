import numpy as np
import matplotlib.pyplot as plt
def plot_radar_matrix(radar_matrix):
    plt.figure(figsize=(12, 6))

    # Plot amplitude
    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(radar_matrix).T, aspect='auto', origin='lower')
    plt.colorbar()
    plt.title("Radar Matrix Amplitude")
    plt.xlabel("Frame")
    plt.ylabel("Bin")

    # Plot phase
    plt.subplot(1, 2, 2)
    plt.imshow(np.angle(radar_matrix).T, aspect='auto', origin='lower')
    plt.colorbar()
    plt.title("Radar Matrix Phase")
    plt.xlabel("Frame")
    plt.ylabel("Bin")

    plt.tight_layout()

def plot_single_bin(radar_matrix, bin):

    plt.figure(figsize=(12, 6))

    # Plot amplitude
    plt.subplot(1, 2, 1)
    plt.plot(np.abs(radar_matrix[:, bin]))
    plt.title(f"Amplitude of Bin {bin}")
    plt.xlabel("Frame")
    plt.ylabel("Amplitude")

    # Plot phase
    plt.subplot(1, 2, 2)
    plt.plot(np.angle(radar_matrix[:, bin]))
    plt.title(f"Phase of Bin {bin}")
    plt.xlabel("Frame")
    plt.ylabel("Phase")

    plt.tight_layout()

def plot_bin_fft(fft_matrix, freqs, bin):
    bin_fft = fft_matrix[:, bin]
    plt.figure(figsize=(12, 6))
    plt.plot(freqs, np.abs(bin_fft))
    plt.title(f"FFT of Bin {bin}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid()
    # increase x-axis granularity
    plt.xticks(np.arange(0, max(freqs), 0.5))