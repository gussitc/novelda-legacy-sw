import radar_config as rc
from pymoduleconnector.extras.auto import auto
from pymoduleconnector.extras.x4_regmap_autogen import X4
import numpy as np
import matplotlib.pyplot as plt
from pymoduleconnector.moduleconnectorwrapper import PyXEP, PyX4M210
import os

SAVE_TO_FILE = True
FILENAME = "radar_matrix.npy"

x4_par_settings = {'downconversion': 1,  # 0: output rf data; 1: output baseband data
                   'dac_min': 949,
                   'dac_max': 1100,
                   'iterations': 16,
                   'tx_center_frequency': 3, #7.29GHz Low band: 3, 8.748GHz High band: 4
                   'tx_power': 2,
                   'pulses_per_step': 87,
                   'frame_area_offset': 0.18,
                   'frame_area': (0.5, 1),
                   'fps': 17,
                   }

def plot_radar_matrix(radar_matrix):
    plt.figure(figsize=(12, 6))

    # Plot amplitude
    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(radar_matrix).T, aspect='auto')
    plt.colorbar()
    plt.title("Radar Matrix Amplitude")
    plt.xlabel("Frame")
    plt.ylabel("Bin")

    # Plot phase
    plt.subplot(1, 2, 2)
    plt.imshow(np.angle(radar_matrix).T, aspect='auto')
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

def plot_bin_fft(radar_matrix, bin):
    bin_fft = np.fft.fft(radar_matrix[:, bin])
    num_frames = radar_matrix.shape[0]
    sample_spacing = 1.0 / x4_par_settings['fps']
    freqs = np.fft.fftfreq(num_frames, sample_spacing)
    
    # Only keep positive frequencies and skip 0
    positive_freqs = freqs > 0
    bin_fft = bin_fft[positive_freqs]
    freqs = freqs[positive_freqs]
    # remove freqs below 10Hz
    freqs = freqs[freqs < 5]
    bin_fft = bin_fft[:len(freqs)]
    
    plt.figure(figsize=(12, 6))
    plt.plot(freqs, np.abs(bin_fft))
    plt.title(f"FFT of Bin {bin}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid()
    # increase x-axis granularity
    plt.xticks(np.arange(0, max(freqs), 0.5))

def plot_matrix_fft(radar_matrix):
    radar_matrix_fft = np.fft.fft(radar_matrix, axis=0)
    num_frames = radar_matrix.shape[0]
    sample_spacing = 1.0 / x4_par_settings['fps']
    freqs = np.fft.fftfreq(num_frames, sample_spacing)
    
    # Only keep positive frequencies and skip 0
    positive_freqs = freqs > 0
    radar_matrix_fft = radar_matrix_fft[positive_freqs, :]
    freqs = freqs[positive_freqs]
    # remove freqs below 10Hz
    freqs = freqs[freqs < 5]
    radar_matrix_fft = radar_matrix_fft[:len(freqs), :]
    
    plt.figure(figsize=(12, 6))
    plt.imshow(np.abs(radar_matrix_fft), aspect='auto', extent=[0, radar_matrix.shape[1], freqs[0], freqs[-1]], origin='lower')
    plt.colorbar()
    plt.title("FFT of Radar Matrix")
    plt.xlabel("Bin")
    plt.ylabel("Frequency (Hz)")
    plt.grid()

def save_radar_matrix(radar_matrix):
    np.save(FILENAME, radar_matrix)

def load_radar_matrix():
    return np.load(FILENAME)

def generate_radar_matrix(num_frames):
    radar_frames = []
    device_name = auto()[0]
    xep = rc.configure_x4(device_name, x4_settings=x4_par_settings)
    while len(radar_frames) < num_frames:
        if xep.peek_message_data_float() > 0:
            d = xep.read_message_data_float()
            frame = np.array(d.data)
            n = len(frame)
            frame = frame[:n//2] + 1j*frame[n//2:]
            radar_frames.append(frame)
    return np.array(radar_frames)

def main():
    radar_matrix = load_radar_matrix()
    # if os.path.exists(FILENAME):
    #     user_input = input(f"{FILENAME} already exists. Do you want to delete it and generate a new radar matrix? (y/n): ")
    #     if user_input.lower() == 'y':
    #         os.remove(FILENAME)
    #         radar_matrix = generate_radar_matrix(200)
    #         save_radar_matrix(radar_matrix)
    #     else:
    #         radar_matrix = load_radar_matrix()
    # else:
    #     radar_matrix = generate_radar_matrix(200)
    #     save_radar_matrix(radar_matrix)
    # plot_radar_matrix(radar_matrix)
    # plot_single_bin(radar_matrix, 4)
    plot_bin_fft(radar_matrix, 4)
    plot_matrix_fft(radar_matrix)
    # bin_fft = np.fft.fft(radar_matrix[:, 4])
    plt.show()

    # calculate fft
    # radar_matrix_fft = np.fft.fft(radar_matrix, axis=1)

    # plot fft matrix
    # plot_radar_matrix(radar_matrix_fft)

if __name__ == "__main__":
    main()
