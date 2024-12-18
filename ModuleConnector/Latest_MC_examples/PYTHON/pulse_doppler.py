import radar_config as rc
from pymoduleconnector.extras.auto import auto
from pymoduleconnector.extras.x4_regmap_autogen import X4
import numpy as np
import matplotlib.pyplot as plt
from pymoduleconnector.moduleconnectorwrapper import PyXEP, PyX4M210
import os
from plot import plot_radar_matrix, plot_single_bin, plot_bin_fft

SAVE_TO_FILE = True
FILENAME = "radar_matrix.npy"
FREQUENCY_CUTOFF = 5

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

def get_fft_matrix(radar_matrix):
    radar_matrix_fft = np.fft.fft(radar_matrix, axis=0)
    num_frames = radar_matrix.shape[0]
    sample_spacing = 1.0 / x4_par_settings['fps']
    freqs = np.fft.fftfreq(num_frames, sample_spacing)
    
    # Only keep positive frequencies and skip 0
    # TODO: why do we remove 0? check pySDR for answer
    positive_freqs = freqs > 0
    radar_matrix_fft = radar_matrix_fft[positive_freqs, :]
    freqs = freqs[positive_freqs]
    freqs = freqs[freqs < FREQUENCY_CUTOFF]
    radar_matrix_fft = radar_matrix_fft[:len(freqs), :]
    return radar_matrix_fft, freqs

def plot_fft_matrix(fft_matrix, freqs):
    abs_matrix = np.abs(fft_matrix)
    plt.figure(figsize=(12, 6))
    plt.imshow(abs_matrix, aspect='auto', extent=[0, abs_matrix.shape[1], freqs[0], freqs[-1]], origin='lower')
    plt.colorbar()
    plt.title("FFT of Radar Matrix")
    plt.xlabel("Bin")
    plt.ylabel("Frequency (Hz)")
    plt.grid()

def get_fft_matrix_max(fft_matrix, freqs):
    abs_matrix = np.abs(fft_matrix)
    max_value = np.max(abs_matrix)
    max_index = np.argmax(abs_matrix)
    row, col = np.unravel_index(max_index, abs_matrix.shape)
    freq = freqs[row]
    bin = col
    print(f"Max value: {max_value} at freq {freq} and bin {bin}")
    return freq, bin

def save_radar_matrix(radar_matrix):
    np.save(FILENAME, radar_matrix)

def load_radar_matrix(file=FILENAME):
    return np.load(file)

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
    num_frames = 200
    short_num_frames = 100
    # radar_matrix = load_radar_matrix('data/easy2.npy')
    radar_matrix = load_radar_matrix('data/hard2.npy')
    # radar_matrix = generate_radar_matrix(num_frames)
    # save_radar_matrix(radar_matrix)

    radar_matrix = radar_matrix[:short_num_frames, :]

    plot_radar_matrix(radar_matrix)
    fft_matrix, freqs = get_fft_matrix(radar_matrix)
    plot_fft_matrix(fft_matrix, freqs)
    freq, bin = get_fft_matrix_max(fft_matrix, freqs)
    plot_single_bin(radar_matrix, bin)
    plot_bin_fft(fft_matrix, freqs, bin)
    plt.show()

if __name__ == "__main__":
    main()
