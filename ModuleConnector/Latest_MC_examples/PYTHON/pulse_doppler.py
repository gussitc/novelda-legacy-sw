from webbrowser import get
import radar_config as rc
from pymoduleconnector.extras.auto import auto
from pymoduleconnector.extras.x4_regmap_autogen import X4
import numpy as np
import matplotlib.pyplot as plt
from pymoduleconnector.moduleconnectorwrapper import PyXEP, PyX4M210
import os
from plot import plot_radar_matrix, plot_single_bin, plot_bin_fft
import time

SAVE_TO_FILE = True
FILENAME = "radar_matrix.npy"
FREQUENCY_CUTOFF = 2

x4_par_settings = {'downconversion': 1,  # 0: output rf data; 1: output baseband data
                   'dac_min': 949,
                   'dac_max': 1100,
                   'iterations': 16,
                   'tx_center_frequency': 3, #7.29GHz Low band: 3, 8.748GHz High band: 4
                   'tx_power': 2,
                   'pulses_per_step': 87,
                   'frame_area_offset': 0.18,
                   'frame_area': (0.5, 1.5),
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

def plot_fft_matrix(fft_matrix, freqs, ax=None):
    abs_matrix = np.abs(fft_matrix)
    if ax is None:
        plt.figure(figsize=(12, 6))
        plt.imshow(abs_matrix, aspect='auto', extent=[0, abs_matrix.shape[1], freqs[0], freqs[-1]], origin='lower')
        plt.colorbar()
        plt.title("FFT of Radar Matrix")
        plt.xlabel("Bin")
        plt.ylabel("Frequency (Hz)")
        plt.grid()
    else:
        ax.imshow(abs_matrix, aspect='auto', extent=[0, abs_matrix.shape[1], freqs[0], freqs[-1]], origin='lower')
        ax.set_title("FFT of Radar Matrix")
        ax.set_xlabel("Bin")
        ax.set_ylabel("Frequency (Hz)")
        ax.grid()

def plot_radar_matrix(radar_matrix, ax=None):
    if ax is None:
        plt.figure(figsize=(12, 6))
        plt.imshow(np.abs(radar_matrix).T, aspect='auto', origin='lower')
        plt.colorbar()
        plt.title("Radar Matrix Amplitude")
        plt.xlabel("Frame")
        plt.ylabel("Bin")
    else:
        ax.imshow(np.abs(radar_matrix).T, aspect='auto', origin='lower')
        ax.set_title("Radar Matrix Amplitude")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Bin")

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

def get_xep() -> PyXEP:
    device_name = auto()[0]
    xep = rc.configure_x4(device_name, x4_settings=x4_par_settings)
    return xep

def get_radar_frame_if_avail(xep: PyXEP):
    if xep.peek_message_data_float() > 0:
        d = xep.read_message_data_float()
        frame = np.array(d.data)
        n = len(frame)
        frame = frame[:n//2] + 1j*frame[n//2:]
        return frame
    return None

def main():
    num_frames = 200
    fps = x4_par_settings['fps']
    interval = 1.0 / fps
    window_size = 100  # Number of frames to keep in view
    # radar_matrix = load_radar_matrix('data/easy2.npy')
    # radar_matrix = load_radar_matrix('data/hard2.npy')

    radar_matrix = generate_radar_matrix(num_frames)

    fft_window = window_size
    fft_interval = 10
    
    # plot_radar_matrix(radar_matrix)
    # derivative = np.diff(radar_matrix, axis=0)
    # plot_radar_matrix(derivative)
    # fft_matrix, freqs = get_fft_matrix(radar_matrix)
    # plot_fft_matrix(fft_matrix, freqs)

    # freq, bin = get_fft_matrix_max(fft_matrix, freqs)
    # print(f"Freq: {freq}, Bin: {bin}")

    # plt.ion()
    # fig, ax = plt.subplots(2, 1, figsize=(12, 12))

    xep = get_xep()
    radar_frames = []

    prev_time = time.time()
    frame_count = 0
    i = 0

    while True:
        frame = get_radar_frame_if_avail(xep)
        if frame is not None:
            radar_frames.append(frame)
            frame_count += 1
            i += 1
            if len(radar_frames) > window_size:
                radar_frames = radar_frames[-window_size:]

            # ax[0].clear()
            # plot_radar_matrix(np.array(radar_frames), ax=ax[0])

            if len(radar_frames) >= fft_window and i % fft_interval == 0:
                fft_matrix, freqs = get_fft_matrix(np.array(radar_frames))
                # ax[1].clear()
                # plot_fft_matrix(fft_matrix, freqs, ax=ax[1])
                freq, bin = get_fft_matrix_max(fft_matrix, freqs)
                print(f"Freq: {freq}, Bin: {bin}")

            # plt.pause(interval)

        # Calculate and print FPS
        now = time.time()
        if now - prev_time > 1:
            print(f"FPS: {frame_count:.2f}")
            frame_count = 0
            prev_time = now

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
