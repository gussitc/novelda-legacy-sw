import radar_config as rc
from pymoduleconnector.extras.auto import auto
from pymoduleconnector.extras.x4_regmap_autogen import X4
import numpy as np
import matplotlib.pyplot as plt
from pymoduleconnector.moduleconnectorwrapper import PyXEP, PyX4M210

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
    plt.imshow(np.abs(radar_matrix).T, aspect='auto')
    plt.colorbar()
    plt.title("Radar Matrix")
    plt.xlabel("Frame")
    plt.ylabel("Bin")
    plt.show()

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
    try:
        radar_matrix = load_radar_matrix()
    except FileNotFoundError:
        radar_matrix = generate_radar_matrix(200)
        save_radar_matrix(radar_matrix)

    plot_radar_matrix(radar_matrix)

    # calculate fft
    # radar_matrix_fft = np.fft.fft(radar_matrix, axis=1)

    # plot fft matrix
    # plot_radar_matrix(radar_matrix_fft)

if __name__ == "__main__":
    main()
