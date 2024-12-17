import radar_config as rc
from pymoduleconnector.extras.auto import auto
from pymoduleconnector.extras.x4_regmap_autogen import X4
import numpy as np
import matplotlib.pyplot as plt

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

def main():
    device_name = auto()[0]
    xep = rc.configure_x4(device_name, x4_settings=x4_par_settings)
    radar_frames = []
    num_frames = 100

    while len(radar_frames) < num_frames:
        if xep.peek_message_data_float() > 0:
            d = xep.read_message_data_float()
            frame = np.array(d.data)
            n = len(frame)
            frame = frame[:n//2] + 1j*frame[n//2:]
            radar_frames.append(frame)

    print("Radar frames collected: ", len(radar_frames))
    print("First frame: ", radar_frames[0])

    # create matrix
    radar_matrix = np.array(radar_frames)
    print("Radar matrix shape: ", radar_matrix.shape)
    print(f"Bin count: {xep.x4driver_get_frame_bin_count()}")

    plot_radar_matrix(radar_matrix)

    # calculate fft
    # radar_matrix_fft = np.fft.fft(radar_matrix, axis=1)

    # plot fft matrix
    # plot_radar_matrix(radar_matrix_fft)

if __name__ == "__main__":
    main()
