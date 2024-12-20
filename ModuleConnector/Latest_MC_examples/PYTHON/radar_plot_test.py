from pulse_doppler import load_radar_matrix
import matplotlib.pyplot as plt
import numpy as np

radar_matrix = load_radar_matrix('data/easy2.npy')

window_size = 200

fig, ax = plt.subplots(1, 1, figsize=(12, 12))
# radar_window = np.zeros((window_size, radar_matrix.shape[1]), dtype=np.complex)
radar_window = radar_matrix[:window_size]

im = ax.imshow(np.abs(radar_window).T, aspect='auto', origin='lower')
colorbar = fig.colorbar(im, ax=ax)

while True:
    radar_window = np.roll(radar_window, -1, axis=0)
    # radar_window[-1] = radar_matrix[np.random.randint(0, radar_matrix.shape[0])]

    im.set_array(np.abs(radar_window).T)
    colorbar.update_normal(im)
    plt.pause(0.1)