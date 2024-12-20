from pulse_doppler import load_radar_matrix
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import time

radar_matrix = load_radar_matrix('data/easy2.npy')

window_size = 200
FPS = 17

fig, ax = plt.subplots(1, 1, figsize=(12, 12))
# radar_window = np.zeros((window_size, radar_matrix.shape[1]), dtype=np.complex)
radar_window = radar_matrix[:window_size]

im = ax.imshow(np.abs(radar_window).T, aspect='auto', origin='lower')
colorbar = fig.colorbar(im, ax=ax)

prev_time = 0
prev_frame = 0

def update(frame):
    global radar_window, im, colorbar, prev_time, prev_frame
    radar_window = np.roll(radar_window, -1, axis=0)
    # radar_window[-1] = radar_matrix[np.random.randint(0, radar_matrix.shape[0])]

    # calculate fps each second
    now = time.time()
    elapsed_time = now - prev_time
    if elapsed_time > 1:
        fps = (frame-prev_frame) / elapsed_time
        prev_frame = frame
        prev_time = now
        print(f"target fps: {FPS:.2f}, actual fps: {fps:.2f}")

    im.set_array(np.abs(radar_window).T)
    colorbar.update_normal(im)
    return im,

ani = FuncAnimation(fig, update, interval=1/FPS * 1000, blit=True)
plt.show()