import numpy as np
import cv2
import time

radar_matrix = np.load('data/easy2.npy')

window_size = 50
FPS = 17

# radar_window = radar_matrix[:window_size]
radar_window = np.zeros((window_size, radar_matrix.shape[1]), dtype=np.complex128)

prev_time = time.time()
prev_frame = 0

# Adjust this value to increase or decrease the image size
scale_factor = 500/window_size
width_to_height_ratio = 2

frame_index = 0

def plot_radar_window(window, scale_factor=1, width_to_height_ratio=1):
    img = np.abs(window).T
    img = cv2.normalize(img, None, 0.0, 1.0, cv2.NORM_MINMAX)
    img = np.float32(img)
    img = cv2.applyColorMap(np.uint8(img * 255), cv2.COLORMAP_PARULA)

    height, width, _ = img.shape
    max_dim = max(height, width)
    new_size = int(max_dim * scale_factor)
    new_height = new_size
    new_width = int(width_to_height_ratio*new_size)

    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    # Display the image
    cv2.imshow('Radar Window', img)

    return img

def plot_show(delay=1):
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        exit()

class FPSCounter:
    def __init__(self):
        self.prev_time = time.time()
        self.prev_frame = 0

    def update(self):
        now = time.time()
        elapsed_time = now - self.prev_time
        if elapsed_time > 1:
            fps = (self.prev_frame + 1) / elapsed_time
            self.prev_frame = 0
            self.prev_time = now
            print(f"target fps: {FPS:.2f}, actual fps: {fps:.2f}")
        else:
            self.prev_frame += 1


fps = FPSCounter()

while True:
    radar_window = np.roll(radar_window, -1, axis=0)
    radar_window[-1] = radar_matrix[frame_index]
    frame_index += 1
    if frame_index >= radar_matrix.shape[0]:
        break

    # calculate fps each second
    fps.update()

    plot_radar_window(radar_window, scale_factor, width_to_height_ratio)
    plot_show(50)

cv2.destroyAllWindows()