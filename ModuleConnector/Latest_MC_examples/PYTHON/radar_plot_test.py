import numpy as np
import cv2
import time

from pymoduleconnector.extras.auto import auto
from pymoduleconnector.extras.x4_regmap_autogen import X4
import numpy as np
# import matplotlib.pyplot as plt
from pymoduleconnector.moduleconnectorwrapper import PyXEP, PyX4M210
import radar_config as rc

# radar_matrix = np.load('data/hard2.npy')

FREQUENCY_CUTOFF = 2
window_size = 100
FPS = 17
FFT_INTERVAL = FPS # calculate FFT every second

# radar_window = radar_matrix[:window_size]

# Adjust this value to increase or decrease the image size
scale_factor = 500/window_size
width_to_height_ratio = 1.6

# plot two large numbers as an image
def show_numbers(bin, freq):
    img = np.zeros((300, 600, 3), dtype=np.uint8)
    cv2.putText(img, f"bin: {int(bin)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)
    cv2.putText(img, f"freq: {freq:.2f}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)
    cv2.imshow('numbers', img)

def plot_radar_window(window, scale_factor=1, width_to_height_ratio=1, name='radar'):
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
    cv2.imshow(name, img)

    return img

def get_fft_matrix(radar_matrix):
    radar_matrix_fft = np.fft.fft(radar_matrix, axis=0)
    num_frames = radar_matrix.shape[0]
    sample_spacing = 1.0 / FPS
    freqs = np.fft.fftfreq(num_frames, sample_spacing)

    # Only keep positive frequencies and skip 0
    # TODO: why do we remove 0? check pySDR for answer
    positive_freqs = freqs > 0
    radar_matrix_fft = radar_matrix_fft[positive_freqs, :]
    freqs = freqs[positive_freqs]
    freqs = freqs[freqs < FREQUENCY_CUTOFF]
    radar_matrix_fft = radar_matrix_fft[:len(freqs), :]
    return radar_matrix_fft, freqs

def get_fft_matrix_max(fft_matrix, freqs):
    abs_matrix = np.abs(fft_matrix)
    max_value = np.max(abs_matrix)
    max_index = np.argmax(abs_matrix)
    row, col = np.unravel_index(max_index, abs_matrix.shape)
    freq = freqs[row]
    bin = col
    print(f"Max value: {max_value} at freq {freq} and bin {bin}")
    return freq, bin

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
            return True
        else:
            self.prev_frame += 1
            return False

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


fps_cntr = FPSCounter()
frame_index = 0
xep = get_xep()
bin_count = xep.x4driver_get_frame_bin_count()
radar_window = np.zeros((window_size, bin_count), dtype=np.complex128)

while True:
    frame = get_radar_frame_if_avail(xep)
    if frame is None:
        continue
    radar_window = np.roll(radar_window, -1, axis=0)
    # radar_window[-1] = radar_matrix[frame_index]
    radar_window[-1] = frame
    # frame_index += 1
    # if frame_index >= radar_matrix.shape[0]:
    #     break

    plot_radar_window(radar_window, scale_factor, width_to_height_ratio, name='radar')

    # calculate fps each second
    # if frame_index % FFT_INTERVAL == 0:
    # this is true only once every second
    if fps_cntr.update():
        fft_matrix, freqs = get_fft_matrix(radar_window)
        freq, bin = get_fft_matrix_max(fft_matrix, freqs)
        show_numbers(bin, freq)
        plot_radar_window(fft_matrix.T, scale_factor*5, name='fft')

    plot_show(1)

cv2.destroyAllWindows()