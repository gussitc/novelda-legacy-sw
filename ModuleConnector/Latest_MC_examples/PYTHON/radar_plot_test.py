import numpy as np
import cv2
import time

radar_matrix = np.load('data/hard2.npy')

window_size = 200
FPS = 17

radar_window = radar_matrix[:window_size]

prev_time = time.time()
prev_frame = 0

scale_factor = 2.0  # Adjust this value to increase or decrease the image size

while True:
    radar_window = np.roll(radar_window, -1, axis=0)
    # radar_window[-1] = radar_matrix[np.random.randint(0, radar_matrix.shape[0])]

    # calculate fps each second
    now = time.time()
    elapsed_time = now - prev_time
    if elapsed_time > 1:
        fps = (prev_frame + 1) / elapsed_time
        prev_frame = 0
        prev_time = now
        print(f"target fps: {FPS:.2f}, actual fps: {fps:.2f}")
    else:
        prev_frame += 1

    # Normalize and convert to floating point image
    img = np.abs(radar_window).T
    img = cv2.normalize(img, None, 0.0, 1.0, cv2.NORM_MINMAX)
    img = np.float32(img)

    # Apply Viridis color map
    img = cv2.applyColorMap(np.uint8(img * 255), cv2.COLORMAP_VIRIDIS)

    # Resize the image to make it larger and square using the scale factor
    height, width, _ = img.shape
    max_dim = max(height, width)
    new_size = int(max_dim * scale_factor)
    img = cv2.resize(img, (new_size, new_size), interpolation=cv2.INTER_LINEAR)

    # Display the image
    cv2.imshow('Radar Visualization', img)

    # Wait for the interval time and check for exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()