import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

# Create a figure and axis
fig, ax = plt.subplots()

# Function to create frames for animation
def create_frames(num_frames):
    frames = []
    for _ in range(num_frames):
        data = np.random.rand(10, 10)
        im = ax.imshow(data, cmap='viridis')
        frames.append([im])  # Each frame contains a list of artists (images)
        colorbar.update_normal(im)  # Update colorbar to reflect new data
    return frames

# Create initial data and colorbar
data = np.random.rand(10, 10)
im = ax.imshow(data, cmap='viridis')
colorbar = fig.colorbar(im, ax=ax)

# Create frames for the animation
frames = create_frames(num_frames=100)

# Create and display the animation
ani = ArtistAnimation(fig, frames, interval=50, blit=True)
plt.show()