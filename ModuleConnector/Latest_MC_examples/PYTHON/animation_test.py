import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Create a figure and axis
fig, ax = plt.subplots()

# Create initial data and colorbar
data = np.random.rand(10, 10)
im = ax.imshow(data)
# colorbar = fig.colorbar(im, ax=ax)

# Function to update the frame
def update(frame):
    data = np.random.rand(10, 10)
    im.set_array(data)
    # colorbar.update_normal(im)
    return im,

# Create and display the animation
ani = animation.FuncAnimation(fig, update, frames=100, interval=50, blit=True)
plt.show()