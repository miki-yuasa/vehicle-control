import time
from vehicle_control.path import create_path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

points_filename = "assets/path_points.json"
path_savename = "assets/path.npz"

px, py, yaws = create_path(points_filename, path_savename)

# Plot the path
ax: Axes
fig, ax = plt.subplots()
ax.plot(px, py, "--", label="Path")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_title("Path")
ax.axis("equal")
ax.grid()
ax.legend()
fig.canvas.draw()
fig.canvas.flush_events()
time.sleep(0.1)
