import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.special import sph_harm

matplotlib.use('TkAgg')


def calculate_radius(theta, parameters):
    """Calculate the radius for each angle using spherical harmonics."""
    radius = np.ones_like(theta)

    for l in range(1, 9):
        # Using only the m=0 harmonics (axially symmetric)
        # Real part of Y(l,0) is sufficient as m=0 harmonics are real
        harmonic = np.real(sph_harm(0, l, 0, theta))
        radius += parameters[l - 1] * harmonic

    return radius


def main():
    # Set up the figure and single axis for the plot
    fig = plt.figure(figsize=(10, 8))
    ax_plot = plt.subplot(111)

    # Adjust the main plot area
    plt.subplots_adjust(left=0.1, bottom=0.4, right=0.9, top=0.95)

    # Initial parameters
    num_harmonics = 8
    initial_params = (0.0,) * num_harmonics
    theta = np.linspace(0, 2 * np.pi, 1000)

    # Calculate and plot initial shape
    radius = calculate_radius(theta, initial_params)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    line, = ax_plot.plot(x, y)

    # Set up the plot
    ax_plot.set_aspect('equal')
    ax_plot.grid(True)
    ax_plot.set_title('2D Shape with Spherical Harmonics Deformation')
    ax_plot.set_xlabel('X')
    ax_plot.set_ylabel('Y')

    # Create sliders
    slider_height = 0.035  # Height between sliders
    sliders = []

    for i in range(num_harmonics):
        ax = plt.axes((0.2, 0.05 + i * slider_height, 0.6, 0.02))

        # Special case for Y20 (index 1)
        if i == 1:
            valmin, valmax = 0.0, 3.0
        else:
            valmin, valmax = -1.0, 1.0

        slider = Slider(
            ax=ax,
            label=f'β{i + 1}0',
            valmin=valmin,
            valmax=valmax,
            valinit=initial_params[i],
            valstep=0.01
        )
        sliders.append(slider)

    # Update function for the plot
    def update(val):
        parameters = [slider.val for slider in sliders]
        radius = calculate_radius(theta, parameters)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        line.set_data(x, y)

        # Update plot limits to accommodate shape changes
        max_radius = np.max(np.abs(radius)) * 1.1
        ax_plot.set_xlim(-max_radius, max_radius)
        ax_plot.set_ylim(-max_radius, max_radius)
        fig.canvas.draw_idle()

    # Connect the update function to the sliders
    for slider in sliders:
        slider.on_changed(update)

    plt.show(block=True)


if __name__ == '__main__':
    main()
