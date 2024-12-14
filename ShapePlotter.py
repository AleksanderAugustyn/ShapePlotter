"""This script is used to plot the shape of a nucleus with volume conservation, using the beta parameter and spherical harmonics parametrization."""

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.special import sph_harm

matplotlib.use('TkAgg')

# Constants
r0 = 1.16  # Radius constant in fm


def calculate_volume(Z, N, parameters):
    """
    Calculate the volume of a nucleus with given parameters.

    Args:
    :parameter    Z (int): The number of protons.
    :parameter    N (int): The number of neutrons.
    :parameter    parameters (tuple): A tuple of deformation parameters (beta10, beta20, ..., beta80).

    Returns:
    :return    float: The calculated volume of the nucleus.
    """
    NumberOfNucleons = Z + N
    beta10, beta20, beta30, beta40, beta50, beta60, beta70, beta80 = parameters

    # Base coefficient
    base_coefficient = 1 / (111546435 * np.sqrt(np.pi))

    # Main terms
    term1 = 148728580 * np.pi ** (3 / 2)
    term2 = 22309287 * np.sqrt(5) * beta10 ** 2 * beta20
    term3 = 5311735 * np.sqrt(5) * beta20 ** 3
    term4 = 47805615 * beta20 ** 2 * beta40
    term5 = 30421755 * beta30 ** 2 * beta40
    term6 = 9026235 * beta40 ** 3
    term7 = 6686100 * np.sqrt(77) * beta30 * beta40 * beta50
    term8 = 25741485 * beta40 * beta50 ** 2
    term9 = 13000750 * np.sqrt(13) * beta30 ** 2 * beta60
    term10 = 7800450 * np.sqrt(13) * beta40 ** 2 * beta60

    # Additional terms
    term11 = 1820105 * np.sqrt(1001) * beta30 * beta50 * beta60
    term12 = 6729800 * np.sqrt(13) * beta50 ** 2 * beta60
    term13 = 25053210 * beta40 * beta60 ** 2
    term14 = 2093000 * np.sqrt(13) * beta60 ** 3
    term15 = 9100525 * np.sqrt(105) * beta30 * beta40 * beta70

    # More complex terms
    term16 = 4282600 * np.sqrt(165) * beta40 * beta50 * beta70
    term17 = 1541736 * np.sqrt(1365) * beta30 * beta60 * beta70
    term18 = 1014300 * np.sqrt(2145) * beta50 * beta60 * beta70
    term19 = 24647490 * beta40 * beta70 ** 2
    term20 = 6037500 * np.sqrt(13) * beta60 * beta70 ** 2

    # Beta80 terms
    term21 = 11241825 * np.sqrt(17) * beta40 ** 2 * beta80
    term22 = 2569560 * np.sqrt(1309) * beta30 * beta50 * beta80
    term23 = 6508425 * np.sqrt(17) * beta50 ** 2 * beta80
    term24 = 3651480 * np.sqrt(221) * beta40 * beta60 * beta80
    term25 = 5494125 * np.sqrt(17) * beta60 ** 2 * beta80

    # Final terms
    term26 = 1338876 * np.sqrt(1785) * beta30 * beta70 * beta80
    term27 = 869400 * np.sqrt(2805) * beta50 * beta70 * beta80
    term28 = 5053125 * np.sqrt(17) * beta70 ** 2 * beta80
    term29 = 24386670 * beta40 * beta80 ** 2
    term30 = 5890500 * np.sqrt(13) * beta60 * beta80 ** 2
    term31 = 1603525 * np.sqrt(17) * beta80 ** 3

    # Sum of squares term
    squares_sum = 111546435 * np.sqrt(np.pi) * (
            beta10 ** 2 + beta20 ** 2 + beta30 ** 2 + beta40 ** 2 +
            beta50 ** 2 + beta60 ** 2 + beta70 ** 2 + beta80 ** 2
    )

    # Beta10 related terms
    beta10_term = 437 * beta10 * (
            21879 * np.sqrt(105) * beta20 * beta30 +
            48620 * np.sqrt(21) * beta30 * beta40 +
            7 * (
                    5525 * np.sqrt(33) * beta40 * beta50 +
                    1530 * np.sqrt(429) * beta50 * beta60 +
                    3927 * np.sqrt(65) * beta60 * beta70 +
                    3432 * np.sqrt(85) * beta70 * beta80
            )
    )

    # Beta20 related terms
    beta20_term = 23 * beta20 * (
            646646 * np.sqrt(5) * beta30 ** 2 +
            629850 * np.sqrt(5) * beta40 ** 2 +
            209950 * np.sqrt(385) * beta30 * beta50 +
            621775 * np.sqrt(5) * beta50 ** 2 +
            508725 * np.sqrt(65) * beta40 * beta60 +
            712215 * np.sqrt(33) * beta50 * beta70 +
            21 * np.sqrt(5) * (
                    29393 * beta60 ** 2 +
                    29260 * beta70 ** 2 +
                    5852 * np.sqrt(221) * beta60 * beta80 +
                    29172 * beta80 ** 2
            )
    )

    # Sum all terms
    total = (term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10 +
             term11 + term12 + term13 + term14 + term15 + term16 + term17 + term18 + term19 + term20 +
             term21 + term22 + term23 + term24 + term25 + term26 + term27 + term28 + term29 + term30 +
             term31 + squares_sum + beta10_term + beta20_term)

    # Final calculation
    volume = base_coefficient * NumberOfNucleons * r0 ** 3 * total

    # print(volume)

    return volume


def calculate_sphere_volume(Z, N):
    """
    Calculate the volume of a spherical nucleus.

    Args:
    :parameter    Z (int): The number of protons.
    :parameter    N (int): The number of neutrons.

    Returns:
    :return    float: The calculated volume of the spherical nucleus.
    """
    SphereVolume = 4 / 3 * np.pi * (Z + N) * r0 ** 3

    # print(SphereVolume)

    return SphereVolume


def calculate_volume_fixing_factor(Z, N, parameters):
    """
    Calculate the volume fixing factor to conserve volume.

    Args:
    :parameter    Z (int): The number of protons.
    :parameter    N (int): The number of neutrons.
    :parameter    parameters (tuple): A tuple of deformation parameters.

    Returns:
    :return    float: The volume fixing factor.
    """
    # Calculate the volume of the initial shape
    initial_volume = calculate_volume(Z, N, parameters)

    # Calculate the volume of the sphere
    sphere_volume = calculate_sphere_volume(Z, N)

    # Calculate the volume fixing factor
    volume_fix = (sphere_volume / initial_volume)

    # print(volume_fix)

    return volume_fix


def calculate_radius(theta, parameters, Z, N):
    """
    Calculate the nuclear radius as a function of polar angle theta.

    Args:
    :parameter    theta (np.ndarray): An array of polar angles.
    :parameter    parameters (tuple): A tuple of deformation parameters.
    :parameter    Z (int): The number of protons.
    :parameter    N (int): The number of neutrons.

    Returns:
    :return    np.ndarray: The calculated nuclear radius for each theta.
    """
    # Base shape from spherical harmonics
    radius = np.ones_like(theta)

    for harmonic_index in range(1, 9):
        # Using only the m=0 harmonics (axially symmetric)
        harmonic = np.real(sph_harm(0, harmonic_index, 0, theta))
        radius += parameters[harmonic_index - 1] * harmonic

    # Calculate radius correction factor
    radius_fix = calculate_volume_fixing_factor(Z, N, parameters) ** (1 / 3)

    # Apply A^(1/3) scaling and volume conservation
    A = Z + N
    nuclear_radius = 1.16 * (A ** (1 / 3)) * radius_fix * radius

    # Check if the calculated radius is not negative
    # if np.any(nuclear_radius < 0):
    #    print("Negative radius detected!")

    return nuclear_radius


def calculate_volume_by_integration(Z, N, parameters):
    """
    Calculate the volume of the nucleus by numerical integration.

    Args:
    :parameter Z (int): Number of protons
    :parameter N (int): Number of neutrons
    :parameter parameters (tuple): Deformation parameters (beta10, beta20, ..., beta80)

    Returns:
    :return float: Volume in fm³
    """
    # Number of points for integration
    n_theta = 200
    n_phi = 200

    # Integration variables
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    theta_mesh, phi_mesh = np.meshgrid(theta, phi)

    # Calculate R(theta) for all theta values
    R = calculate_radius(theta_mesh, parameters, Z, N)

    # Volume element in spherical coordinates: r²sin(θ)drdθdφ
    # Since we're integrating from 0 to R(θ,φ), the r integral gives us R³/3
    integrand = (R ** 3 * np.sin(theta_mesh)) / 3

    # Numerical integration using trapezoidal rule
    volume = np.trapezoid(np.trapezoid(integrand, theta, axis=1), phi)

    # print(volume)

    return volume


def main():
    """
    Main function to create and display the nuclear shape plot.
    """
    # Set up the figure
    fig = plt.figure(figsize=(15, 8))
    ax_plot = fig.add_subplot(121)
    ax_text = fig.add_subplot(122)
    ax_text.axis('off')

    # Adjust the main plot area to make room for all sliders and buttons
    plt.subplots_adjust(left=0.1, bottom=0.45, right=0.9, top=0.95)

    # Initial parameters
    num_harmonics = 8
    initial_params = (0.0,) * num_harmonics
    initial_Z = 102
    initial_N = 154
    theta = np.linspace(0, 2 * np.pi, 2000)

    # Calculate and plot initial shape
    radius = calculate_radius(theta, initial_params, initial_Z, initial_N)
    x = radius * np.sin(theta)
    y = radius * np.cos(theta)
    line, = ax_plot.plot(x, y)

    # Set up the plot
    ax_plot.set_aspect('equal')
    ax_plot.grid(True)
    ax_plot.set_title('Nuclear Shape with Volume Conservation', fontsize=18)
    ax_plot.set_xlabel('X (fm)', fontsize=18)
    ax_plot.set_ylabel('Y (fm)', fontsize=18)

    # Create a text box for volume information
    volume_text = ax_text.text(0.1, 0.4, '', fontsize=24)

    # Create sliders and button pairs
    slider_height = 0.03
    sliders = []
    decrease_buttons = []
    increase_buttons = []

    # Create sliders for Z and N with buttons
    ax_Z = plt.axes((0.25, 0.05, 0.5, 0.02))
    ax_Z_decrease = plt.axes((0.16, 0.05, 0.04, 0.02))
    ax_Z_increase = plt.axes((0.80, 0.05, 0.04, 0.02))

    ax_N = plt.axes((0.25, 0.08, 0.5, 0.02))
    ax_N_decrease = plt.axes((0.16, 0.08, 0.04, 0.02))
    ax_N_increase = plt.axes((0.80, 0.08, 0.04, 0.02))

    slider_Z = Slider(ax=ax_Z, label='Z', valmin=82, valmax=120, valinit=initial_Z, valstep=1)
    slider_N = Slider(ax=ax_N, label='N', valmin=100, valmax=180, valinit=initial_N, valstep=1)

    btn_Z_decrease = Button(ax_Z_decrease, '-')
    btn_Z_increase = Button(ax_Z_increase, '+')
    btn_N_decrease = Button(ax_N_decrease, '-')
    btn_N_increase = Button(ax_N_increase, '+')

    # Style settings for Z and N
    for slider in [slider_Z, slider_N]:
        slider.label.set_fontsize(18)
        slider.valtext.set_size(18)

    # Create sliders for deformation parameters with buttons
    for i in range(num_harmonics):
        # Create axes for slider and buttons
        ax_decrease = plt.axes((0.16, 0.11 + i * slider_height, 0.04, 0.02))
        ax_slider = plt.axes((0.25, 0.11 + i * slider_height, 0.5, 0.02))
        ax_increase = plt.axes((0.80, 0.11 + i * slider_height, 0.04, 0.02))

        # Special case for β20
        if i == 0:
            valmin, valmax = -1.6, 1.6
        elif i == 1:
            valmin, valmax = 0.0, 3.0
        else:
            valmin, valmax = -1.0, 1.0

        slider = Slider(
            ax=ax_slider,
            label=f'β{i + 1}0',
            valmin=valmin,
            valmax=valmax,
            valinit=initial_params[i],
            valstep=0.01
        )

        # Create buttons
        btn_decrease = Button(ax_decrease, '-')
        btn_increase = Button(ax_increase, '+')

        slider.label.set_size(18)
        slider.valtext.set_size(18)

        sliders.append(slider)
        decrease_buttons.append(btn_decrease)
        increase_buttons.append(btn_increase)

    # Create a button for saving the plot
    ax_save = plt.axes((0.8, 0.4, 0.1, 0.04))
    save_button = Button(ax=ax_save, label='Save Plot')

    def save_plot(_):
        parameters = [s.val for s in sliders]
        Z = int(slider_Z.val)
        N = int(slider_N.val)
        beta_values = "_".join(f"{p:.2f}" for p in parameters)
        filename = f"{Z}_{N}_{beta_values}.png"
        fig.savefig(filename)
        print(f"Plot saved as {filename}")

    save_button.on_clicked(save_plot)

    # Function to find the nearest point on the curve to a given angle
    def find_nearest_point(plot_x, plot_y, angle):
        # Calculate the angular difference between each point and the target angle
        angles = np.arctan2(plot_y, plot_x)
        angle_diff = np.abs(angles - angle)

        # Find the index of the point with the smallest angular difference
        nearest_index = np.argmin(angle_diff)

        return plot_x[nearest_index], plot_y[nearest_index]

    # Update function for the plot
    def update(_):
        parameters = [s.val for s in sliders]
        Z = int(slider_Z.val)
        N = int(slider_N.val)

        plot_radius = calculate_radius(theta, parameters, Z, N)
        plot_x = plot_radius * np.cos(theta)
        plot_y = plot_radius * np.sin(theta)
        line.set_data(plot_x, plot_y)

        # Find intersection points with the x and y axes
        x_axis_positive = find_nearest_point(plot_x, plot_y, 0)
        x_axis_negative = find_nearest_point(plot_x, plot_y, np.pi)
        y_axis_positive = find_nearest_point(plot_x, plot_y, np.pi / 2)
        y_axis_negative = find_nearest_point(plot_x, plot_y, -np.pi / 2)

        # Remove previous axis lines if they exist
        if hasattr(ax_plot, 'x_axis_line'):
            ax_plot.x_axis_line.remove()
        if hasattr(ax_plot, 'y_axis_line'):
            ax_plot.y_axis_line.remove()

        # Draw new axis lines
        ax_plot.x_axis_line = ax_plot.plot([x_axis_negative[0], x_axis_positive[0]], [x_axis_negative[1], x_axis_positive[1]], color='red')[0]
        ax_plot.y_axis_line = ax_plot.plot([y_axis_negative[0], y_axis_positive[0]], [y_axis_negative[1], y_axis_positive[1]], color='blue')[0]

        max_radius = np.max(np.abs(plot_radius)) * 1.5
        ax_plot.set_xlim(-max_radius, max_radius)
        ax_plot.set_ylim(-max_radius, max_radius)

        max_x_length = np.max(plot_y) - np.min(plot_y)
        max_y_length = np.max(plot_x) - np.min(plot_x)

        along_x_length = calculate_radius(0.0, parameters, Z, N) + calculate_radius(np.pi, parameters, Z, N)
        along_y_length = calculate_radius(np.pi / 2, parameters, Z, N) + calculate_radius(-np.pi / 2, parameters, Z, N)

        sphere_volume = calculate_sphere_volume(Z, N)
        shape_volume = calculate_volume(Z, N, parameters)
        volume_fix = calculate_volume_fixing_factor(Z, N, parameters)

        # Check if the volume calculation is correct
        shape_volume_integration = calculate_volume_by_integration(Z, N, parameters)
        if abs(sphere_volume - shape_volume_integration) > 1.0:
            print(f"Volume mismatch: {sphere_volume} vs {shape_volume_integration}")

        # Check if the calculated radius is negative
        negative_radius = False
        if np.any(plot_radius < 0):
            negative_radius = True

        volume_text.set_text(
            f'Sphere Volume: {sphere_volume:.2f} fm³\n'
            f'Shape Volume: {shape_volume:.2f} fm³\n'
            f'Volume Fixing Factor: {volume_fix:.4f}\n'
            f'Radius Fixing Factor: {volume_fix ** (1 / 3):.4f}\n'
            f'Max X Length: {max_x_length:.2f} fm\n'
            f'Max Y Length: {max_y_length:.2f} fm\n'
            f'Length Along X Axis (red): {along_x_length:.2f} fm\n'
            f'Length Along Y Axis (blue): {along_y_length:.2f} fm\n' +
            ('Negative radius detected!' if negative_radius else '')
        )

        fig.canvas.draw_idle()

    # Function to create button click handlers
    def create_button_handler(slider_counter, increment):
        def handler(_):
            new_val = slider_counter.val + increment * slider_counter.valstep
            if slider_counter.valmin <= new_val <= slider_counter.valmax:
                slider_counter.set_val(new_val)

        return handler

    # Connect button click handlers
    for i, slider in enumerate(sliders):
        decrease_buttons[i].on_clicked(create_button_handler(slider, -1))
        increase_buttons[i].on_clicked(create_button_handler(slider, 1))

    # Connect Z and N button handlers
    btn_Z_decrease.on_clicked(create_button_handler(slider_Z, -1))
    btn_Z_increase.on_clicked(create_button_handler(slider_Z, 1))
    btn_N_decrease.on_clicked(create_button_handler(slider_N, -1))
    btn_N_increase.on_clicked(create_button_handler(slider_N, 1))

    # Connect the update function to all sliders
    for slider in sliders:
        slider.on_changed(update)
    slider_Z.on_changed(update)
    slider_N.on_changed(update)

    # Update the plot once to show initial values
    update(None)

    plt.show(block=True)


if __name__ == '__main__':
    main()
