import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.special import sph_harm

matplotlib.use('TkAgg')

# Constants
r0 = 1.16  # Radius constant in fm


def calculate_volume(Z, N, parameters):
    """Calculate the volume of the shape using an analytical equation."""
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
    """Calculate the volume of a sphere using the formula for a sphere."""
    SphereVolume = 4 / 3 * np.pi * (Z + N) * r0 ** 3

    # print(SphereVolume)

    return SphereVolume


def calculate_volume_fixing_factor(Z, N, parameters):
    """Calculate the volume fixing factor for the shape."""
    # Calculate the volume of the initial shape
    initial_volume = calculate_volume(Z, N, parameters)

    # Calculate the volume of the sphere
    sphere_volume = calculate_sphere_volume(Z, N)

    # Calculate the volume fixing factor
    volume_fix = (sphere_volume / initial_volume) ** (1 / 3)

    # print(volume_fix)

    return volume_fix


def calculate_radius(theta, parameters, Z, N):
    """Calculate the radius for each angle using spherical harmonics with volume conservation."""
    # Base shape from spherical harmonics
    radius = np.ones_like(theta)

    for harmonic_index in range(1, 9):
        # Using only the m=0 harmonics (axially symmetric)
        harmonic = np.real(sph_harm(0, harmonic_index, 0, theta))
        radius += parameters[harmonic_index - 1] * harmonic

    # Calculate volume correction factor
    VolumeFix = calculate_volume_fixing_factor(Z, N, parameters)

    # Apply A^(1/3) scaling and volume conservation
    A = Z + N
    nuclear_radius = 1.16 * (A ** (1 / 3)) * VolumeFix * radius

    return nuclear_radius


def main():
    # Set up the figure
    fig = plt.figure(figsize=(15, 8))
    ax_plot = fig.add_subplot(121)
    ax_text = fig.add_subplot(122)
    ax_text.axis('off')

    # Adjust the main plot area to make room for all sliders
    plt.subplots_adjust(left=0.1, bottom=0.45, right=0.9, top=0.95)

    # Initial parameters
    num_harmonics = 8
    initial_params = (0.0,) * num_harmonics
    initial_Z = 102
    initial_N = 154
    theta = np.linspace(0, 2 * np.pi, 2000)  # Note: Changed to [0, π] for proper volume calculation

    # Calculate and plot initial shape
    radius = calculate_radius(theta, initial_params, initial_Z, initial_N)
    x = radius * np.sin(theta)  # Note: Using sin(θ) for x and cos(θ) for y to match standard convention
    y = radius * np.cos(theta)
    line, = ax_plot.plot(x, y)

    # Set up the plot
    ax_plot.set_aspect('equal')
    ax_plot.grid(True)
    ax_plot.set_title('Nuclear Shape with Volume Conservation', fontsize=18)
    ax_plot.set_xlabel('X (fm)', fontsize=18)
    ax_plot.set_ylabel('Y (fm)', fontsize=18)

    # Create a text box for volume information
    volume_text = ax_text.text(0.1, 0.7, '', fontsize=24)

    # Create sliders for deformation parameters
    slider_height = 0.03
    sliders = []

    # Create sliders for Z and N
    ax_Z = plt.axes((0.2, 0.05, 0.6, 0.02))
    ax_N = plt.axes((0.2, 0.08, 0.6, 0.02))

    slider_Z = Slider(ax=ax_Z, label='Z', valmin=82, valmax=120, valinit=initial_Z, valstep=1)
    slider_N = Slider(ax=ax_N, label='N', valmin=100, valmax=180, valinit=initial_N, valstep=1)

    slider_Z.label.set_size(18)
    slider_Z.valtext.set_size(18)

    slider_N.label.set_size(18)
    slider_N.valtext.set_size(18)

    # Create sliders for deformation parameters
    for i in range(num_harmonics):
        ax = plt.axes((0.2, 0.11 + i * slider_height, 0.6, 0.02))

        # Special case for β20
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

        slider.label.set_size(18)
        slider.valtext.set_size(18)

        sliders.append(slider)

    # Update function for the plot
    def update(val):
        parameters = [plot_slider.val for plot_slider in sliders]
        Z = slider_Z.val
        N = slider_N.val

        plot_radius = calculate_radius(theta, parameters, Z, N)
        plot_x = plot_radius * np.sin(theta)
        plot_y = (plot_radius * np.cos(theta))
        line.set_data(plot_x, plot_y)

        # Update plot limits to accommodate shape changes
        # Update plot limits to accommodate shape changes
        max_radius = np.max(np.abs(plot_radius)) * 1.5
        ax_plot.set_xlim(-max_radius, max_radius)
        ax_plot.set_ylim(-max_radius, max_radius)

        # Update volume information
        sphere_volume = calculate_sphere_volume(Z, N)
        shape_volume = calculate_volume(Z, N, parameters)
        volume_fix = calculate_volume_fixing_factor(Z, N, parameters)

        volume_text.set_text(
            f'Sphere Volume: {sphere_volume:.2f} fm³\n'
            f'Shape Volume: {shape_volume:.2f} fm³\n'
            f'Volume Fixing Factor: {volume_fix:.4f}'
        )

        fig.canvas.draw_idle()

    # Connect the update function to all sliders
    for slider in sliders:
        slider.on_changed(update)
    slider_Z.on_changed(update)
    slider_N.on_changed(update)

    plt.show(block=True)


if __name__ == '__main__':
    main()
