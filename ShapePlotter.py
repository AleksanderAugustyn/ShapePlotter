import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.special import sph_harm

matplotlib.use('TkAgg')

# Constants
r0 = 1.16  # Radius constant in fm


def CalculateVolume(Z, N, Parameters):
    """Calculate the volume of the shape using an analytical equation."""
    NumberOfNucleons = Z + N
    Beta10, Beta20, Beta30, Beta40, Beta50, Beta60, Beta70, Beta80 = Parameters

    # Base coefficient
    BaseCoeff = 1 / (111546435 * np.sqrt(np.pi))

    # Main terms
    Term1 = 148728580 * np.pi ** (3 / 2)
    Term2 = 22309287 * np.sqrt(5) * Beta10 ** 2 * Beta20
    Term3 = 5311735 * np.sqrt(5) * Beta20 ** 3
    Term4 = 47805615 * Beta20 ** 2 * Beta40
    Term5 = 30421755 * Beta30 ** 2 * Beta40
    Term6 = 9026235 * Beta40 ** 3
    Term7 = 6686100 * np.sqrt(77) * Beta30 * Beta40 * Beta50
    Term8 = 25741485 * Beta40 * Beta50 ** 2
    Term9 = 13000750 * np.sqrt(13) * Beta30 ** 2 * Beta60
    Term10 = 7800450 * np.sqrt(13) * Beta40 ** 2 * Beta60

    # Additional terms
    Term11 = 1820105 * np.sqrt(1001) * Beta30 * Beta50 * Beta60
    Term12 = 6729800 * np.sqrt(13) * Beta50 ** 2 * Beta60
    Term13 = 25053210 * Beta40 * Beta60 ** 2
    Term14 = 2093000 * np.sqrt(13) * Beta60 ** 3
    Term15 = 9100525 * np.sqrt(105) * Beta30 * Beta40 * Beta70

    # More complex terms
    Term16 = 4282600 * np.sqrt(165) * Beta40 * Beta50 * Beta70
    Term17 = 1541736 * np.sqrt(1365) * Beta30 * Beta60 * Beta70
    Term18 = 1014300 * np.sqrt(2145) * Beta50 * Beta60 * Beta70
    Term19 = 24647490 * Beta40 * Beta70 ** 2
    Term20 = 6037500 * np.sqrt(13) * Beta60 * Beta70 ** 2

    # Beta80 terms
    Term21 = 11241825 * np.sqrt(17) * Beta40 ** 2 * Beta80
    Term22 = 2569560 * np.sqrt(1309) * Beta30 * Beta50 * Beta80
    Term23 = 6508425 * np.sqrt(17) * Beta50 ** 2 * Beta80
    Term24 = 3651480 * np.sqrt(221) * Beta40 * Beta60 * Beta80
    Term25 = 5494125 * np.sqrt(17) * Beta60 ** 2 * Beta80

    # Final terms
    Term26 = 1338876 * np.sqrt(1785) * Beta30 * Beta70 * Beta80
    Term27 = 869400 * np.sqrt(2805) * Beta50 * Beta70 * Beta80
    Term28 = 5053125 * np.sqrt(17) * Beta70 ** 2 * Beta80
    Term29 = 24386670 * Beta40 * Beta80 ** 2
    Term30 = 5890500 * np.sqrt(13) * Beta60 * Beta80 ** 2
    Term31 = 1603525 * np.sqrt(17) * Beta80 ** 3

    # Sum of squares term
    SquaresSum = 111546435 * np.sqrt(np.pi) * (
            Beta10 ** 2 + Beta20 ** 2 + Beta30 ** 2 + Beta40 ** 2 +
            Beta50 ** 2 + Beta60 ** 2 + Beta70 ** 2 + Beta80 ** 2
    )

    # Beta10 related terms
    Beta10Term = 437 * Beta10 * (
            21879 * np.sqrt(105) * Beta20 * Beta30 +
            48620 * np.sqrt(21) * Beta30 * Beta40 +
            7 * (
                    5525 * np.sqrt(33) * Beta40 * Beta50 +
                    1530 * np.sqrt(429) * Beta50 * Beta60 +
                    3927 * np.sqrt(65) * Beta60 * Beta70 +
                    3432 * np.sqrt(85) * Beta70 * Beta80
            )
    )

    # Beta20 related terms
    Beta20Term = 23 * Beta20 * (
            646646 * np.sqrt(5) * Beta30 ** 2 +
            629850 * np.sqrt(5) * Beta40 ** 2 +
            209950 * np.sqrt(385) * Beta30 * Beta50 +
            621775 * np.sqrt(5) * Beta50 ** 2 +
            508725 * np.sqrt(65) * Beta40 * Beta60 +
            712215 * np.sqrt(33) * Beta50 * Beta70 +
            21 * np.sqrt(5) * (
                    29393 * Beta60 ** 2 +
                    29260 * Beta70 ** 2 +
                    5852 * np.sqrt(221) * Beta60 * Beta80 +
                    29172 * Beta80 ** 2
            )
    )

    # Sum all terms
    total = (term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10 +
             term11 + term12 + term13 + term14 + term15 + term16 + term17 + term18 + term19 + term20 +
             term21 + term22 + term23 + term24 + term25 + term26 + term27 + term28 + term29 + term30 +
             term31 + squares_sum + beta10_term + beta20_term)

    # Final calculation
    Volume = BaseCoeff * NumberOfNucleons * r0 ** 3 * Total

    # print(volume)

    return Volume


def CalculateSphereVolume(Z, N):
    """Calculate the volume of a sphere using the formula for a sphere."""
    SphereVolume = 4 / 3 * np.pi * (Z + N) * r0 ** 3

    # print(SphereVolume)

    return SphereVolume


def CalculateVolumeFixingFactor(Z, N, Parameters):
    """Calculate the volume fixing factor for the shape."""
    # Calculate the volume of the initial shape
    InitialVolume = CalculateVolume(Z, N, Parameters)

    # Calculate the volume of the sphere
    SphereVolume = CalculateSphereVolume(Z, N)

    # Calculate the volume fixing factor
    VolumeFix = (SphereVolume / InitialVolume) ** (1 / 3)

    # print(volume_fix)

    return VolumeFix


def CalculateRadius(Theta, Parameters, Z, N):
    """Calculate the radius for each angle using spherical harmonics with volume conservation."""
    # Base shape from spherical harmonics
    Radius = np.ones_like(Theta)

    for harmonic_index in range(1, 9):
        # Using only the m=0 harmonics (axially symmetric)
        harmonic = np.real(sph_harm(0, harmonic_index, 0, theta))
        radius += parameters[harmonic_index - 1] * harmonic

    # Calculate volume correction factor
    VolumeFix = CalculateVolumeFixingFactor(Z, N, Parameters)

    # Apply A^(1/3) scaling and volume conservation
    A = Z + N
    NuclearRadius = 1.16 * (A ** (1 / 3)) * VolumeFix * Radius

    return NuclearRadius


def main():
    # Set up the figure
    Fig = plt.figure(figsize=(12, 8))
    AxPlot = plt.subplot(111)

    # Adjust the main plot area to make room for all sliders
    plt.subplots_adjust(left=0.1, bottom=0.45, right=0.9, top=0.95)

    # Initial parameters
    NumHarmonics = 8
    InitialParams = (0.0,) * NumHarmonics
    InitialZ = 102
    InitialN = 154
    Theta = np.linspace(0, 2 * np.pi, 2000)  # Note: Changed to [0, π] for proper volume calculation

    # Calculate and plot initial shape
    radius = calculate_radius(theta, initial_params, initial_Z, initial_N)
    x = radius * np.sin(theta)  # Note: Using sin(θ) for x and cos(θ) for y to match standard convention
    y = radius * np.cos(theta)
    line, = ax_plot.plot(x, y)

    # Set up the plot
    ax_plot.set_aspect('equal')
    ax_plot.grid(True)
    ax_plot.set_title('Nuclear Shape with Volume Conservation')
    ax_plot.set_xlabel('X (fm)')
    ax_plot.set_ylabel('Y (fm)')

    # Create sliders for deformation parameters
    SliderHeight = 0.03
    Sliders = []

    # Create sliders for Z and N
    AxZ = plt.axes((0.2, 0.05, 0.6, 0.02))
    AxN = plt.axes((0.2, 0.08, 0.6, 0.02))

    slider_Z = Slider(ax=ax_Z, label='Z', valmin=82, valmax=120, valinit=initial_Z, valstep=1)
    slider_N = Slider(ax=ax_N, label='N', valmin=100, valmax=180, valinit=initial_N, valstep=1)

    # Create sliders for deformation parameters
    for i in range(num_harmonics):
        ax = plt.axes((0.2, 0.11 + i * slider_height, 0.6, 0.02))

        # Special case for β20
        if i == 1:
            Valmin, Valmax = 0.0, 3.0
        else:
            Valmin, Valmax = -1.0, 1.0

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
        Z = slider_Z.val
        N = slider_N.val

        radius = calculate_radius(theta, parameters, Z, N)
        x = radius * np.sin(theta)
        y = radius * np.cos(theta)
        line.set_data(x, y)

        # Update plot limits to accommodate shape changes
        max_radius = np.max(np.abs(radius)) * 1.5
        ax_plot.set_xlim(-max_radius, max_radius)
        ax_plot.set_ylim(-max_radius, max_radius)
        fig.canvas.draw_idle()

    # Connect the update function to all sliders
    for slider in sliders:
        slider.on_changed(update)
    slider_Z.on_changed(update)
    slider_N.on_changed(update)

    plt.show(block=True)


if __name__ == '__main__':
    main()
