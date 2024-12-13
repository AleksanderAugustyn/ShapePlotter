
# ShapePlotter

## Goal

The goal of ShapePlotter is to calculate and plot the shape of atomic nuclei based on user-specified parameters. It uses a model that describes the radius of a nucleus as a function of spherical coordinates, allowing for the visualization of various nuclear shapes, including spherical, deformed, and exotic shapes. Additionally, the program now includes the capability to save the generated plots to PNG files.

## Structure

The program is structured as follows:

-   **`main()` function**:
    -   Parses command-line arguments for the atomic number (Z), neutron number (N), and deformation parameters.
    -   Calculates the nuclear volume using different methods based on the provided arguments.
    -   Calculates the nuclear radius as a function of the polar angle (theta).
    -   Plots the nuclear shape in 2D or 3D, depending on the user's choice.
-   **`save_plot()` function**: Saves the generated plot to a PNG file with a filename based on the current parameters.
-   **`calculate_volume()` function**: Calculates the volume of the nucleus using a numerical integration method.
-   **`calculate_sphere_volume()` function**: Calculates the volume of a spherical nucleus with the same Z and N.
-   **`calculate_volume_fixing_factor()` function**: Calculates the nuclear volume while keeping a specific deformation parameter fixed.
-   **`calculate_radius()` function**: Calculates the nuclear radius as a function of theta, taking into account the deformation parameters.

The program uses the `matplotlib` library for plotting and `numpy` for numerical calculations.
