
# ShapePlotter

## Goal

The goal of ShapePlotter is to calculate and plot the shape of atomic nuclei based on user-specified parameters. It uses a model that describes the radius of a nucleus as a function of spherical coordinates, allowing for the visualization of various nuclear shapes, including spherical, deformed, and exotic shapes. The program ensures volume conservation during deformation and checks for non-physical negative radius values. Additionally, it features an interactive interface with sliders and buttons for adjusting parameters and includes the capability to save the generated plots to PNG files.

## Structure

The program is structured as follows:

-   **`main()` function**:
    -   Sets up an interactive plot with sliders and buttons for adjusting the atomic number (Z), neutron number (N), and deformation parameters.
    -   Calculates the nuclear volume using numerical integration and ensures volume conservation.
    -   Calculates the nuclear radius as a function of the polar angle (theta), with checks to prevent negative radius values.
    -   Plots the nuclear shape in 2D and updates the plot in response to user interactions.
-   **`save_plot()` function**: An event handler connected to a button that saves the generated plot to a PNG file with a filename based on the current parameters.
-   **`calculate_volume()` function**: Calculates the volume of the nucleus using a numerical integration method.
-   **`calculate_sphere_volume()` function**: Calculates the volume of a spherical nucleus with the same Z and N.
-   **`calculate_volume_fixing_factor()` function**: Calculates the volume fixing factor to conserve volume during deformation.
-   **`calculate_radius()` function**: Calculates the nuclear radius as a function of theta, taking into account the deformation parameters and volume conservation.
-   **`update()` function**: This function is connected to the sliders and is called whenever a slider's value changes. It updates the plot with the new shape of the nucleus based on the current values of Z, N, and the deformation parameters. It also recalculates and displays the volume, volume fixing factor, radius fixing factor, and checks for negative radius values.
-   **`create_button_handler()` function**: This function creates event handlers for the buttons associated with each slider. It allows for incrementing or decrementing the slider values by a fixed step, providing an alternative way to adjust the parameters.

The program uses the `matplotlib` library for plotting and `numpy` for numerical calculations.
