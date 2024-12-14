# Nuclear Shape Plotter

## Overview
ShapePlotter is a Python program designed to visualize and analyze nuclear shapes using spherical harmonics parametrization. The program provides an interactive interface for exploring how different deformation parameters (β10 through β80) affect nuclear shapes while maintaining volume conservation. This tool is particularly useful for nuclear physics research and education, allowing users to study nuclear deformations and their properties.

## Features
- Interactive visualization of nuclear shapes
- Real-time adjustment of deformation parameters (β10-β80)
- Volume conservation calculations
- Neck thickness measurements
- Automatic axis scaling
- Length measurements along major axes
- Proton and neutron number adjustment
- Save functionality for generated plots

## Dependencies
- matplotlib
- numpy
- scipy (for spherical harmonics calculations)

## Technical Details

### Core Functions

#### Volume Calculations
- `calculate_volume(number_of_protons, number_of_neutrons, parameters)`: Computes the volume of a deformed nucleus using analytical expressions
- `calculate_sphere_volume(number_of_protons, number_of_neutrons)`: Calculates the volume of an equivalent spherical nucleus
- `calculate_volume_by_integration(number_of_protons, number_of_neutrons, parameters)`: Performs numerical integration to verify volume calculations
- `calculate_volume_fixing_factor(number_of_protons, number_of_neutrons, parameters)`: Determines the scaling factor needed for volume conservation

#### Shape Calculations
- `calculate_radius(theta, parameters, number_of_protons, number_of_neutrons)`: Computes the nuclear radius as a function of angle using spherical harmonics
- `find_neck_thickness(x_coords, y_coords, theta_vals)`: Determines the thickness of the nuclear neck in deformed configurations

### User Interface Functions
- `main()`: Sets up the interactive plotting interface with sliders and buttons
- `update()`: Handles real-time updates of the plot based on parameter changes
- `find_nearest_point(plot_x, plot_y, angle)`: Helper function for finding points on the nuclear surface
- `reset_values()`: Resets all parameters to their initial values
- `save_plot()`: Saves the current plot configuration to a file

## Usage
1. Launch the program by running `python ShapePlotter.py`
2. Use the sliders to adjust:
   - Proton number (Z)
   - Neutron number (N)
   - Deformation parameters (β10-β80)
3. The plot updates in real-time, showing:
   - The nuclear shape (black outline)
   - Major axis measurements (red and blue lines)
   - Neck thickness (green line)
4. Additional information is displayed on the right panel:
   - Volume calculations
   - Length measurements
   - Warning messages for invalid configurations

## Physical Constraints
- Proton number (Z): 82-120
- Neutron number (N): 100-180
- β20 parameter: 0.0-3.0
- Other β parameters: -1.0-1.0 (except β10: -1.6-1.6)

The program automatically ensures volume conservation and provides warnings for physically unrealistic configurations such as negative radii or volume mismatches.