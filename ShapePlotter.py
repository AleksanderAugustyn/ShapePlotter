"""
Nuclear Shape Plotter - A program to visualize and analyze nuclear shapes using spherical harmonics.
This version uses only numerical integration for volume calculations and supports 20 beta parameters.
"""

import tkinter as tk
import warnings
from dataclasses import dataclass, field
from typing import Any, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider, TextBox
from numpy import dtype, ndarray
from scipy.integrate import simpson
from scipy.special import sph_harm_y

matplotlib.use('TkAgg')


@dataclass
class NuclearParameters:
    """Class to store nuclear shape parameters."""
    protons: int
    neutrons: int
    beta_values: List[float] = field(default_factory=lambda: [0.0] * 20)
    r0: float = 1.16  # Radius constant in fm

    def __post_init__(self):
        """Validate and adjust beta_values after initialization."""
        if not isinstance(self.beta_values, list):
            raise TypeError("beta_values must be a list")

        if len(self.beta_values) != 20:
            original_length = len(self.beta_values)
            if len(self.beta_values) < 20:
                # Pad with zeros if a list is too short
                self.beta_values.extend([0.0] * (20 - len(self.beta_values)))
                warnings.warn(f"beta_values list was too short (length {original_length}). Padded with zeros to length 20.")
            else:
                # Truncate if a list is too long
                self.beta_values = self.beta_values[:20]
                warnings.warn(f"beta_values list was too long (length {original_length}). Truncated to length 20.")

    @property
    def nucleons(self) -> int:
        """Total number of nucleons."""
        return self.protons + self.neutrons


class NuclearShapeCalculator:
    """Class for performing nuclear shape calculations."""

    def __init__(self, params: NuclearParameters, n_theta_points: int = 720):
        self.params = params
        self.n_theta_points = n_theta_points

    def calculate_sphere_volume(self) -> float:
        """Calculate the volume of a spherical nucleus."""
        return 4 / 3 * np.pi * self.params.nucleons * self.params.r0 ** 3

    def calculate_radius(self, theta: np.ndarray) -> np.ndarray:
        """Calculate nuclear radius as a function of polar angle theta."""
        radius = np.ones_like(theta)

        for harmonic_index in range(1, 21):  # Now up to 20 harmonics
            harmonic = np.real(sph_harm_y(harmonic_index, 0, theta, 0.0))
            radius += self.params.beta_values[harmonic_index - 1] * harmonic

        volume_fix = self.calculate_volume_fixing_factor() ** (1 / 3)
        return self.params.r0 * (self.params.nucleons ** (1 / 3)) * volume_fix * radius

    def calculate_volume_fixing_factor(self) -> float:
        """Calculate volume fixing factor to conserve volume."""
        initial_volume = self.calculate_volume_by_integration(n_theta=self.n_theta_points)
        sphere_volume = self.calculate_sphere_volume()
        return sphere_volume / initial_volume

    def calculate_volume_by_integration(self, n_theta: int = None) -> float:
        """Calculate nucleus volume by numerical integration.

        Args:
            n_theta: Number of points for theta discretization

        Returns:
            float: Volume of the nucleus in fm³
        """
        if n_theta is None:
            n_theta = self.n_theta_points

        theta = np.linspace(0, np.pi, n_theta)

        # For volume calculation without fixing factor
        radius = np.ones_like(theta)
        for harmonic_index in range(1, 21):
            harmonic = np.real(sph_harm_y(harmonic_index, 0, theta, 0.0))
            radius += self.params.beta_values[harmonic_index - 1] * harmonic

        radius *= self.params.r0 * (self.params.nucleons ** (1 / 3))

        integrand = 2 * np.pi * (radius ** 3 * np.sin(theta)) / 3

        return simpson(integrand, x=theta)

    def check_convexity(self, n_points: int = 1000) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Check if the nuclear shape is convex by analyzing its curvature.

        Args:
            n_points: Number of points for discretization

        Returns:
            tuple containing:
                - bool: True if shape is convex
                - np.ndarray: theta values where convexity was checked
                - np.ndarray: curvature values at each point
        """
        theta = np.linspace(0, np.pi, n_points)
        h = theta[1] - theta[0]  # Step size

        # Calculate radius and its derivatives
        r = self.calculate_radius(theta)

        # First derivative using a central difference
        dr = np.gradient(r, h)

        # Second derivative using a central difference
        d2r = np.gradient(dr, h)

        # Calculate curvature in polar coordinates
        # κ = (r² + 2(dr/dθ)² - r(d²r/dθ²)) / (r² + (dr/dθ)²)^(3/2)
        curvature = (r ** 2 + 2 * dr ** 2 - r * d2r) / (r ** 2 + dr ** 2) ** (3 / 2)

        # A shape is convex if its curvature is positive everywhere
        is_convex = np.all(curvature > 0)

        return is_convex, theta, dr, d2r, curvature

    def check_derivative_sign_changes(self, n_points: int = 1000) -> bool:
        """Check if the first derivative of R(theta) changes the sign at most once.

        Args:
            n_points: Number of points for discretization

        Returns:
            bool: True if the sign of the first derivative changes at most once, False otherwise.
        """
        theta = np.linspace(0, np.pi, n_points)
        h = theta[1] - theta[0]  # Step size

        # Calculate radius and its derivatives
        r = self.calculate_radius(theta)

        # First derivative using a central difference
        dr = np.gradient(r, h)

        sign_changes = 0
        previous_sign = np.sign(dr[0])
        for i in range(1, len(dr)):
            current_sign = np.sign(dr[i])
            if current_sign != 0 and current_sign != previous_sign and previous_sign != 0:
                sign_changes += 1
            if current_sign != 0:
                previous_sign = current_sign

        return sign_changes <= 1

    def check_r_cos_theta_increasing(self, n_points: int = 1000) -> bool:
        """Check if R(theta)*cos(theta) is not increasing for theta in (0, pi).

        Args:
            n_points: Number of points for discretization

        Returns:
            bool: True if R(theta)*cos(theta) is not increasing, False otherwise.
        """
        theta = np.linspace(0, np.pi, n_points)
        r = self.calculate_radius(theta)
        r_cos_theta = r * np.cos(theta)

        # Check if r_sin_theta is non-increasing
        is_non_increasing = np.all(np.diff(r_cos_theta) <= 0)

        return is_non_increasing


class ShapeAnalyzer:
    """Class for analyzing nuclear shapes and finding key measurements."""

    def __init__(self, x_coords: np.ndarray, y_coords: np.ndarray, theta_vals: np.ndarray):
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.theta_vals = theta_vals

    def find_neck_thickness(self, degree_range: Tuple[float, float]) -> tuple[float, float, float]:
        """Find neck thickness between specified degree range."""
        start_rad, end_rad = np.radians(degree_range)
        mask = (self.theta_vals >= start_rad) & (self.theta_vals <= end_rad)

        relevant_x = self.x_coords[mask]
        relevant_y = self.y_coords[mask]

        # For horizontal symmetry axis, neck thickness is measured vertically
        distances = np.abs(relevant_y)
        neck_idx = np.argmin(distances)
        neck_thickness = distances[neck_idx] * 2

        return neck_thickness, relevant_x[neck_idx], relevant_y[neck_idx]

    @staticmethod
    def find_nearest_point(plot_x: np.ndarray, plot_y: np.ndarray, angle: float) -> tuple[ndarray[tuple[int, ...], dtype[Any] | Any], ndarray[tuple[int, ...], dtype[Any] | Any]]:
        """Find the nearest point on a curve to a given angle."""
        angles = np.arctan2(plot_y, plot_x)
        angle_diff = np.abs(angles - angle)
        nearest_index = np.argmin(angle_diff)
        return plot_x[nearest_index], plot_y[nearest_index]


class NuclearShapePlotter:
    """Class for handling the plotting interface and user interaction."""

    def __init__(self):
        """Initialize the plotter with default settings."""
        # Get screen resolution and DPI
        root = tk.Tk()
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        self.screen_dpi = root.winfo_fpixels('1i')
        root.destroy()

        # Calculate base font size based on screen resolution with enhanced scaling
        base_scale = min(self.screen_width, self.screen_height) / 1080  # Scale relative to 1080p
        self.base_font_size = 14 * base_scale  # Start from a reasonable base size

        # Nuclear physics parameters
        self.initial_z = 102
        self.initial_n = 154
        self.num_harmonics = 20  # Expanded to 20
        self.n_theta_points = 720  # Default number of theta points
        self.initial_betas = [0.0] * self.num_harmonics
        self.nuclear_params = None

        # Mathematical parameters
        self.theta = np.linspace(0, np.pi, self.n_theta_points)

        # Plot elements - Main nuclear shape
        self.line = None
        self.sphere_line = None

        # Plot elements - Radius analysis
        self.radius_line = None
        self.dr_line = None
        self.d2r_line = None
        self.r_sin_theta_line = None
        self.r_cos_theta_line = None
        self.theta_radius = None

        # Matplotlib figure elements
        self.fig = None
        self.ax_plot = None
        self.ax_radius = None
        self.ax_text = None

        # Text display elements
        self.volume_text = None
        self.error_text = None

        # UI Controls - Sliders
        self.slider_z = None
        self.slider_n = None
        self.sliders = None

        # UI Controls - Buttons
        self.btn_z_increase = None
        self.btn_z_decrease = None
        self.btn_n_increase = None
        self.btn_n_decrease = None
        self.increase_buttons = None
        self.decrease_buttons = None
        self.reset_button = None
        self.save_button = None

        # UI Controls - Text input
        self.text_box = None
        self.submit_button = None
        self.root = None

        # Initialize all components
        self.setup_initial_parameters()
        self.create_figure()
        self.setup_controls()
        self.setup_event_handlers()

    def setup_initial_parameters(self):
        """Initialize default parameters."""
        self.num_harmonics = 20
        self.initial_z = 102
        self.initial_n = 154
        self.initial_betas = [0.0] * self.num_harmonics
        self.theta = np.linspace(0, np.pi, self.n_theta_points)

        self.nuclear_params = NuclearParameters(
            protons=self.initial_z,
            neutrons=self.initial_n,
            beta_values=self.initial_betas
        )

        self.sliders = []
        self.decrease_buttons = []
        self.increase_buttons = []

    def get_font_size(self, size_factor=1.0):
        """Calculate font size based on window size and screen resolution."""
        # Get the current figure size in pixels
        fig_width_px = self.fig.get_size_inches()[0] * self.fig.dpi
        fig_height_px = self.fig.get_size_inches()[1] * self.fig.dpi

        # Calculate a window scaling factor
        window_scale = min(fig_width_px / self.screen_width, fig_height_px / self.screen_height)

        # Enhanced scaling that prevents fonts from getting too small
        scale = max(window_scale, 0.5)  # Minimum scale factor of 0.5

        return self.base_font_size * scale * size_factor

    def create_figure(self):
        """Create and set up the matplotlib figure."""
        self.fig = plt.figure(figsize=(20, 10))
        gs = self.fig.add_gridspec(ncols=3, width_ratios=[1, 1, 1.2])

        # Create three subplots using gridspec
        self.ax_radius = self.fig.add_subplot(gs[0])  # R(θ) plot with derivatives
        self.ax_plot = self.fig.add_subplot(gs[1])  # Nuclear shape plot
        self.ax_text = self.fig.add_subplot(gs[2])  # Text information

        self.ax_text.axis('off')

        plt.subplots_adjust(left=0.05, bottom=0.35, right=0.95, top=0.98, wspace=0.2)

        # Create a text box for volume information
        self.volume_text = self.ax_text.text(0.0, 0.25, '', fontsize=self.get_font_size(1.6))

        # Add keyboard input instructions
        self.ax_text.text(0.00, 0.22, 'Keyboard Input Format (works with Ctrl+V):\n'
                                      'Z N β10 β20 β30 β40 β50 β60 β70 β80 β90 β100 β110 β120 β130 β140 β150 β160 β170 β180 β190 β200\n'
                                      'Example: 102 154 0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0',
                          fontsize=self.get_font_size(0.8), verticalalignment='top')

        # Add error message text (initially empty)
        self.error_text = self.ax_text.text(0.02, 0.15, '', color='red', fontsize=self.get_font_size(0.8),
                                            verticalalignment='top')

        # Set up the radius plot with a range from 0 to pi
        self.ax_radius.set_aspect('auto')
        self.ax_radius.grid(True)
        self.ax_radius.set_title('R(θ) and Derivatives', fontsize=self.get_font_size(1.2))
        self.ax_radius.set_xlabel('θ (radians)', fontsize=self.get_font_size(1.0))
        self.ax_radius.set_ylabel('Value', fontsize=self.get_font_size(1.0))
        self.ax_radius.set_xlim(0, np.pi)
        self.theta_radius = np.linspace(0, np.pi, self.n_theta_points)

        # Set up the main plot
        self.ax_plot.set_aspect('equal')
        self.ax_plot.grid(True)
        self.ax_plot.set_title('Nuclear Shape with Volume Conservation', fontsize=self.get_font_size(1.2))
        self.ax_plot.set_xlabel('X (fm)', fontsize=self.get_font_size(1.0))
        self.ax_plot.set_ylabel('Y (fm)', fontsize=self.get_font_size(1.0))

        # Initialize the plots
        calculator = NuclearShapeCalculator(self.nuclear_params, self.n_theta_points)
        radius = calculator.calculate_radius(self.theta)

        # Calculate derivatives
        h = self.theta_radius[1] - self.theta_radius[0]
        dr = np.gradient(radius, h)
        d2r = np.gradient(dr, h)

        # Create full shape for plotting by mirroring
        # Use cos for x and sin for y to make x-axis the symmetry axis
        x_upper = radius * np.cos(self.theta)
        y_upper = radius * np.sin(self.theta)

        # Mirror across the x-axis (horizontal) to create full shape
        x_full = np.concatenate([x_upper, x_upper[::-1]])
        y_full = np.concatenate([y_upper, -y_upper[::-1]])

        self.line, = self.ax_plot.plot(x_full, y_full)
        self.radius_line, = self.ax_radius.plot(self.theta_radius, radius, label='R(θ)', color='blue', linewidth=2.0, linestyle='solid')
        self.dr_line, = self.ax_radius.plot(self.theta_radius, dr, label='dR/dθ', color='red', linewidth=2.0, linestyle='dotted')
        self.d2r_line, = self.ax_radius.plot(self.theta_radius, d2r, label='d²R/dθ²', color='green', linewidth=2.0, linestyle='dashdot')
        self.r_cos_theta_line, = self.ax_radius.plot(self.theta_radius, radius * np.cos(self.theta_radius), label='R(θ)cos(θ)', color='orange', linewidth=2.0, linestyle='dashed')
        self.r_sin_theta_line, = self.ax_radius.plot(self.theta_radius, radius * np.sin(self.theta_radius), label='R(θ)sin(θ)', color='purple', linewidth=2.0, linestyle=(0, (5, 10)))  # loosely dashed

        self.ax_radius.legend(fontsize=12)

    def setup_controls(self):
        """Set up all UI controls."""
        self.create_proton_neutron_controls()
        self.create_beta_controls()
        self.create_action_buttons()
        self.create_text_input()

    def create_proton_neutron_controls(self):
        """Create controls for proton and neutron numbers."""
        # Proton controls
        ax_z = plt.axes((0.25, 0.00, 0.5, 0.015))
        ax_z_decrease = plt.axes((0.16, 0.00, 0.04, 0.015))
        ax_z_increase = plt.axes((0.80, 0.00, 0.04, 0.015))

        self.slider_z = Slider(ax=ax_z, label='Z', valmin=82, valmax=120,
                               valinit=self.initial_z, valstep=1)
        self.btn_z_decrease = Button(ax_z_decrease, '-')
        self.btn_z_increase = Button(ax_z_increase, '+')

        # Neutron controls
        ax_n = plt.axes((0.25, 0.02, 0.5, 0.015))
        ax_n_decrease = plt.axes((0.16, 0.02, 0.04, 0.015))
        ax_n_increase = plt.axes((0.80, 0.02, 0.04, 0.015))

        self.slider_n = Slider(ax=ax_n, label='N', valmin=100, valmax=180,
                               valinit=self.initial_n, valstep=1)
        self.btn_n_decrease = Button(ax_n_decrease, '-')
        self.btn_n_increase = Button(ax_n_increase, '+')

        # Style settings
        for slider in [self.slider_z, self.slider_n]:
            slider.label.set_fontsize(self.get_font_size(1.2))
            slider.valtext.set_fontsize(self.get_font_size(1.2))

    def create_beta_controls(self):
        """Create controls for beta parameters."""
        slider_height = 0.015  # Reduced height to fit 20 sliders

        for i in range(self.num_harmonics):
            ax_decrease = plt.axes((0.16, 0.04 + i * slider_height, 0.04, 0.012))
            ax_slider = plt.axes((0.25, 0.04 + i * slider_height, 0.5, 0.012))
            ax_increase = plt.axes((0.80, 0.04 + i * slider_height, 0.04, 0.012))

            valmin, valmax = (-1.6, 1.6) if i == 0 else (0.0, 4.0) if i == 1 else (-2.0, 2.0)

            slider = Slider(
                ax=ax_slider,
                label=f'β{i + 1}0',
                valmin=valmin,
                valmax=valmax,
                valinit=self.initial_betas[i],
                valstep=0.01
            )

            btn_decrease = Button(ax_decrease, '-')
            btn_increase = Button(ax_increase, '+')

            slider.label.set_fontsize(14)
            slider.valtext.set_fontsize(14)

            self.sliders.append(slider)
            self.decrease_buttons.append(btn_decrease)
            self.increase_buttons.append(btn_increase)

    def create_action_buttons(self):
        """Create save and reset buttons."""
        ax_save = plt.axes((0.75, 0.32, 0.1, 0.025))
        self.save_button = Button(ax=ax_save, label='Save Plot')

        ax_reset = plt.axes((0.86, 0.32, 0.1, 0.025))
        self.reset_button = Button(ax=ax_reset, label='Reset')

    def create_text_input(self):
        """Create text input field and submit button."""
        ax_input = plt.axes((0.25, 0.29, 0.5, 0.02))
        self.text_box = TextBox(ax_input, 'Parameters')
        self.text_box.label.set_fontsize(12)

        ax_submit = plt.axes((0.80, 0.29, 0.1, 0.02))
        self.submit_button = Button(ax_submit, 'Submit')

        # Enable key events for the text box
        text_box_widget = self.text_box.ax.figure.canvas.get_tk_widget()
        self.root = text_box_widget.master
        self.root.bind_all('<Control-v>', self.handle_paste)

    def handle_paste(self, _):
        """Handle paste events from the clipboard."""
        try:
            clipboard_text = self.root.clipboard_get()
            self.text_box.set_val(clipboard_text)
            return "break"  # Prevents default paste behavior
        except Exception as e:
            print(f"Error pasting from clipboard: {e}")
            return "break"

    def setup_event_handlers(self):
        """Set up all event handlers for controls."""
        # Connect slider update functions
        for slider in self.sliders:
            slider.on_changed(self.update_plot)
        self.slider_z.on_changed(self.update_plot)
        self.slider_n.on_changed(self.update_plot)

        # Connect button handlers
        for i, slider in enumerate(self.sliders):
            self.decrease_buttons[i].on_clicked(self.create_button_handler(slider, -1))
            self.increase_buttons[i].on_clicked(self.create_button_handler(slider, 1))

        self.btn_z_decrease.on_clicked(self.create_button_handler(self.slider_z, -1))
        self.btn_z_increase.on_clicked(self.create_button_handler(self.slider_z, 1))
        self.btn_n_decrease.on_clicked(self.create_button_handler(self.slider_n, -1))
        self.btn_n_increase.on_clicked(self.create_button_handler(self.slider_n, 1))

        # Connect action buttons
        self.submit_button.on_clicked(self.submit_parameters)
        self.save_button.on_clicked(self.save_plot)
        self.reset_button.on_clicked(self.reset_values)

    @staticmethod
    def create_button_handler(slider_obj: Slider, increment: int):
        """Create a button click handler for a slider object."""

        def handler(_):
            """Handle button click event."""
            new_val = slider_obj.val + increment * slider_obj.valstep
            if slider_obj.valmin <= new_val <= slider_obj.valmax:
                slider_obj.set_val(new_val)

        return handler

    def reset_values(self, _):
        """Reset all sliders to their initial values."""
        for slider in self.sliders:
            slider.set_val(0.0)
        self.slider_z.set_val(self.initial_z)
        self.slider_n.set_val(self.initial_n)
        self.text_box.set_val('')

    def submit_parameters(self, _):
        """Handle parameter submission from text input."""
        try:
            values = [float(val) for val in self.text_box.text.split()]
            if len(values) != 22:  # 2 for Z,N + 20 for betas
                raise ValueError("Expected 22 values: Z N β10 β20 ... β200")

            # Validate Z and N ranges
            if not (82 <= values[0] <= 120 and 100 <= values[1] <= 180):
                raise ValueError("Z must be between 82-120 and N between 100-180")

            # Update Z and N sliders
            self.slider_z.set_val(int(values[0]))
            self.slider_n.set_val(int(values[1]))

            # Update beta parameter sliders
            for i, slider in enumerate(self.sliders):
                if not (slider.valmin <= values[i + 2] <= slider.valmax):
                    raise ValueError(f"β{i + 1}0 must be between {slider.valmin} and {slider.valmax}")
                slider.set_val(values[i + 2])

            # Clear the text box and error message
            self.text_box.set_val('')
            self.error_text.set_text('')
            self.fig.canvas.draw_idle()

        except (ValueError, IndexError) as e:
            error_msg = f"Error: {str(e)}\nPlease use format: Z N β10 β20 ... β200"
            self.error_text.set_text(error_msg)
            self.fig.canvas.draw_idle()

    def save_plot(self, _=None):
        """Save the current plot to a file."""
        parameters = [s.val for s in self.sliders]
        number_of_protons = int(self.slider_z.val)
        number_of_neutrons = int(self.slider_n.val)
        beta_values = "_".join(f"{p:.2f}" for p in parameters)
        filename = f"{number_of_protons}_{number_of_neutrons}_{beta_values}.png"
        self.fig.savefig(filename)
        print(f"Plot saved as {filename}")

    def update_plot(self, _=None):
        """Update the plot with new parameters and calculate measurements."""
        # Get current parameters
        current_params = NuclearParameters(
            protons=int(self.slider_z.val),
            neutrons=int(self.slider_n.val),
            beta_values=[s.val for s in self.sliders]
        )

        # Calculate a new shape
        calculator = NuclearShapeCalculator(current_params, self.n_theta_points)
        plot_radius = calculator.calculate_radius(self.theta)

        # Create full shape for plotting by mirroring
        # Use cos for x and sin for y to make x-axis the symmetry axis
        x_upper = plot_radius * np.cos(self.theta)
        y_upper = plot_radius * np.sin(self.theta)

        # Mirror across the x-axis (horizontal) to create full shape
        x_full = np.concatenate([x_upper, x_upper[::-1]])
        y_full = np.concatenate([y_upper, -y_upper[::-1]])

        # Update shape plot data
        self.line.set_data(x_full, y_full)

        # Calculate derivatives for the radius plot
        h = self.theta_radius[1] - self.theta_radius[0]
        dr = np.gradient(plot_radius, h)
        d2r = np.gradient(dr, h)
        plot_r_cos_theta = plot_radius * np.cos(self.theta_radius)
        plot_r_sin_theta = plot_radius * np.sin(self.theta_radius)

        # Update radius plot data
        self.radius_line.set_data(self.theta_radius, plot_radius)
        self.dr_line.set_data(self.theta_radius, dr)
        self.d2r_line.set_data(self.theta_radius, d2r)
        self.r_cos_theta_line.set_data(self.theta_radius, plot_r_cos_theta)
        self.r_sin_theta_line.set_data(self.theta_radius, plot_r_sin_theta)

        self.ax_radius.relim()
        self.ax_radius.autoscale_view()

        # Initialize shape analyzer with full shape
        # Create full theta array for the mirrored shape
        theta_full = np.concatenate([self.theta, 2 * np.pi - self.theta[::-1]])
        analyzer = ShapeAnalyzer(x_full, y_full, theta_full)

        # Find intersection points with axes
        # For x-axis: theta = 0 (right) and theta = π (left)
        x_axis_positive = analyzer.find_nearest_point(x_full, y_full, 0)
        x_axis_negative = analyzer.find_nearest_point(x_full, y_full, np.pi)
        # For y-axis: theta = π/2 (top) and theta = 3π/2 (bottom)
        y_axis_positive = analyzer.find_nearest_point(x_full, y_full, np.pi / 2)
        y_axis_negative = analyzer.find_nearest_point(x_full, y_full, 3 * np.pi / 2)

        # Remove previous lines if they exist
        for attr in ['x_axis_line', 'y_axis_line', 'neck_line']:
            if hasattr(self.ax_plot, attr):
                getattr(self.ax_plot, attr).remove()

        # Draw axis lines
        self.ax_plot.x_axis_line = self.ax_plot.plot(
            [x_axis_negative[0], x_axis_positive[0]],
            [x_axis_negative[1], x_axis_positive[1]],
            color='red',
            label='X Axis'
        )[0]

        self.ax_plot.y_axis_line = self.ax_plot.plot(
            [y_axis_negative[0], y_axis_positive[0]],
            [y_axis_negative[1], y_axis_positive[1]],
            color='blue',
            label='Y Axis'
        )[0]

        # Calculate and draw necks
        neck_thickness_45_135, neck_x_45_135, neck_y_45_135 = analyzer.find_neck_thickness(
            (45, 135)
        )
        neck_thickness_30_150, neck_x_30_150, neck_y_30_150 = analyzer.find_neck_thickness(
            (30, 150)
        )

        # Remove previous neck lines
        for attr in ['neck_line_45_135', 'neck_line_30_150']:
            if hasattr(self.ax_plot, attr):
                getattr(self.ax_plot, attr).remove()

        # Draw neck lines (vertical lines for horizontal symmetry)
        self.ax_plot.neck_line_45_135 = self.ax_plot.plot(
            [neck_x_45_135, neck_x_45_135],
            [-neck_thickness_45_135 / 2, neck_thickness_45_135 / 2],
            color='green',
            linewidth=2,
            label='Neck (45-135°)'
        )[0]

        self.ax_plot.neck_line_30_150 = self.ax_plot.plot(
            [neck_x_30_150, neck_x_30_150],
            [-neck_thickness_30_150 / 2, neck_thickness_30_150 / 2],
            color='purple',
            linewidth=2,
            label='Neck (30-150°)'
        )[0]

        # Update plot limits and tick label sizes
        max_radius = np.max(np.abs(plot_radius)) * 1.5
        self.ax_plot.set_xlim(-max_radius, max_radius)
        self.ax_plot.set_ylim(-max_radius, max_radius)
        self.ax_plot.tick_params(axis='both', labelsize=self.get_font_size(0.9))
        self.ax_radius.tick_params(axis='both', labelsize=self.get_font_size(0.9))

        # Calculate measurements
        max_x_length = np.max(x_full) - np.min(x_full)
        max_y_length = np.max(y_full) - np.min(y_full)
        # Along x-axis: from theta=0 to theta=pi
        along_x_length = (calculator.calculate_radius(np.array([0.0]))[0] + calculator.calculate_radius(np.array([np.pi]))[0])
        # Along y-axis: at theta=pi/2, doubled because of symmetry
        along_y_length = 2 * calculator.calculate_radius(np.array([np.pi / 2]))[0]

        # Calculate volumes
        sphere_volume = calculator.calculate_sphere_volume()
        shape_volume_integration = calculator.calculate_volume_by_integration()

        # Check calculations
        negative_radius = np.any(plot_radius < 0)
        derivative_sign_changes_ok = calculator.check_derivative_sign_changes()
        r_cos_theta_non_increasing = calculator.check_r_cos_theta_increasing()

        # Clear the old beta plot if it exists
        if self.sphere_line is not None:
            self.sphere_line.remove()

        # Update a reference sphere
        radius_0 = current_params.r0 * (current_params.nucleons ** (1 / 3))
        sphere_theta = np.linspace(0, 2 * np.pi, 200)
        sphere_x = radius_0 * np.cos(sphere_theta)
        sphere_y = radius_0 * np.sin(sphere_theta)
        self.sphere_line, = self.ax_plot.plot(sphere_x, sphere_y, '--', color='gray', alpha=0.5, label='R₀')

        # Update information display
        self.volume_text.set_text(
            f'Sphere Volume: {sphere_volume:.4f} fm³\n'
            f'Shape Volume (Numerical): {calculator.calculate_volume_fixing_factor() * shape_volume_integration:.4f} fm³\n'
            f'Volume Fixing Factor: {calculator.calculate_volume_fixing_factor():.4f}\n'
            f'Volume Conservation: {sphere_volume:.4f} vs {shape_volume_integration:.4f} fm³\n'
            f'Number of theta points: {self.n_theta_points}\n'
            f'Max X Length: {max_x_length:.2f} fm\n'
            f'Max Y Length: {max_y_length:.2f} fm\n'
            f'Length Along X Axis (red): {along_x_length:.2f} fm\n'
            f'Length Along Y Axis (blue): {along_y_length:.2f} fm\n'
            f'Neck Thickness (45°-135°, green): {neck_thickness_45_135:.2f} fm\n'
            f'Neck Thickness (30°-150°, purple): {neck_thickness_30_150:.2f} fm\n' +
            ('Negative radius detected!\n' if negative_radius else '') +
            f'dR/dθ sign changes OK: {"✓" if derivative_sign_changes_ok else "✗"}\n' +
            f'Rcos(θ) non-increasing: {"✓" if r_cos_theta_non_increasing else "✗"}'
        )

        # Update the legend
        self.ax_plot.legend(fontsize=self.get_font_size(0.8), loc='lower left')
        self.ax_radius.legend(fontsize=self.get_font_size(0.8), loc='lower left')
        self.fig.canvas.draw_idle()

    def on_resize(self, event):
        """Handle window resize events."""
        if event.name == 'resize_event':
            # Update all font sizes
            self.ax_radius.set_title('R(θ) and Derivatives', fontsize=self.get_font_size(1.2))
            self.ax_radius.set_xlabel('θ (radians)', fontsize=self.get_font_size(1.0))
            self.ax_radius.set_ylabel('Value', fontsize=self.get_font_size(1.0))
            self.ax_radius.tick_params(axis='both', labelsize=self.get_font_size(0.9))

            self.ax_plot.set_title('Nuclear Shape with Volume Conservation', fontsize=self.get_font_size(1.2))
            self.ax_plot.set_xlabel('X (fm)', fontsize=self.get_font_size(1.0))
            self.ax_plot.set_ylabel('Y (fm)', fontsize=self.get_font_size(1.0))
            self.ax_plot.tick_params(axis='both', labelsize=self.get_font_size(0.9))

            for slider in self.sliders:
                slider.label.set_fontsize(self.get_font_size(1.2))
                slider.valtext.set_fontsize(self.get_font_size(1.2))

            self.slider_z.label.set_fontsize(self.get_font_size(1.2))
            self.slider_z.valtext.set_fontsize(self.get_font_size(1.2))
            self.slider_n.label.set_fontsize(self.get_font_size(1.2))
            self.slider_n.valtext.set_fontsize(self.get_font_size(1.2))

            self.ax_plot.legend(fontsize=self.get_font_size(0.8), loc='lower left')
            self.ax_radius.legend(fontsize=self.get_font_size(0.8), loc='lower left')

            self.volume_text.set_fontsize(self.get_font_size(1.2))
            self.ax_text.texts[1].set_fontsize(self.get_font_size(0.8))
            self.error_text.set_fontsize(self.get_font_size(0.8))

            self.fig.canvas.draw_idle()

    def run(self):
        """Start the interactive plotting interface."""
        self.update_plot()
        self.fig.canvas.mpl_connect('resize_event', self.on_resize)
        plt.show(block=True)


def main():
    """Main entry point for the application."""
    plotter = NuclearShapePlotter()
    plotter.run()


if __name__ == '__main__':
    main()