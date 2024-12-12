import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider

# Define axially symmetric spherical harmonics (X0)
def spherical_harmonic(l, theta):
    if l == 0:
        return 0.282095  # 1 / (2 * sqrt(pi))
    elif l == 1:
        return 0.488603 * np.cos(theta)  # sqrt(3 / (4 * pi)) * cos(theta)
    elif l == 2:
        return 0.315392 * (3 * np.cos(theta)**2 - 1)  # sqrt(5 / (16 * pi)) * (3 * cos(theta)^2 - 1)
    elif l == 3:
        return 0.546274 * (5 * np.cos(theta)**3 - 3 * np.cos(theta))  # sqrt(7 / (16 * pi)) * (5 * cos(theta)^3 - 3 * cos(theta))
    elif l == 4:
        return 0.373176 * (35 * np.cos(theta)**4 - 30 * np.cos(theta)**2 + 3)  # sqrt(9 / (256 * pi)) * (35 * cos(theta)^4 - 30 * cos(theta)^2 + 3)
    elif l == 5:
        return 0.590044 * (63 * np.cos(theta)**5 - 70 * np.cos(theta)**3 + 15 * np.cos(theta))  # sqrt(11 / (256 * pi)) * (63 * cos(theta)^5 - 70 * cos(theta)^3 + 15 * cos(theta))
    elif l == 6:
        return 0.418501 * (231 * np.cos(theta)**6 - 315 * np.cos(theta)**4 + 105 * np.cos(theta)**2 - 5)  # sqrt(13 / (1024 * pi)) * (231 * cos(theta)^6 - 315 * cos(theta)^4 + 105 * cos(theta)^2 - 5)
    elif l == 7:
        return 0.628069 * (429 * np.cos(theta)**7 - 693 * np.cos(theta)**5 + 315 * np.cos(theta)**3 - 35 * np.cos(theta))  # sqrt(15 / (1024 * pi)) * (429 * cos(theta)^7 - 693 * cos(theta)^5 + 315 * cos(theta)^3 - 35 * cos(theta))
    elif l == 8:
        return 0.456946 * (6435 * np.cos(theta)**8 - 12012 * np.cos(theta)**6 + 6930 * np.cos(theta)**4 - 1260 * np.cos(theta)**2 + 35)  # sqrt(17 / (16384 * pi)) * (6435 * cos(theta)^8 - 12012 * cos(theta)^6 + 6930 * cos(theta)^4 - 1260 * cos(theta)^2 + 35)
    else:
        return 0

# Calculate radius
def calculate_radius(theta, params):
    radius = 1.0  # Base radius
    for l, param in enumerate(params):
        radius += param * spherical_harmonic(l, theta)
    return radius

# Create plot
def plot_shape(params):
    theta = np.linspace(0, 2 * np.pi, 1000)
    radius = calculate_radius(theta, params)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    plt.figure(figsize=(6, 6))
    plt.plot(x, y)
    plt.fill(x, y, alpha=0.3)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.title("2D Shape Plot")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

# Create sliders
initial_params = [0.0] * 9
sliders = [FloatSlider(min=-1.0, max=1.0, step=0.01, value=initial_params[i], description=f'β{i}') for i in range(9)]

# Interactive plot
interact(plot_shape, params=sliders)
