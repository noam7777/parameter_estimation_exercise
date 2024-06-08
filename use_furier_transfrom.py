import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def generate_encoder_data(theta0_true, omega_true, duration=30, dt=0.3, noise_std=0.03, junk_prob=0.05):
    """
    Generate simulated encoder data with noise and occasional junk measurements.
    """
    start_time = 0
    end_time = duration
    time_stamps = np.arange(start_time, end_time, dt)
    theta_measurements = np.mod(theta0_true + omega_true * time_stamps, 1)
    gaussian_noise = np.random.normal(0, noise_std, size=time_stamps.shape)
    theta_measurements += gaussian_noise

    random_values = np.random.rand(theta_measurements.shape[0])
    uniform_replacement = np.random.uniform(0, 1, size=theta_measurements.shape[0])
    theta_measurements[random_values < junk_prob] = uniform_replacement[random_values < junk_prob]

    return time_stamps, theta_measurements

def filter_junk_measurements(time_stamps, theta_measurements, threshold=0.25):
    filtered_theta = []
    filtered_time_stamps = []
    for i in range(4, len(theta_measurements) - 4):
        adjacent_elements = theta_measurements[i-4:i+5]
        median_adjacent = np.median(adjacent_elements)
        if abs(theta_measurements[i] - median_adjacent) <= threshold:
            filtered_theta.append(theta_measurements[i])
            filtered_time_stamps.append(time_stamps[i])
    return np.array(filtered_time_stamps), np.array(filtered_theta)

def to_cartesian_coordinates(theta_measurements):
    x = np.cos(2 * np.pi * theta_measurements)
    y = np.sin(2 * np.pi * theta_measurements)
    return x, y

def find_angular_velocity_via_fft(time_stamps, x, y):
    complex_signal = x + 1j * y
    N = len(time_stamps)
    T = (time_stamps[-1] - time_stamps[0]) / N
    
    # Perform FFT
    fft_values = np.fft.fft(complex_signal)
    fft_frequencies = np.fft.fftfreq(N, T)
    
    # Find the peak frequency
    peak_frequency = np.abs(fft_frequencies[np.argmax(np.abs(fft_values))])
    
    return peak_frequency

def unwrap_phase(theta_measurements):
    return np.unwrap(2 * np.pi * theta_measurements) / (2 * np.pi)

def linear_model(t, omega, theta0):
    return omega * t + theta0

def find_initial_angle(time_stamps, theta_measurements_unwrapped, omega_estimated):
    initial_guess = [0]  # initial guess for theta0
    params, _ = curve_fit(lambda t, theta0: linear_model(t, omega_estimated, theta0), 
                          time_stamps, theta_measurements_unwrapped, p0=initial_guess)
    theta0_estimated = params[0]
    return theta0_estimated

def estimate_encoder_parameters(time_stamps, theta_measurements):
    # Filter junk measurements
    filtered_time_stamps, filtered_theta = filter_junk_measurements(time_stamps, theta_measurements)
    
    # Convert to Cartesian coordinates
    x, y = to_cartesian_coordinates(filtered_theta)
    
    # Remove mean to mitigate the effect of initial angle
    x -= np.mean(x)
    y -= np.mean(y)
    
    # Find angular velocity using FFT
    omega_estimated = find_angular_velocity_via_fft(filtered_time_stamps, x, y)
    
    # Unwrap phase data
    theta_measurements_unwrapped = unwrap_phase(filtered_theta)
    
    # Find initial angle using curve fitting
    theta0_estimated = find_initial_angle(filtered_time_stamps, theta_measurements_unwrapped, omega_estimated)
    
    return omega_estimated, theta0_estimated

# Example usage
theta0_true = 0.4  # rounds
omega_true = 1.45  # rounds per second

# Generate data
time_stamps, theta_measurements = generate_encoder_data(theta0_true, omega_true, duration=30, dt=0.1)

# Estimate parameters
omega_estimated, theta0_estimated = estimate_encoder_parameters(time_stamps, theta_measurements)

print(f"Estimated angular velocity: {omega_estimated}")
print(f"Estimated initial angle: {theta0_estimated}")
print(f"True angular velocity: {omega_true}")
print(f"True initial angle: {theta0_true}")

# Plot the original and filtered data
filtered_time_stamps, filtered_theta = filter_junk_measurements(time_stamps, theta_measurements)
unwrapped_theta = unwrap_phase(filtered_theta)

fig1, ax1 = plt.subplots()
ax1.plot(time_stamps, theta_measurements, 'o', label='Original Data')
ax1.plot(filtered_time_stamps, unwrapped_theta, 'o', label='Filtered Unwrapped Data')
ax1.plot(filtered_time_stamps, linear_model(filtered_time_stamps, omega_estimated, theta0_estimated), 'r-', label='Fitted Model')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Theta [radians]')
ax1.legend()

plt.show()
