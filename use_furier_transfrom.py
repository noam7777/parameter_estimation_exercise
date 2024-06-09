import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def generate_encoder_data(theta0_true, omega_true, duration=30, noise_std=0.03, junk_prob=0.05):
    """
    Generates encoder data with random time intervals.
    """
    start_time = 0
    time_stamps = [start_time]
    
    while time_stamps[-1] < duration:
        next_dt = np.random.uniform(0, 1)
        next_time = time_stamps[-1] + next_dt
        if next_time >= duration:
            break
        time_stamps.append(next_time)
    
    time_stamps = np.array(time_stamps)
    
    theta_measurements = np.mod(theta0_true + omega_true * time_stamps, 1)
    gaussian_noise = np.random.normal(0, noise_std, size=time_stamps.shape)
    theta_measurements += gaussian_noise

    random_values = np.random.rand(theta_measurements.shape[0])
    uniform_replacement = np.random.uniform(0, 1, size=theta_measurements.shape[0])
    theta_measurements[random_values < junk_prob] = uniform_replacement[random_values < junk_prob]

    return time_stamps, theta_measurements

def to_cartesian_coordinates(theta_measurements):
    x = np.cos(2 * np.pi * theta_measurements)
    y = np.sin(2 * np.pi * theta_measurements)
    return x, y

def find_angular_velocity_via_fft(time_stamps, x, y):
    complex_signal = x + 1j * y
    N = len(time_stamps)
    T = (time_stamps[-1] - time_stamps[0]) / N
    
    fft_values = np.fft.fft(complex_signal)
    fft_frequencies = np.fft.fftfreq(N, T)
    
    peak_frequency = np.abs(fft_frequencies[np.argmax(np.abs(fft_values))])
    
    return peak_frequency

def unwrap_phase(theta_measurements):
    return np.unwrap(2 * np.pi * theta_measurements) / (2 * np.pi)

def linear_model(t, omega, theta0):
    return omega * t + theta0

def find_initial_angle(time_stamps, theta_measurements_unwrapped, omega_estimated):
    initial_guess = [omega_estimated, 0.5]

    params, params_covariance = curve_fit(linear_model, time_stamps, theta_measurements_unwrapped, p0=initial_guess)

    # Extract the estimated parameters
    omega_estimated, theta0_estimated = params

    return omega_estimated, theta0_estimated

def filter_junk_measurements(time_stamps, theta_measurements, omega_estimated, noise_std):
    filtered_theta = []
    filtered_time_stamps = []
    num_points = len(theta_measurements)
    
    for i in range(num_points):
        valid_count = 0
        total_count = 0
        for j in range(max(0, i-4), min(num_points, i+5)):
            if i != j:
                expected_value = linear_model(time_stamps[j] - time_stamps[i], omega_estimated, theta_measurements[i])
                diff = np.mod(abs(theta_measurements[j] - expected_value), 1)
                if (diff > 0.5) :
                    diff = 1-diff

                if diff <= 3 * noise_std:
                    valid_count += 1
                total_count += 1
        if valid_count >= total_count / 2:
            filtered_theta.append(theta_measurements[i])
            filtered_time_stamps.append(time_stamps[i])
    
    return np.array(filtered_time_stamps), np.array(filtered_theta)

def estimate_encoder_parameters(time_stamps, theta_measurements, noise_std=0.03):
    # Convert to Cartesian coordinates
    x, y = to_cartesian_coordinates(theta_measurements)
       
    # Find angular velocity using FFT
    omega_estimated = find_angular_velocity_via_fft(time_stamps, x, y)
    print(f"Estimated angular velocity: {omega_estimated}")

    # Filter junk measurements using the new method
    filtered_time_stamps, filtered_theta = filter_junk_measurements(time_stamps, theta_measurements, omega_estimated, noise_std)

    fig1, ax1 = plt.subplots()
    ax1.scatter(time_stamps, theta_measurements)
    ax1.plot(filtered_time_stamps, filtered_theta)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Theta [radians]')
    ax1.legend()

    plt.show()

    # Unwrap the phase data again after filtering
    theta_measurements_unwrapped = unwrap_phase(filtered_theta)
    
    # Find initial angle using curve fitting
    omega_estimated_updated, theta0_estimated = find_initial_angle(filtered_time_stamps, theta_measurements_unwrapped, omega_estimated)
    
    return omega_estimated_updated, theta0_estimated, filtered_time_stamps, filtered_theta

# Example usage
theta0_true = 0.4  # rounds
omega_true = 0.16  # rounds per second

# Generate data
time_stamps, theta_measurements = generate_encoder_data(theta0_true, omega_true, duration=30, noise_std=0.03, junk_prob=0.05)

# Estimate parameters
omega_estimated, theta0_estimated, filtered_time_stamps, filtered_theta = estimate_encoder_parameters(time_stamps, theta_measurements)

print(f"Estimated angular velocity: {omega_estimated}")
print(f"Estimated initial angle: {theta0_estimated}")
print(f"True angular velocity: {omega_true}")
print(f"True initial angle: {theta0_true}")

# Plot the original and filtered data
unwrapped_theta = unwrap_phase(filtered_theta)

fig1, ax1 = plt.subplots()
ax1.plot(time_stamps, theta_measurements, 'o', label='Original Data')
ax1.plot(filtered_time_stamps, unwrapped_theta, 'o', label='Filtered Unwrapped Data')
ax1.plot(filtered_time_stamps, linear_model(filtered_time_stamps, omega_estimated, theta0_estimated), 'r-', label='Fitted Model')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Theta [radians]')
ax1.legend()

plt.show()
