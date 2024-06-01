import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Ground truth
start_time = 0
end_time = 30
dt = 0.3  #random.uniform(0.01, 1)
theta0_true = 0.4  # rounds
omega_true = 0.1  # rounds per second

# Gaussian noise
mean = 0
std_dev = 0.03

# Uniform random noise:
junk_measurement_prob = 0.05


def generate_encoder_data(theta0_true, omega_true):
    print('dt = ' + str(dt))
    num_data_points = (end_time - start_time) / dt

    time_stamps = np.arange(start_time, end_time, dt)
    theta_measurements = np.mod(theta0_true + omega_true * time_stamps, 1)
    gaussian_noise = np.random.normal(mean, std_dev, size=time_stamps.shape)
    theta_measurements += gaussian_noise
    fig1, ax1 = plt.subplots()

    random_values = np.random.rand(theta_measurements.shape[0])
    uniform_replacement = np.random.uniform(0, 1, size=theta_measurements.shape[0])
    theta_measurements[random_values < junk_measurement_prob] = uniform_replacement[random_values < junk_measurement_prob]
    ax1.plot(time_stamps, theta_measurements, 'o', label='Original Data')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Theta [radians]')
    ax1.legend()



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

def unwrap_phase(theta_measurements):
    return np.unwrap(2 * np.pi * theta_measurements) / (2 * np.pi)

def linear_model(t, omega, theta0):
    return omega * t + theta0

def estimate_omega_and_theta0(time_stamps, theta_measurements):

    # Filter outliers that disrupt the cyclic nature of the data
    filtered_time_stamps, filtered_theta = filter_junk_measurements(time_stamps, theta_measurements)

    # Unwrap the phase data
    unwrapped_theta = unwrap_phase(filtered_theta)

    # Initial guesses for curve fitting
    initial_guess = [omega_true, theta0_true]

    # Perform the curve fitting for linear model
    params, params_covariance = curve_fit(linear_model, filtered_time_stamps, unwrapped_theta, p0=initial_guess)

    # Extract the estimated parameters
    omega_estimated, theta0_estimated = params

    print("omega_estimated = " + str(omega_estimated))
    print("omega_true = " + str(omega_true))
    print("theta0_estimated = " + str(theta0_estimated))
    print("theta0_true = " + str(theta0_true))

    # Plot the original and filtered data
    fig1, ax1 = plt.subplots()
    ax1.plot(time_stamps, theta_measurements, 'o', label='Original Data')
    ax1.plot(filtered_time_stamps, unwrapped_theta, 'o', label='Filtered Unwrapped Data')
    ax1.plot(filtered_time_stamps, linear_model(filtered_time_stamps, omega_estimated, theta0_estimated), 'r-', label='Fitted Model')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Theta [radians]')
    ax1.legend()

    plt.show()


# Generate data
time_stamps, theta_measurements = generate_encoder_data(theta0_true, omega_true)

# Estimate and plot the parameters
estimate_omega_and_theta0(time_stamps, theta_measurements)
