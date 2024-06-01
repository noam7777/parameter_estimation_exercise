import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def getEncoderData(theta0Gt, omegaGt):
    startTime = 0
    endTime = 30
    dt = 0.3  # fixed time step for reproducibility
    print('dt = ' + str(dt))
    numOfDataPoints = (endTime - startTime) / dt
    mean = 0
    std = 0.03
    probabilityOfGettingJunkMeasurement = 0.05

    timeStamps = np.arange(startTime, endTime, dt)
    thetaMeasurement = np.mod(theta0Gt + omegaGt * timeStamps, 1)
    gaussianNoise = np.random.normal(mean, std, size=timeStamps.shape)
    thetaMeasurement += gaussianNoise

    random_values = np.random.rand(thetaMeasurement.shape[0])
    uniform_replacement = np.random.uniform(0, 1, size=thetaMeasurement.shape[0])
    thetaMeasurement[random_values < probabilityOfGettingJunkMeasurement] = uniform_replacement[random_values < probabilityOfGettingJunkMeasurement]

    return timeStamps, thetaMeasurement

def filter_junk_measurements(timeStamps, thetaMeasurement, threshold=0.25):
    filtered_theta = []
    filtered_timestamps = []
    for i in range(4, len(thetaMeasurement) - 4):
        adjacent_elements = thetaMeasurement[i-4:i+5]
        median_adjacent = np.median(adjacent_elements)
        if abs(thetaMeasurement[i] - median_adjacent) <= threshold:
            filtered_theta.append(thetaMeasurement[i])
            filtered_timestamps.append(timeStamps[i])
    return np.array(filtered_timestamps), np.array(filtered_theta)

def unwrap_phase(thetaMeasurement):
    return np.unwrap(2 * np.pi * thetaMeasurement) / (2 * np.pi)

def linear_model(t, omega, theta0):
    return omega * t + theta0

theta0Gt = 0.4  # rounds
omegaGt = 0.1  # rounds per second

# Generate data
timeStamps, thetaMeasurement = getEncoderData(theta0Gt, omegaGt)

# Filter outliers that disrupt the cyclic nature of the data
filtered_timeStamps, filtered_theta = filter_junk_measurements(timeStamps, thetaMeasurement)

# Unwrap the phase data
thetaMeasurement_unwrapped = unwrap_phase(filtered_theta)

# Initial guesses for curve fitting
initial_guess = [omegaGt, theta0Gt]

# Perform the curve fitting for linear model
params, params_covariance = curve_fit(linear_model, filtered_timeStamps, thetaMeasurement_unwrapped, p0=initial_guess)

# Extract the estimated parameters
omega_estimated, theta0_estimated = params

print("omega_estimated = " + str(omega_estimated))
print("omegaGt = " + str(omegaGt))
print("theta0_estimated = " + str(theta0_estimated))
print("theta0Gt = " + str(theta0Gt))

# Plot the original and filtered data
fig1, ax1 = plt.subplots()
ax1.plot(timeStamps, thetaMeasurement, 'o', label='Original Data')
ax1.plot(filtered_timeStamps, thetaMeasurement_unwrapped, 'o', label='Filtered Unwrapped Data')
ax1.plot(filtered_timeStamps, linear_model(filtered_timeStamps, omega_estimated, theta0_estimated), 'r-', label='Fitted Model')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Theta [radians]')
ax1.legend()

plt.show()
