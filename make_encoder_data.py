import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def getEncoderData(theta0Gt, omegaGt):
    # now lets create our ground truth data of the wheel:
    # some parameters:
    startTime = 0
    endTime = 30
    dt = 0.3#random.uniform(0.01, 1)
    print('dt = ' + str(dt))
    numOfDataPoints = (endTime - startTime) / dt
    # gaussian noise for the encoder:
    mean = 0
    std =  0.03
    # junk measurements
    probabilityOfGettingJunkMeasurement = 0.05 


    timeStamps = np.arange(startTime, endTime, dt)
    thetaMeasurement = np.mod(theta0Gt + omegaGt * timeStamps, 1)
    # add gaussian noise:
    gaussianNoise = np.random.normal(mean, std, size=timeStamps.shape)
    thetaMeasurement += gaussianNoise;

    # add uniform white noise to our measurements:
    random_values = np.random.rand(thetaMeasurement.shape[0])

    # Generate random uniform values for replacement
    uniform_replacement = np.random.uniform(0, 1, size=thetaMeasurement.shape[0])

    # Replace elements with uniform random values based on the probability
    thetaMeasurement[random_values < probabilityOfGettingJunkMeasurement] = uniform_replacement[random_values < probabilityOfGettingJunkMeasurement]

    return [timeStamps, thetaMeasurement]

def filter_junk_measurements(timeStamps, thetaMeasurement, threshold=0.25):
    filtered_theta = []
    filtered_timestamps = []
    for i in range(4, len(thetaMeasurement) - 4):
        # Extract the 8 adjacent elements
        adjacent_elements = thetaMeasurement[i-4:i+5]
        # Calculate the median of these elements
        median_adjacent = np.median(adjacent_elements)
        # Check if the absolute difference is within the threshold
        if abs(thetaMeasurement[i] - median_adjacent) <= threshold:
            filtered_theta.append(thetaMeasurement[i])
            filtered_timestamps.append(timeStamps[i])
    return np.array(filtered_timestamps), np.array(filtered_theta)

def model(t, omega, theta0):
    return np.mod(theta0 + omega * t, 1)

theta0Gt = 0.4 # rounds
omegaGt = 0.1 #rounds per second

# generate data:
timeStamps, thetaMeasurement = getEncoderData(theta0Gt, omegaGt)
# thetaMeasurement = np.load('thetaMeasurement.npy')
# timeStamps = np.load('timeStamps.npy')

# filter outliers that james the cyclic nature of the data:
filtered_timeStamps, filtered_theta = filter_junk_measurements(timeStamps, thetaMeasurement)
np.save('thetaMeasurement.npy', thetaMeasurement)
np.save('timeStamps.npy', timeStamps)

thetaMeasurement_unwrapped =  np.unwrap(2 * np.pi * filtered_theta) / (2 * np.pi)

# curve fitting
# Initial guesses
initial_guess = [0.5, 1]

# Perform the curve fitting
params, params_covariance = curve_fit(model, filtered_timeStamps, thetaMeasurement_unwrapped, p0=initial_guess)

# Extract the estimated parameters
omega_estimated, theta0_estimated = params

print("omega_estimated = " + str(omega_estimated))
print("omegaGt = " + str(omegaGt))
print("theta0_estimated = " + str(theta0_estimated))
print("theta0Gt = " + str(theta0Gt))

# Create a figure and axis

fig1, ax1 = plt.subplots()
ax1.plot(timeStamps, thetaMeasurement, marker='o')


fig2, ax2 = plt.subplots()
print
# Plot the data
ax2.plot(filtered_timeStamps, thetaMeasurement_unwrapped, marker='o')



plt.show()