import random
import numpy as np
import matplotlib.pyplot as plt

def getEncoderData(theta0Gt, omegaGt):
    # now lets create our ground truth data of the wheel:
    # some parameters:
    startTime = 0
    endTime = 30
    dt = random.uniform(0.01, 1)
    numOfDataPoints = (endTime - startTime) / dt
    # gaussian noise for the encoder:
    mean = 0
    std = 0.03
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


theta0Gt = 0.4 # rounds
omegaGt = 0.1 #rounds per second
timeStamps, thetaMeasurement = getEncoderData(theta0Gt, omegaGt)
# Create a figure and axis
fig, ax = plt.subplots()

# Plot the data
ax.plot(timeStamps, thetaMeasurement, marker='o')
plt.show()