import matplotlib.pyplot as plt

# Example data
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

# Create a figure and axis
fig, ax = plt.subplots()

# Plot the data
ax.plot(x, y, marker='o')

# Add a title and labels
ax.set_title('Simple Line Plot')
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')

# Show the plot
plt.show()