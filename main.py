import csv
import numpy as np
import matplotlib.pyplot as plt

# Calculating coefficients based on the equation given in the instructions
def regression(x, y):
    numerator, denominator = 0, 0

    for i in range(len(x)):
        numerator += (x[i] - np.mean(x)) * (y[i] - np.mean(y))
        denominator += (x[i] - np.mean(x)) ** 2

    b_1 = numerator / denominator
    b_0 = np.mean(y) - (b_1 * np.mean(x))

    return b_0, b_1

# Function to plot scatter plot and regression line
def simlin_plot(x, y, b_0, b_1, color):
    plt.scatter(x, y, color=color)
    y_pred = b_0 + b_1 * x
    plt.plot(x, y_pred, color='black')

# ---------------------------------- Script starts here -----------------------------------------

# Read the first .csv file into a list
with open("Football_players.csv") as f:
    csv_list = list(csv.reader(f))

# Initialize empty arrays to fill in later
age = np.array([])
height = np.array([])
salary = np.array([])

# Filling in the arrays
for row in csv_list:
    if row != csv_list[0]:
        age = np.append(age, int(row[4]))
        height = np.append(height, int(row[5]))
        salary = np.append(salary, int(row[8]))

# Split the columns into 3 different arrays
age_train_1, age_test_1 = age[:80], age[80:]
hgt_train_1, hgt_test_1 = height[:80], height[80:]
sal_train_1, sal_test_1 = salary[:80], salary[80:]

# Calculate coefficients
b_age0, b_age1 = regression(age_train_1, sal_train_1)

# Plot the first graph
plt.subplot(1, 2, 1)
simlin_plot(age_train_1, sal_train_1, b_age0, b_age1, 'blue')
simlin_plot(age_test_1, sal_test_1, b_age0, b_age1, 'red')
plt.title('Salary vs Age')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend(['Regression line', 'Train data', 'Test data'])

# Repeat
age_train_2, age_test_2 = age[:80], age[80:]
hgt_train_2, hgt_test_2 = height[:80], height[80:]
sal_train_2, sal_test_2 = salary[:80], salary[80:]

b_hgt0, b_hgt1 = regression(hgt_train_2, sal_train_2)

# Plot the second graph
plt.subplot(1, 2, 2)
simlin_plot(hgt_train_2, sal_train_2, b_hgt0, b_hgt1, 'blue')
simlin_plot(hgt_test_2, sal_test_2, b_hgt0, b_hgt1, 'red')
plt.title('Salary vs Height')
plt.xlabel('Height')
plt.ylabel('Salary')
plt.legend(['Regression line', 'Train data', 'Test data'])

# Show plots
plt.show()
