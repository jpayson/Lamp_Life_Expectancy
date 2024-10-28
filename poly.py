import numpy as np
import matplotlib.pyplot as plt
import csv

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import axes3d

'''
 As an example of polynomial and multivariable regression, this
 program seeks to relate the ratio of nominal (advertised) to observed
 lifetime for fluorescent lamps (z = LT0_over_LT) to the integrated
 glow current (x = I_Glow, measured by lamp gas discharge) as well
 as the ratio of observed to nominal voltage (y = V_Lamp / V_Lampnom).

 Data is sourced from:
 F.G. Rosillo and N.M. Chivelet (2009). "Lifetime Prediction of
 Fluorescent Lamps Used in Photovoltaic Systems," Lighting Research
 and Technology, Vol. 41, #2, pp.183-197.

 Note some lamp models had multiple samples, these are reflected by
 duplicated rows in the .csv

 Based on the results of the source article as well as experimentation,
 the x^2 and xy terms are omitted from the model to get a more consistent
 fit. This results in the general equation: (z = b0 + b1*x + b2*y + b3*(y^2))

 Due to the sample size and different lamp model sample numbers, re-running
 the program may provide better predictions (by generating different random
 samples for training/testing each time.)

'''

LAMP_DATA_FORMAT = np.dtype([
    ('Model', '<U8'),
    ('LT0_over_LT', float),
    ('I_Glow', float),
    ('V_Lamp', float),
    ('V_Lampnom', float),
])

# Reads in data from given file and returns it as a record array of dtype=LAMP_DATA_FORMAT
def lamp_data_from_csv(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        next(reader) # Discard headers
        raw_data = np.asarray(list(reader))

        # Strip out individual columns
        model = np.asarray(raw_data[:, 0], dtype='<U8')[:, np.newaxis]
        LT0_over_LT = np.asarray(raw_data[:, 1], dtype=float)[:, np.newaxis]
        I_Glow = np.asarray(raw_data[:, 2], dtype=float)[:, np.newaxis]
        V_Lamp = np.asarray(raw_data[:, 3], dtype=float)[:, np.newaxis]
        V_Lampnom = np.asarray(raw_data[:, 4], dtype=float)[:, np.newaxis]

        # Combine individual arrays into one record array in LAMP_DATA_FORMAT
        return np.rec.fromarrays([model, LT0_over_LT, I_Glow, V_Lamp, V_Lampnom],
                                 dtype=LAMP_DATA_FORMAT)

# Takes an xy data set that has been transformed to quadratic features
# Returns the data set with the x^2 and xy terms removed
def omit_features(xy_quadratic):
    return np.delete(np.delete(xy_quadratic, 3, axis=1), 3, axis=1)

# Iterates over a plane and predicts z for each x, y
def predict_plane(X, Y, regressor, featurizer):
    Z = np.empty((X.shape[0], Y.shape[1]))

    for c in range(X.shape[1]):
        xy_to_plot = np.hstack((X[:, np.newaxis, c], Y[:, np.newaxis, c]))
        
        xy_to_plot_quadratic = featurizer.transform(xy_to_plot)
        
        xy_to_plot_quadratic = omit_features(xy_to_plot_quadratic)
        
        Z[:, np.newaxis, c] = regressor.predict(xy_to_plot_quadratic)

    return Z

# Prints the equation, MSE, and r-square of given test set and regressor
def print_results(xy_test_poly, z_test, regressor):
    mse = np.mean((regressor.predict(xy_test_poly) - z_test) **2)
    score = regressor.score(xy_test_poly, z_test)

    equation_string = "z = {:.3f} + {:.3f}x + {:.3f}y + {:.3f}(y^2)"

    print(equation_string.format(regressor.intercept_[0],
                                 regressor.coef_[0][1],
                                 regressor.coef_[0][2],
                                 regressor.coef_[0][3]))
    print("MSE = {:.4f}, Score = {:.4f}".format(mse, score))

# Initialize data set
data = lamp_data_from_csv('fluorescent_lamp.csv')

# Randomly assign indexes for testing and training sets
num_samples = data.shape[0]
indexes = np.arange(num_samples)

test_indexes = np.sort(np.random.choice(indexes, int(num_samples * 0.2), replace=False))
train_indexes = np.sort(np.setdiff1d(indexes, test_indexes))

# Extract data and divide into testing and training sets
x = data['I_Glow']
y = data['V_Lamp'] / data['V_Lampnom']
z = data['LT0_over_LT'].reshape(-1,1)

x_train, y_train, z_train = x[train_indexes], y[train_indexes], z[train_indexes]
x_test, y_test, z_test = x[test_indexes], y[test_indexes], z[test_indexes]

# Combine x and y sets into a single xy set
xy_train = np.hstack((x_train, y_train))
xy_test = np.hstack((x_test, y_test))

# Apply quadratic features, omit selected features
quadratic_featurizer = PolynomialFeatures(degree=2)
xy_train_quadratic = quadratic_featurizer.fit_transform(xy_train)
xy_test_quadratic = quadratic_featurizer.transform(xy_test)

xy_train_quadratic = omit_features(xy_train_quadratic)
xy_test_quadratic = omit_features(xy_test_quadratic)

# Create an xy plane to be used for plotting
x_to_plot = np.linspace(min(x), max(x), 100)
y_to_plot = np.linspace(min(y), max(y), 100)
X, Y = np.meshgrid(x_to_plot, y_to_plot)

# Train regressor and predict for the plotting plane
regressor = LinearRegression()
regressor.fit(xy_train_quadratic, z_train)

Z = predict_plane(X, Y, regressor, quadratic_featurizer)

# Print and plot results
print_results(xy_test_quadratic, z_test, regressor)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel("Current (Amps)")
ax.set_ylabel("Ratio of Actual:Advertised Voltage")
ax.set_zlabel("Ratio of Actual:Advertised Lifetime")

ax.scatter(x, y, z, c='r', marker='o', label="True Values")
ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5, label="Prediction Function")
plt.legend()
plt.show()

                        
