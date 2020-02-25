import matplotlib.pyplot as plt
import numpy as np
from multivariate_linear_regression import Multivariate_Linear_Regression_Model
from feature_scaling import min_max_normalization_multivariate, min_max_normalization_univariate

def to_degrees(rad) :
    return rad * 180 / pi

def to_radians(deg) :
    return deg * pi / 180

inputs = []
exp_outputs = []

for x in range(0,200) :
    inputs.append([x**2])
    exp_outputs.append((40000 - x**2)**0.5)

scaled_inputs = min_max_normalization_multivariate(inputs)

reg_model = Multivariate_Linear_Regression_Model(1)

#reg_model.train_with_gradient_descent(min_max_normalization_multivariate(inputs[:100]), exp_outputs[:100], 0.3, 6000)
reg_model.train_with_normal_equation(scaled_inputs, exp_outputs)

x = []
opts = []
error = 0
for a in range(0, 100) :
    x.append(a)
    opts.append(float(reg_model.get_output(scaled_inputs[a])))

    error += abs(exp_outputs[a] - opts[-1])

print(f"Avg error = {error / 100}")

plt.plot(x,exp_outputs[:100],"go",x,opts)
plt.show()