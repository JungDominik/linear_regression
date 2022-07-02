




# def residue_point(xi, yi, m, t):            #Not used, replaced with squared residue (Root mean squared error)
#     residue = abs(yi - (xi*m + t))
#     return residue


def residue_point_squared(xi, yi, m, t):
    residue = (yi - (xi*m + t)) ** 2
    return residue

def residue_list(x_list, y_list, m, t):
    #Ensure lists are same length
    residue_sum = 0 #Use the point residue function to cumulate up the value into a sum variable
    n = len(x_list)
    for i in range(n):
        residue_sum += residue_point_squared(x_list[i], y_list[i], m, t)
    residue_avg = residue_sum / n
    #print ("residue_avg is " + str(residue_avg))
    return residue_avg

def residue_gradient (x_list, y_list, residue_m1, residue_m2):
    gradient = residue_list(x_list, y_list, residue_m2, 0) - residue_list(x_list, y_list, residue_m1, 0)
    print ("Func residue_gradient: Calculated gradient is " + str(gradient))
    #print ("residue_m1 is " + str(residue_m1))
    #print ("residue_m2 is " + str(residue_m2))
    return (gradient)

def gradient_descent_m(x_list, y_list, iterations):
    previous_m  = 0
    current_m  = 99999999 # Not elegant
    stepsize = 50000000

    
    for it in range(iterations):
        current_m = previous_m - stepsize
        if residue_gradient (x_list, y_list, previous_m, current_m) > 0: 
            stepsize = stepsize * (-0.5)
            previous_m = current_m
        else:
            stepsize = stepsize
            previous_m = current_m
        print("gradient descent iteration m" + str(it) + ": previous_m is " + str(previous_m) +", current_m is " + str(current_m))
    return current_m



#### Test Data

test_X = [11,4,6,3,6,7,4,3,6,7,8,1,3,3,5,4,3,6,235,7,-6]
test_Y = [11,4,4,2,0,1,8,5,5,3,2,1,7,5,8,9,6,7,32,8,-6]
test_null = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
test_one = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
test_linear = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
test_multi = []
for num in test_linear: test_multi.append(num*3)

# import numpy as np
# import matplotlib.pyplot as plt

# arr_x = np.array(test_X)
# arr_y = np.array(test_Y)

# plt.scatter(arr_x, arr_y)
# plt.plot(arr_x + 0.75*arr_x, color = "g")
# #plt.show()

# import pandas as pd
# df_x = pd.DataFrame(test_X)
# df_y = pd.DataFrame(test_Y)


# from sklearn.linear_model import LinearRegression
# linear_regressor = LinearRegression()
# linear_regressor.fit(arr_x,arr_y)