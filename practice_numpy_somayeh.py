# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 10:40:52 2025

@author: Somayeh doosti
"""

# Create a 1D NumPy array and print its shape, ndim, dtype, and size

import numpy as np
arr=np.array([10, 20, 30, 40, 50])
print("shape is:",arr.shape)
print("ndim:",arr.ndim)
print("data type:",arr.dtype)
print("size of array:",arr.size)

# Create arrays of zeros, ones, and a range with a step using NumPy

import numpy as np
data=np.zeros(5)
data1=np.ones(4)
data2=np.arange(5,30,3)
print(f""""array with zeros is {data}
      array with ones is {data1}
      array between 5 and 30 whit 3 step is{data2}""")
      
# Reshape a 1D NumPy array into a 2D array

import numpy as np 
arr=np.arange(2,48,4)
new_shape=arr.reshape(3,4)
print("new shape is :",new_shape)

# Slice specific rows and columns from a 2D NumPy array

import numpy as np 
arr=np.arange(10).reshape(2,5)

first_row=arr[0,]
third_column=arr[0: , 3]
middle_value = arr[1, 2]

print("first row is:",first_row)
print("third column is:",third_column)
print("middle value is:",middle_value)

# Filter elements from an array using a boolean condition

import numpy as np 
arry=np.array([3, 7, 2, 9, 12, 5, 6])
mask=arry>5
print("value of arry that are >5 is ",arry[mask])


# Calculate mean, median, standard deviation, and sum of a NumPy array

import numpy as np 
arr=np.arange(55,105,5)
mean_of_arr=arr.mean()
median_of_arr=np.median(arr)
std_of_arr=arr.std()
sumation_of_arr=arr.sum()
print("mean of values is " ,mean_of_arr,"\n median of values is",median_of_arr,
      "\n std of values is ",std_of_arr,
      "\n sumation of value is " , sumation_of_arr)

# Use broadcasting to add a 1D array to each row of a 2D array

import numpy as np 
arr1=np.array([[1 ,2, 3],
               [4 ,5 ,6],
               [7 ,8 ,9]])
arr2=np.array([10 ,20 ,30])
sum1=arr2+arr1
print(sum1)

# Perform matrix multiplication and transpose using NumPy

import numpy as np 
A=np.array([[6,8],
            [4,3],
            [1,9]])

B=np.array([[1,2,3],
            [7,8,5]])
zarb=np.dot(A,B)
taranahade=zarb.T
print("zarb A ,B is:\n",zarb)
print("transpose zarb bein A , B is :\n",taranahade)

# Filter elements based on multiple conditions using NumPy

import numpy as np 
arr=np.arange(1,20)
mask=arr>5
even=arr%2==0
both=mask & even 
print("Numbers greater than 5 and even:\n", arr[both])

# #matrix_creation #random_numbers #numpy #3x3_matrix

import numpy as np 
arr=np.random.randint(17,75,(3,3))
print(arr)

#greater_than_mean#numpy_masking#array_filtering

import numpy as np 
arr=np.random.randint(1,100,(1,20))
mean1=np.mean(arr)
mask=arr > mean1
print("values are:\n",arr)
print("values mean is :\n",mean1)
print("values that  are greater than mean: \n ",arr[mask])

# Convert vector to matrix and matrix to vector

import numpy as np 
arr=np.random.randint(1,20,(1,8))
matrix=arr.reshape(2,4)
print("vactor is:\n",arr)
print("matrix of this vactor is:\n",matrix)

mat=np.array([[1,2,3],
              [4,5,6],
              [7,8,9]])
arr1=mat.flatten()
print("the matrix is :\n",mat)
print("the vactor of this matrix is:\n",arr1)

# salary_stats

import numpy as np 

salaries = np.array([
    8500000, 9200000, 7800000, 10000000, 12000000,
    9500000, 8700000, 11000000, 10500000, 11500000,
    8800000, 9900000, 10800000, 9300000, 8900000,
    10200000, 7600000, 9400000, 9800000, 9700000
])
mean_val=np.mean(salaries)
std_val=np.std(salaries)
min_val=np.min(salaries)
max_val=np.max(salaries)

mask=salaries > mean_val
above_mean= salaries[mask]
percent_above_mean = (above_mean.size / salaries.size) * 100
print(f"Percentage of people with salary above mean: {percent_above_mean:.2f}%")

print(f"""mean of people salaries is {mean_val} and minimum salaries is {min_val}
      and maximum salaries is {max_val}and std is {std_val}""")

 
# Solving a system of two linear equations with NumPy


import numpy as np 
A=np.array([[3,4],[8,2]])
b=np.array([11,8])

solving=np.linalg.solve(A,b)
print("x:",solving[0])
print("y:",solving[1])

# Removing NaN values from a NumPy array


import numpy as np 

data=np.array([1,2,6 ,75 , np.nan , 78, np.nan])

data_clean=(~np.isnan(data))
print(data[data_clean])

# #TemperatureStatistics

import numpy as np
data=np.loadtxt(r"C:\Users\Rayanegostar\Desktop\somayeh.txt")
mean_temperature=np.mean(data[:,2])
max_temperature=np.max(data[:,2])
min_temperature=np.min(data[:,2])
print("mean temperature is:",mean_temperature)
print("max temperature is:",max_temperature)
print("min temperature is:",min_temperature) 


# #MinMaxNormalization


import numpy as np
data=np.loadtxt(r"C:\Users\Rayanegostar\Desktop\somayeh.txt")
temp=data[:,2]
max_temperature=np.max(data[:,2])
min_temperature=np.min(data[:,2])
normal=(temp-min_temperature)/(max_temperature-min_temperature)
print("normalaize temperature is:",normal)

# Statistical analysis and histogram of student grades

import numpy as np
import matplotlib.pyplot as plt
grade=np.array([15, 18, 14, 20, 17, 19, 13, 16, 15, 18])
mean_grade=np.mean(grade)
std_grade=np.std(grade)
print("mean of grades is :",mean_grade, "std of grade is:",std_grade)

plt.hist(grade,bins=len(np.unique(grade)),edgecolor="blue")
plt.xlabel("nomarat")
plt.ylabel("tedad")
plt.show()

# Dice roll simulation and probability estimation

import numpy as np
halat=np.random.randint(1,7,size=1000)
counts = np.bincount(halat)[1:]
print(counts)
ehtemal= counts[0:]/1000
print(ehtemal)

# Histogram of 10x10 random matrix values

import numpy as np
import matplotlib.pyplot as plt
matrix=np.random.rand(10,10)
data = matrix.flatten()

print(matrix)

plt.hist(matrix,bins=len(data),edgecolor="red")
plt.xlabel("adad")
plt.ylabel("tedad")
plt.show()