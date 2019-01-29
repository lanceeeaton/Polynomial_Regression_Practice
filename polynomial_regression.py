#%% [markdown]
# # The Goal:
# ## Create a model that given a level in our business can give the correct salary.
# This exericse serves as a way to learn the basics of polynomial regression.
#%% [markdown]
# ## Imports.
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error

#%% [markdown]
# ## Importing the dataset.
#%%
dataset = pd.read_csv('Position_Salaries.csv')

#%% [markdown]
# ## Let's see how Position Level relates to Salary.
#%%
fig = plt.figure()
ax = fig.add_subplot(111)

salaries = ax.scatter(dataset['Level'],dataset['Salary'],color='red')
ax.set_facecolor('white')
ax.set_title('Salary by Position level',color='black')
ax.set_xlabel('Position level',color='black')
ax.set_ylabel('Salary',color='black')

ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')

ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')
plt.legend((salaries,),('Salaries',),facecolor='grey',loc='upper left', fontsize=12)
plt.show()

#%% [markdown]
# We now can see that polynomial regression should be the right choice for a regression model.
#%% [markdown]
# ## Spliting data into independent variables (X) and dependent variables (y).
#%%
X = pd.DataFrame(dataset['Level'], columns=['Level'])
y = pd.DataFrame(dataset['Salary'], columns=['Salary'])
#%% [markdown]
# We don't use the Position feature as an independent variable because it is simply a label for the Level feature.
#%% [markdown]
# ## Take a look at X.
#%%
X

#%% [markdown]
# ## Take a look at y.
#%%
y

#%% [markdown]
# ## Note:
# #### We do not split the data into training and testing data.
# We don't have enough data to split it and still make accurate predictions. 
# This does raise a concern about overfitting.

#%% [markdown]
# ## Here we fit a basic linear regression model to the dataset.
#%%
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)

#%% [markdown]
# ## Let's see how our models predictions compare to the actual salaries.
#%%
predictions = linear_regressor.predict(X)

fig = plt.figure()
ax = fig.add_subplot(111)

actual = ax.scatter(X, y, color = 'red')
pred = ax.plot(X, predictions, color = 'blue')

ax.set_title('Linear Regression Predictions vs Actual',color='black')
ax.set_facecolor('white')
ax.set_xlabel('Position level',color='black')
ax.set_ylabel('Salary',color='black')

ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')

ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')

ax.legend((actual,pred[0]),('Actual','Predictions'),facecolor='grey',loc='upper left', fontsize=12)
plt.show()

#%% [markdown]
# We can see that our basic linear regression model does not do a very good job of predicting salaries.

#%% [markdown]
# ## Let's just check the mean absolute error so we can see just how off we are.
#%%
mae = mean_absolute_error(predictions,y)
mae

#%% [markdown]
# We are off by over 120k on average, this is certainly not an acceptable model for predicting salaries.

#%% [markdown]
# ## Here we are adding polynomial features to our X matrix and defining it as X_poly.
#%%
polynomial_feature = PolynomialFeatures(degree = 4, include_bias=False)
X_poly = polynomial_feature.fit_transform(X)
#%% [markdown]
# The degree specified is what power to raise X to. I have decided on 4 after testing degrees both above and below this value.
# We set the include_bias parameter to False as the libraries we are working with do not require us to have a bias in our matrix.
# The libaries we are working with add in the bias for the linear regression equation silently.

#%% [markdown]
# ## Let's take a look at our new independent matrix: X_poly.

#%%
pd.DataFrame(X_poly,columns=['Level','Level^2','Level^3','Level^4'])

#%% [markdown]
# We can see there is are 3 new features in our X matrix, one for each degree after 1 up to the degree specified (4).
#%% [markdown]
# ## Here we fit a linear regression model to the dataset based on our new features.
# This is what polynomial regression is, a linear regression model based on polynomial features we create.
#%%
linear_regressor_poly = LinearRegression()
linear_regressor_poly.fit(X_poly, y)

#%% [markdown]
# ## Let's see if this model does any better than our first.
#%%
predictions = linear_regressor_poly.predict(X_poly)

fig = plt.figure()
ax = fig.add_subplot(111)

actual = ax.scatter(X, y, color = 'red')
pred = ax.plot(X, predictions, color = 'blue')

ax.set_title('Polynomial Regression Predictions vs Actual',color='black')
ax.set_facecolor('white')

ax.set_xlabel('Position level',color='black')
ax.set_ylabel('Salary',color='black')

ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')

ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')

ax.legend((actual,pred[0]),('Actual','Predictions'),facecolor='grey',loc='upper left', fontsize=12)
plt.show()
#%% [markdown]
# We can see that our polynomial regression model does a much better job.

#%% [markdown]
# ## As before let's see mean absolute error.
#%%
mae = mean_absolute_error(predictions,y)
mae

#%% [markdown]
# We can see that we are only off by around 12k this time.
# Using polynomial regression certainly made a drastic difference.
# In the future we could make an even better model, using various methods that I am still learning. 
#%% [markdown]
# ### Note
# The Position_Salaries.csv was taken from the Machine Learning A-Z™: Hands-On Python & R In Data Science course offered on Udemy.
# This served as an exercise for me to learn what polynomial regression is and how to implement it.
