# Progress

## Machine Learning A-Z™: Hands-On Python & R In Data Science

Just using this doc to keep track of working on as I go.

## Day 1 - Set up/Exploration
OOOOOOOOOOOOkay bunch of new stuff here.

Looks like a lot of the code is in ipynb format, for jupyter notebooks. vscode seems to handle it well, though the course recommends using google colab, a web based env. Dunno about that. I would rather use vscode.

So we have ipynb files and py files. The ipynb is IPythonNotebook, and it's what jupyter notebooks are saved as. This is a format where you can have explanatory text, code and visualisations in one doc. It's kinda nift?

Course recommends using google colab, because it doesn't involve installing/configuring anything, but I think that task is worth practicing anyway.

Ok so you can use the ipynb files for this, or the py files. Even in the py files if you want to have things ... 'cellurised' you can do that by using #%% which indicates the start of a new cell which can then be run individually and inspected in the python interactive window.

Feels like a whole new language tbh :/

I've set up a separate folder for my implementations of things (doing), and this is all I'll commit. 

## Day 2 Finishing data preprocessing section
Implemented the preprocessing functions, comments in pre.py are pretty comprehensive about my understanding of things as I was going.

## Day 3 Linear regression
Let's get this bread.

Ok preprocessing is straightforward as there is only one independent variable and no missing data. 

We use the LinearRegression class from sklearn to create a linear model based on the training data. This class uses ordinary least squares to fit the model, which is the process of calculating the sum of the square of the difference between the actual data and model data, then picking the line that has the smallest sum. 

Where yi is the actual value, and y^ is the modeled value, something like: min( sum( (yi - y^)^2 ) )

So with the visualisation we can see the training data and regression model, and it's a good fit. We put the test data on the same model to see if it fits there too and it does. The point of this is that we pulled the test data from the set, trained the model on the rest of the data, then we check to see if the test data fits the same model. It should, because it's from the same data set. I guess if the data were smaller, or more varied this might be tougher.

## Day 4 Multiple linear regression
Ok so we've got a set of data to make a linear model, and it has multiple independent variables. 

Preprocessing is straightforward, no missing data, do need to encode the state information. 

Feature scaling only needs to be done in simple linear regression. With multiple linear regression the 
coefficients scale the variables as required. 

Here's y_pred vs y_test, so the first value is the model predicted profit, the second is the test value.

[[114664.42 105008.31]
 [ 90593.16  96479.51]
 [ 75692.84  78239.91]
 [ 70221.89  81229.06]
 [179790.26 191050.39]
 [171576.92 182901.99]
 [ 49753.59  35673.41]
 [102276.66 101004.64]
 [ 58649.38  49490.75]
 [ 98272.03  97483.56]]

 Ok but I forgot about dummy variable, we encoded state info into 3 columns. But tutorial mentioned that having
 multiple variables stand in for 1 can cause issues. We want to include n - 1 dummy columns, where n is the number
 of dummy columns. This is because if we have 3 states (eg) we could have 

ca, ny, fl
0   1   0
1   0   0
0   0   1

We can have just as much information about the state by using 2 columns:
ca, ny
0   1
1   0
0   0

In row 3 we know the state is fl because it's not the other two. 

So let's see if that gets discussed further?!

Ah ok, the LinearRegression class we use manages this for us. This class also handles identifying significant independent
variables.

## Day 5
Starting with polynomial regression.

Pretty straightfroward, we still use the LinearRegression class for this, we just need to set up a polynomialFeatures to pass to the LinearRegression fit function. 

Now starting support vector regression.

I need to do some reading into why it's all called linear regression, even the polynomial one. They say in the tutorial none of the linear regression models require feature scaling, but the svr one does. 