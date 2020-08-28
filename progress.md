# Progress

## Machine Learning A-Z™: Hands-On Python & R In Data Science

Just using this doc to keep track of working on as I go.

# Day 1 - Set up/Exploration
OOOOOOOOOOOOkay bunch of new stuff here.

Looks like a lot of the code is in ipynb format, for jupyter notebooks. vscode seems to handle it well, though the course recommends using google colab, a web based env. Dunno about that. I would rather use vscode.

So we have ipynb files and py files. The ipynb is IPythonNotebook, and it's what jupyter notebooks are saved as. This is a format where you can have explanatory text, code and visualisations in one doc. It's kinda nift?

Course recommends using google colab, because it doesn't involve installing/configuring anything, but I think that task is worth practicing anyway.

Ok so you can use the ipynb files for this, or the py files. Even in the py files if you want to have things ... 'cellurised' you can do that by using #%% which indicates the start of a new cell which can then be run individually and inspected in the python interactive window.

Feels like a whole new language tbh :/

I've set up a separate folder for my implementations of things (doing), and this is all I'll commit. 

# Day 2 Finishing data preprocessing section
Implemented the preprocessing functions, comments in pre.py are pretty comprehensive about my understanding of things as I was going.

# Day 3 Linear regression
Let's get this bread.

Ok preprocessing is straightforward as there is only one independent variable and no missing data. 

We use the LinearRegression class from sklearn to create a linear model based on the training data. This class uses ordinary least squares to fit the model, which is the process of calculating the sum of the square of the difference between the actual data and model data, then picking the line that has the smallest sum. 

Where yi is the actual value, and y^ is the modeled value, something like: min( sum( (yi - y^)^2 ) )

So with the visualisation we can see the training data and regression model, and it's a good fit. We put the test data on the same model to see if it fits there too and it does. The point of this is that we pulled the test data from the set, trained the model on the rest of the data, then we check to see if the test data fits the same model. It should, because it's from the same data set. I guess if the data were smaller, or more varied this might be tougher.

# Day 4 Multiple linear regression
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

# Day 5
Starting with polynomial regression.

Pretty straightfroward, we still use the LinearRegression class for this, we just need to set up a polynomialFeatures to pass to the LinearRegression fit function. 

Now starting support vector regression.

I need to do some reading into why it's all called linear regression, even the polynomial one. They say in the tutorial none of the linear regression models require feature scaling, but the svr one does.

Ok so I ran svr without scaling and it very much does not work. It draws a straight horizontal line. Need to do more reading to figure out exactly why that would be the case. 

The default value for type of kernel used is radial basis function, or rbf. This it what I used initially. 

I also then ran it with the other kernel types.

## Results
### RBF
170370.02

### Linear
216903.83

### Polynomial
197301.33

### Sigmoid
350649.44

### Precomputed:
Got a value error here that I'll just leave alone for now:
"ValueError: Precomputed matrix must be a square matrix. Input is a 10x1 matrix."

Seems over my head.

### Without scaling
130001.82

### And for comparison the poynomial linear regression results
- degree 2: $189495.11 
- degree 3: $133259.47
- degree 4: $158862.45
- degree 5: $174878.08
- degree 6: $174192.81

Also higher order polynomials are generally avoided. In this case the data looks to actually settle on a neat polynomial curve. But less convenient data would work better with lower order polynomials. Otherwise the line will wobble about fitting the data perfectly without showing the trend we're looking for. 

# Day 5 still or maybe 6?
This link is wonderfully explaining nearly all my gaps above. 

https://towardsdatascience.com/polynomial-regression-bbe8b9d97491

# Day 6
Starting decision tree regression.

No need for feature scaling with DTR. Honestly a bit confused about when it's necessary and not.

This was pretty straightforward, done before watching the tutorial. Visualisation was interesting, the high res version looks stepwise, due to the way dtr works. 

dtr prediction: $150000.00

# Day 7 
Worked through random forest regression.

This creates many trees based on a random sub set of the data, then averages out the results to get the prediction.

When creating the regression you can choose the number of trees to run, this was done with 10, and we got $167000, which is more reasonable than the single decision tree regression.

Now on to evaluating model performance.

Rsquared = 1 - SSres / SStot

Where SSres is the sum the squared differences between the data and the model. 
And SStot is the sum of the squared differences between the data and the average.

So the smaller SSres, and the larger SStot, the closer to 1 R^2 becomes. A perfect model would have an R^2 of 1.

Adjusted R^2 has additional parameters which take number of samples and number of independent variables into account. The more variables the lower R^2, so this works to penalise having loads of variables which don't contribute to the model.

Now time to use R^2 to compare model performance on a bigger dataset.

So we've got a larger dataset and we'll run all the models I know now then compare their results. 

Might be neat to have a set up where all models are called from a main script that then compares results.

Linear = 0.9321860060402446
Polynomial = 0.9435538032031084
Support Vector = 0.9431332255808436
Decision Tree = 0.9342783714449767
Random Forest = 0.9628673278135129

I've set up a mainpy that imports and calls these models as functions, then computes the best model based on their R^2.

I just need to double check that all I've got sort of lines up with the tutorial, and maybe go over support vector regression again, still not sure on the scaling part. 

# Day 8
Moving on to part 3, classification now. So the gist is we've been working with a dependent variable that's a number so far, what if it's a category?

Probably just going to get through the first lecture today. 

# Day 9
Logistic Regression, let's get it.

Ok I implemented this, but feels pretty shallow at the moment, found a solid article that has helped:
https://machinelearningmastery.com/logistic-regression-for-machine-learning/

Here's the logistic function, it maps any value to a value between 0 and 1 (exclusive):
    1 / (1 + e^-value)

    Where e is the base of the natural logarithms (Euler’s number or the EXP() function in your spreadsheet) and value is the actual numerical value that you want to transform. Below is a plot of the numbers between -5 and 5 transformed into the range 0 and 1 using the logistic function.

Here is the regression equation:

    y = e^(b0 + b1*x) / (1 + e^(b0 + b1*x))

While this looks wobbly, it still results in linear predictions. In the visualisations you can see that the model draws a straight line through the features and assigns 0/1 based off  that. 

The visualisation is created by drawing pixels in the colour predicted by the model, red for nobuy/0 and green for buy/1, then adds dots for the actual data. So you can see a few green data points on the red side, and vv. 

Logistic regression had a confusion matrix of
[[65  3] - 65 correct nobuy predictions, 3 incorrect
[ 8 24]] - 8 incorrect buy predictions, 24 correct

You can see all of these incorrect/correct points on the test set visualisation.

And an accuracy of 0.89
This is the number of correct predictions divided by total number of predictions, so 89% correct.

# Day 10
K nearest neighbor classification
Step 1 - choose a number for k (number of neighbours, often 5)
Step 2 - take the k nearest neighbours of the new data point according to euclidean distance
Step 3 - among those k neighbours, how many fall into each category
Step 4 - assign the new data point to the category where we counted the most neighbours

Same visualisation used for knn, takes a while to run. 

knn had a confusion matrix of:
[[64  4] - 64 correct nobuy predictions, 4 incorrect
[ 3 29]] - 3 incorrect buy predictions, 29 correct

And an accuracy of 0.93, so 4% better than logistic regression.

Support vector machine works similarly to SVR in regression, in that we're looking at the data points closer together, and finding a classification line with the maximum margin between that line and the closest vectors. 

This line is called the max margin hyperplane, while we're visualising it here as a line, with more features this line becomes a hyperplane of many dimensions. 

I tried out SVM before watching the lecture, and got these results:
[[66  2] - 66 correct nobuy predictions, 2 incorrect
[ 8 24]] - 8 incorrect buy predictions, 24 correct

Accuracy 0.9 - number of correct predictions divided by total number of predictions, so 90% correct.

# Day 11
Kernel SVM
So here we're looking at how we can use svm to classify non-linearly separable data (this is data which cannot be separated by a single line).

We first looked at mapping data to a higher plane, where it would be possible to use a straight line to separate data. So if we had one dimensional data, we could project that into 2 dimensions with some arbitrary function, then draw a straight line which could separate that data. 

This is computationally expensive, and there are trickier ways to do it which don't involve completing the mapping/unmapping. 

Many types of kernel functions some:
gaussian RBF Kernel (roughly bell shaped)
Sigmoid kernel 
Polynomial kernel

This site has a reasonable one-two of the kernel trick:
https://github.com/trthatcher/MLKernels.jl

This is also a good read:
https://towardsdatascience.com/the-kernel-trick-c98cdbcaeb3f

More :
https://towardsdatascience.com/kernel-function-6f1d2be6091

Just ran this, I think,

Using the rbf kernel, svm had a confusion matrix of:
[[64  4] - 64 correct nobuy predictions, 4 incorrect
[ 3 29]] - 3 incorrect buy predictions, 29 correct

And an accuracy of 0.93, same as knn. 

Let's try different kernels.

Sigmoid:
[[54 14]
 [12 20]]
0.74
Ooooff

Polynomial:
[[67  1]
 [13 19]]
0.86

Precomputed:
"If a callable is given it is  
    used to pre-compute the kernel matrix from data matrices; that matrix  
    should be an array of shape `(n_samples, n_samples)`.  "

Interestingly, increasing the training size (by setting test set to .2 instead of .25), get's us better results. This with rbf:
[[55  3]
 [ 1 21]]
0.95

# Day 12
Bayes Theorem

P(A|B) = P(B|A) * P(A)
            P(B)

The probability of A given some condition B is: the probability of that condition B, given A multiplied by the probability of A, all divided the probability of B.

The example in the tutorial is you have two machines creating spanners. Machine 1 makes 30 per hour, machine 2 makes 20 per hour. Out of all the spanners produced, 1% are defective. And out of all the spanners produced, 50% of the defective spanners are from Machine 1/2. 

What is the probability of any given spanner picked from Machine 2 being defective? 

P(Defective | made by machine 2) = P(made by machine 2 | defective) * P(defective)
                                            P(made by machine 2)

P(made by machine 2 | defective) = 50% of defective spanners, are from machine 2
P(defective) = 1% of spanners are defective
P(made by machine 2) = 40% of spanners are made by machine 2

P(Defective | made by machine 2) = 0.5 * 0.01
                                      0.4

P(Defective | made by machine 2) = 0.00125 or 1.25%

Now just to make sure I've got this, the probability of any given spanner picked from Machine 1 being defective is:

P(Defective | made by machine 1) = P(made by machine 1 | defective) * P(defective)
                                            P(made by machine 1)

P(Defective | made by machine 1) = 0.5 * 0.01
                                      0.6

P(Defective | made by machine 1) = 0.00833 or 0.83%

Ok we used Gaussian Naive Bayes:
y_pred_single:[0]
[[65  3] 65 correct, 3 incorrect
 [ 7 25]] 7 incorrect, 25 correct
0.9

Not the best so far.

# Day 13
On to decision tree and random forest classification. 
Decision tree:
y_pred_single:[0]
[[62  6]
 [ 3 29]]
0.91

Random forest:
y_pred_single:[0]
[[63  5]
 [ 4 28]]
0.91

# Day 14 
Putting them all together.