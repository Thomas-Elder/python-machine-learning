
from models import linearRegression
from models import polynomialRegression
from models import supportVectorRegression
from models import decisionTreeRegression
from models import randomForestRegression

models = []

models.append({'model': 'linearRegression', 'R^2': linearRegression.modelSelection_LinearRegression('Data.csv')})
models.append({'model': 'polynomialRegression', 'R^2': polynomialRegression.modelSelection_PolynomialRegression('Data.csv')})
models.append({'model': 'supportVectorRegression', 'R^2': supportVectorRegression.modelSelection_SupportVectorRegression('Data.csv')})
models.append({'model': 'decisionTreeRegression', 'R^2': decisionTreeRegression.modelSelection_DecisionTreeRegression('Data.csv')})
models.append({'model': 'randomForestRegression', 'R^2': randomForestRegression.modelSelection_RandomForestRegression('Data.csv')})

best = {'model': '', 'R^2': 0}

print()
print('Here are the results for the 5 models tested:')

for model in models:
    print('{:<30}:{}'.format(model['model'], model['R^2']))
    
    if model['R^2'] > best['R^2']:
        best = model

print()
print('The best performing model was:{} with an R^2 of {}'.format(best['model'], best['R^2']))