
from models import linearRegression
from models import polynomialRegression
from models import supportVectorRegression
from models import decisionTreeRegression
from models import randomForestRegression

linearRegression_R2 = linearRegression.modelSelection_LinearRegression('Data.csv')
polynomialRegression_R2 = polynomialRegression.modelSelection_PolynomialRegression('Data.csv')
supportVectorRegression_R2 = supportVectorRegression.modelSelection_SupportVectorRegression('Data.csv')
decisionTreeRegression_R2 = decisionTreeRegression.modelSelection_DecisionTreeRegression('Data.csv')
randomForestRegression_R2 = randomForestRegression.modelSelection_RandomForestRegression('Data.csv')

print()
print('{:<30}:{}'.format('linearRegression_R2', linearRegression_R2))
print('{:<30}:{}'.format('polynomialRegression_R2', polynomialRegression_R2))
print('{:<30}:{}'.format('supportVectorRegression_R2', supportVectorRegression_R2))
print('{:<30}:{}'.format('decisionTreeRegression_R2', decisionTreeRegression_R2))
print('{:<30}:{}'.format('randomForestRegression_R2', randomForestRegression_R2))
