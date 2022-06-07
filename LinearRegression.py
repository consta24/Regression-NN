import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


names = ['VENDOR','MODEL_NAME','MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP'];
data = pd.read_csv('machine.data', names=names)


x = data.iloc[:,2:]
y = data.iloc[:,-1]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.1, random_state = 0, shuffle = True)

scaler = StandardScaler()
scaler.fit(xTrain, yTrain)

xTrain = scaler.transform(xTrain)
xTest = scaler.transform(xTest)

yTrain = yTrain.values.reshape(-1, 1)
yTest  = yTest.values.reshape(-1, 1)

yScale = StandardScaler()
yScale.fit(yTrain)

yTrain = yScale.transform(yTrain)
yTest = yScale.transform(yTest)

model = linear_model.LinearRegression()
model.fit(xTrain,yTrain)
linearRegressionScore = model.score(xTrain, yTrain)

yPredict = model.predict(xTest)


print("---> Linear Regression <---")

print("Coefficient of determination R^2 of the prediction.: ", linearRegressionScore)

print("Mean squared error: %.2f" % mean_squared_error(yTest, yPredict))

print('Test Variance score: %.2f' % r2_score(yTest, yPredict))



def plot1():

    fig, ax = plt.subplots()
    
    ax.scatter(yTest, yPredict, edgecolors=(0, 0, 0))
    
    ax.plot([yTest.min(), yTest.max()], [yTest.min(), yTest.max()], 'k--', lw=4)
    
    ax.set_xlabel('Actual')
    
    ax.set_ylabel('Predicted')
    
    ax.set_title("Ground Truth vs Predicted")
    
    plt.show()
    
def plot2():
    
    df = pd.DataFrame({'Tests': yTest.flatten(), 'Prediction': yPredict.flatten()})
    df.sort_index(inplace=True)
    df.plot(kind='line', figsize=(18,8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()

#plot1()
#plot2()

