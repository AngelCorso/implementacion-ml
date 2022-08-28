import pandas as pd
from multipleLinearRegression import multipleLinearRegression

df = pd.read_csv('CaloriesPrediction.csv')
df.head()

y = df['Y: Cantidad de calorias (kcal)']
X = df[['X1: Cantidad de grasas (gr)',	'X2: Cantidad de proteinas (gr)','X3: Cantidad de carbohidratos (gr)']]


model = multipleLinearRegression()

model.fit(y,X)

print('Predicciones: ', model.predict(X))

print('RMSE: ', model.rmse(y))
