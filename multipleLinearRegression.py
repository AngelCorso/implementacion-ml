import numpy as np
import math

class multipleLinearRegression:

  def fit(self,y,X):
    y = y.to_numpy()
    Xm = X.to_numpy()

    self.B = np.dot(np.linalg.inv(np.dot(Xm.transpose(),Xm)), np.dot(Xm.T,y))

    b0 = 0

    for i in range(len(y)):
      b0 += y[i] - np.dot(Xm[i].T, self.B)
 
    self.b0 = b0 / len(y)

    pass

  def predict(self,X):
    self.y_ = []
    Xm = X.to_numpy()

    for j in range(len(Xm)):
      result = 0
      for k in range(len(self.B)):
        result += self.B[k] * Xm[j][k]

      self.y_.append(result)

    return self.y_
    

  def rmse(self,y):
    mse = np.mean((y - self.y_) ** 2)
    return math.sqrt(mse)

# df = pd.read_csv('CaloriesPrediction.csv')
# df.head()

# y = df['Y: Cantidad de calorias (kcal)']
# X = df[['X1: Cantidad de grasas (gr)',	'X2: Cantidad de proteinas (gr)','X3: Cantidad de carbohidratos (gr)']]


# model = multipleLinearRegression()

# model.fit(y,X)

# model.predict(X)

# model.rmse(y)