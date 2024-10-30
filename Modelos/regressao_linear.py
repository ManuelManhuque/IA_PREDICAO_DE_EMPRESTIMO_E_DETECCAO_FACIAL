import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Carregar o arquivo CSV
df = pd.read_csv('contas_bancarias.csv')

# Prepare data
X = df[['idade', 'tipo_conta', 'saldo_conta']]
y = df['salario']

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Engenharia de Recursos
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Normalização dos dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)

# Modelo com Regularização (Ridge)
model = Ridge(alpha=0.1)
model.fit(X_train_scaled, y_train)

# Fazer previsões
y_pred = model.predict(X_test_scaled)

# Avaliação do modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Salvar o modelo treinado
with open('modelo_regressao.pkl', 'wb') as f:
    pickle.dump(model, f)

# Salvar os objetos PolynomialFeatures e StandardScaler
with open('poly_scaler.pkl', 'wb') as f:
    pickle.dump((poly, scaler), f)
