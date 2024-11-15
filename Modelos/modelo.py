import joblib
import pickle
import csv
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# Carregando os dados do arquivo CSV
data = []
with open('C:/Users/manuel/Desktop/projecto inar/contas_bancarias.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        data.append(row)

# Preparando os dados para treinamento do modelo
X = []
y = []
for row in data:
    X.append([row['sexo'], int(row['idade']), row['tipo_conta'], int(row['funcionario']), float(row['salario']), float(row['saldo_conta']), int(row['credito_imobiliario'])])
    y.append(int(row['emprestimo']))
le = LabelEncoder()
X_encoded = [le.fit_transform(x) for x in X]  # Convertendo dados categóricos em numéricos
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

# Treinamento do modelo de aprendizado de máquina
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Salvando o modelo treinado
joblib.dump(model, 'modelo_classificacao.pkl')

# Salvando os resultados do modelo (acurácia e matriz de confusão)
resultados = {
    'accuracy': accuracy,
    'confusion_matrix': cm
}

with open('resultados_classificacao.pkl', 'wb') as f:
    pickle.dump(resultados, f)

print("Modelo treinado e resultados salvos com sucesso!")
