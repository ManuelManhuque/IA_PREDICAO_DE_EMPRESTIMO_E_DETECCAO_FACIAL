import pickle
import cv2
from flask import Flask, render_template, request, redirect, url_for, flash
import sqlite3
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans  

app = Flask(__name__)
app.secret_key = 'manuel#2002'



def create_table():
    conn = sqlite3.connect('contas_bancarias.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS contas_bancarias
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 nome TEXT NOT NULL,
                 sexo TEXT NOT NULL,
                 idade INTEGER NOT NULL,
                 tipo_conta TEXT NOT NULL,
                 funcionario BOOLEAN NOT NULL,
                 salario REAL,
                 saldo_conta REAL,
                 credito_imobiliario BOOLEAN NOT NULL,
                 emprestimo BOOLEAN NOT NULL)''')
    conn.commit()
    conn.close()

create_table()

# Função para obter os dados das contas bancárias
def get_banking_data():
    conn = sqlite3.connect('C:/Users/manuel/Desktop/projecto inar/contas_bancarias.db')
    c = conn.cursor()
    c.execute("SELECT * FROM contas_bancarias")
    rows = c.fetchall()
    conn.close()
    return rows

# Função para carregar o modelo treinado e os resultados do modelo
def load_model_and_results():
    model = joblib.load('C:/Users/manuel/Desktop/projecto inar/Modelos/modelo_classificacao.pkl')
    with open('C:/Users/manuel/Desktop/projecto inar/Modelos/resultados_classificacao.pkl', 'rb') as f:
        resultados = pickle.load(f)
    return model, resultados

# Carregar o modelo treinado e os resultados do modelo
model, resultados = load_model_and_results()


def load_model_and_cluster():
    cluster = None
    with open('C:/Users/manuel/Desktop/projecto inar/Modelos/modelo_clusters.pkl', 'rb') as f:
        cluster = pickle.load(f)
    return cluster

def load_regress_model():
    with open('C:/Users/manuel/Desktop/projecto inar/Modelos/modelo_regressao.pkl', 'rb') as f:
        rmodel = pickle.load(f)
    return rmodel

# Carregar o modelo treinado de regressão
rmodel = load_regress_model()


def load_regression_results():
    # Carregar os resultados da regressão
    with open('resultados_regressao.txt', 'r') as f:
        mse = float(f.readline().split(':')[1].strip())
        r2 = float(f.readline().split(':')[1].strip())

    # Carregar as imagens
    predictions_image = 'C:/Users/manuel/Desktop/projecto inar/static/imagem/previsões.png'
    residuals_plot_image = 'C:/Users/manuel/Desktop/projecto inar/static/imagem/resíduos.png'
    residuals_distribution_image = 'C:/Users/manuel/Desktop/projecto inar/static/imagem/distribuição.png'

    return mse, r2, predictions_image, residuals_plot_image, residuals_distribution_image


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image):
    # Converter a imagem para tons de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detectar rostos na imagem
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    return faces




@app.route('/')
def index():
    conn = sqlite3.connect('C:/Users/manuel/Desktop/projecto inar/contas_bancarias.db')
    c = conn.cursor()
    c.execute("SELECT * FROM contas_bancarias")
    contas_bancarias = c.fetchall()
    conn.close()
    return render_template('index.html', contas_bancarias=contas_bancarias)

@app.route('/add_account', methods=['GET', 'POST'])
def add_account():
    if request.method == 'POST':
        conn = sqlite3.connect('C:/Users/manuel/Desktop/projecto inar/contas_bancarias.db')
        c = conn.cursor()
        nome = request.form['nome']
        sexo = request.form['sexo']
        idade = request.form['idade']
        tipo_conta = request.form['tipo_conta']
        funcionario = request.form.get('funcionario', False)
        salario = request.form['salario'] if request.form['salario'] else None
        saldo_conta = request.form['saldo_conta'] if request.form['saldo_conta'] else None
        credito_imobiliario = request.form.get('credito_imobiliario', False)
        emprestimo = request.form.get('emprestimo', False)
        c.execute("INSERT INTO contas_bancarias (nome, sexo, idade, tipo_conta, funcionario, salario, saldo_conta, credito_imobiliario, emprestimo) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  (nome, sexo, idade, tipo_conta, funcionario, salario, saldo_conta, credito_imobiliario, emprestimo))
        conn.commit()
        conn.close()
        flash('Conta bancária adicionada com sucesso!', 'success')
        return redirect(url_for('index'))
    else:
        return render_template('add_account.html')

@app.route('/view_account/<int:id>', methods=['GET'])
def view_account(id):
    conn = sqlite3.connect('C:/Users/manuel/Desktop/projecto inar/contas_bancarias.db')
    c = conn.cursor()
    c.execute("SELECT * FROM contas_bancarias WHERE id=?", (id,))
    account = c.fetchone()
    conn.close()
    return render_template('view_account.html', account=account)

@app.route('/edit_account/<int:id>', methods=['GET', 'POST'])
def edit_account(id):
    conn = sqlite3.connect('C:/Users/manuel/Desktop/projecto inar/contas_bancarias.db')
    c = conn.cursor()
    if request.method == 'GET':
        c.execute("SELECT * FROM contas_bancarias WHERE id=?", (id,))
        account = c.fetchone()
        conn.close()
        return render_template('edit_account.html', account=account)
    else:
        nome = request.form['nome']
        sexo = request.form['sexo']
        idade = request.form['idade']
        tipo_conta = request.form['tipo_conta']
        funcionario = request.form.get('funcionario', False)
        salario = request.form['salario'] if request.form['salario'] else None
        saldo_conta = request.form['saldo_conta'] if request.form['saldo_conta'] else None
        credito_imobiliario = request.form.get('credito_imobiliario', False)
        emprestimo = request.form.get('emprestimo', False)
        c.execute("UPDATE contas_bancarias SET nome=?, sexo=?, idade=?, tipo_conta=?, funcionario=?, salario=?, saldo_conta=?, credito_imobiliario=?, emprestimo=? WHERE id=?",
                  (nome, sexo, idade, tipo_conta, funcionario, salario, saldo_conta, credito_imobiliario, emprestimo, id))
        conn.commit()
        conn.close()
        flash('Informações da conta bancária atualizadas com sucesso!', 'success')
        return redirect(url_for('index'))

@app.route('/delete_account/<int:id>', methods=['POST'])
def delete_account(id):
    conn = sqlite3.connect('C:/Users/manuel/Desktop/projecto inar/contas_bancarias.db')
    c = conn.cursor()
    c.execute("DELETE FROM contas_bancarias WHERE id=?", (id,))
    conn.commit()
    conn.close()
    flash('Conta bancária excluída com sucesso!', 'success')
    return redirect(url_for('index'))

@app.route('/predict_loan_approval')
def predict_loan_approval():
    # Carregar o modelo treinado e os resultados do modelo
    model, resultados = load_model_and_results()
    
    # Extrair a acurácia e a matriz de confusão dos resultados
    accuracy = resultados['accuracy']
    cm = resultados['confusion_matrix']
    
    # Plotar a matriz de confusão
    cm_plt = plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Rotulos Previstos')
    plt.ylabel('Rotulos Verdadeiros')
    plt.title('Matrix de Confusao')
    plt.savefig('C:/Users/manuel/Desktop/projecto inar/static/imagem/confusion_matrix.png')
    plt.close(cm_plt)
    
    # Passar a acurácia e o caminho para a imagem da matriz de confusão para o template HTML
    return render_template('predict_loan_approval.html', accuracy=accuracy, confusion_matrix_image='static/imagem/confusion_matrix.png')

@app.route('/predict_model')
def predict_model():
    # Passar a acurácia e o caminho para a imagem da matriz de confusão para o template HTML
    return render_template('prediction_model.html')


@app.route('/predict_model_result', methods=['POST'])
def predict_model_result():
    # Extrair os dados do formulário
    nome = request.form['nome']
    sexo = request.form['sexo']
    idade = int(request.form['idade'])
    tipo_conta = request.form['tipo_conta']
    funcionario = bool(int(request.form['funcionario']))
    salario = float(request.form['salario']) if request.form['salario'] else None
    saldo_conta = float(request.form['saldo_conta']) if request.form['saldo_conta'] else None
    credito_imobiliario = bool(int(request.form['credito_imobiliario']))
    
    # Codificar variáveis categóricas
    sexo_codificado = 1 if sexo == 'Masculino' else 0
    tipo_conta_codificado = 1 if tipo_conta == 'Corrente' else 0
    
    # Preparar os dados para a previsão
    dados_previsao = [[sexo_codificado, idade, tipo_conta_codificado, funcionario, salario, saldo_conta, credito_imobiliario]]
    
    # Prever com o modelo treinado
    resultado_previsao = model.predict(dados_previsao)
    print(resultado_previsao)
    # Traduzir o resultado da previsão para texto
    resultado_texto = 'Cliente aprovado para empréstimo.' if resultado_previsao[0] == 1 else 'Cliente não aprovado para empréstimo.'
    
    # Passar o resultado da previsão para o template HTML
    return render_template('prediction_model.html', prediction_text=resultado_texto)

@app.route('/regression_results')
def regression_results():
    # Carregar os resultados da regressão
    mse, r2, predictions_image, residuals_plot_image, residuals_distribution_image = load_regression_results()
    
    # Passar os resultados e os caminhos das imagens para o template HTML
    return render_template('regression_results.html', mse=mse, r2=r2,
                           predictions_image=predictions_image, residuals_plot_image=residuals_plot_image,
                           residuals_distribution_image=residuals_distribution_image)
  
  
  

@app.route('/predict_regress')
def predict_regress():
    return render_template('prediction_regress.html')
  
@app.route('/predict_regression_result', methods=['POST'])
def predict_regression_result():
    # Carregar o modelo treinado
    with open('C:/Users/manuel/Desktop/projecto inar/Modelos/modelo_regressao.pkl', 'rb') as f:
        model = pickle.load(f)

    # Carregar os objetos PolynomialFeatures e StandardScaler
    with open('C:/Users/manuel/Desktop/projecto inar/Modelos/poly_scaler.pkl', 'rb') as f:
        poly, scaler = pickle.load(f)

    # Extrair os dados do formulário
    idade = int(request.form['idade'])
    tipo_conta = request.form['tipo_conta']
    saldo_conta = float(request.form['saldo_conta'])

    # Codificar variáveis categóricas
    tipo_conta_codificado = 1 if tipo_conta == 'Corrente' else 0

    # Preparar os dados para a previsão
    dados_previsao = [[idade, tipo_conta_codificado, saldo_conta]]
    dados_previsao_poly = poly.transform(dados_previsao)
    dados_previsao_scaled = scaler.transform(dados_previsao_poly)

    # Fazer a previsão
    resultado_previsao = model.predict(dados_previsao_scaled)

    # Passar o resultado da previsão para o template HTML
    return render_template('prediction_regress.html', prediction_value=resultado_previsao[0])




@app.route('/clusters')
def clusters():
    # Carregar os dados
    data = pd.read_csv('C:/Users/manuel/Desktop/projecto inar/contas_bancarias.csv')
    
    # Selecionar apenas as colunas relevantes para clustering
    data_for_clustering = data[['idade', 'tipo_conta', 'saldo_conta']]
    
    # Carregar o modelo treinado
    cluster = load_model_and_cluster()
    
    # Fazer previsões de cluster
    clusters = cluster.predict(data_for_clustering)
    
    # Adicionar os clusters ao DataFrame original
    data['cluster'] = clusters
    
    # Visualizar os resultados dos clusters
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.scatter(data['idade'], data['saldo_conta'], c=data['cluster'], cmap='viridis', s=50, alpha=0.5)
    plt.xlabel('Idade')
    plt.ylabel('Saldo Conta')
    plt.title('Clusters de Clientes (Idade vs Saldo Conta)')
    plt.savefig('C:/Users/manuel/Desktop/projecto inar/static/imagem/clusters_idade_saldo.png')

    plt.subplot(1, 3, 2)
    plt.scatter(data['idade'], data['tipo_conta'], c=data['cluster'], cmap='viridis', s=50, alpha=0.5)
    plt.xlabel('Idade')
    plt.ylabel('Tipo Conta')
    plt.title('Clusters de Clientes (Idade vs Tipo Conta)')
    plt.savefig('C:/Users/manuel/Desktop/projecto inar/static/imagem/clusters_idade_tipo_conta.png')

    plt.subplot(1, 3, 3)
    plt.scatter(data['tipo_conta'], data['saldo_conta'], c=data['cluster'], cmap='viridis', s=50, alpha=0.5)
    plt.xlabel('Tipo Conta')
    plt.ylabel('Saldo Conta')
    plt.title('Clusters de Clientes (Tipo Conta vs Saldo Conta)')
    plt.savefig('C:/Users/manuel/Desktop/projecto inar/static/imagem/clusters_tipo_conta_saldo.png')
    
    return render_template('clusters.html', clusters_idade_saldo_image='C:/Users/manuel/Desktop/projecto inar/static/imagem/clusters_idade_saldo.png', clusters_idade_tipo_conta_image='C:/Users/manuel/Desktop/projecto inar/static/imagem/clusters_idade_tipo_conta.png', clusters_tipo_conta_saldo_image='C:/Users/manuel/Desktop/projecto inar/static/imagem/clusters_tipo_conta_saldo.png')


@app.route('/Visao')
def Visao():
    return render_template('Visao.html')


@app.route('/detect', methods=['POST'])
def detect():
    # Verificar se o arquivo da imagem está presente no formulário
    if 'file' not in request.files:
        return "Nenhum arquivo enviado!"

    file = request.files['file']

    # Verificar se o arquivo tem um nome e é uma imagem
    if file.filename == '':
        return "Nenhum arquivo selecionado!"
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        return "O arquivo selecionado não é uma imagem!"

    # Ler a imagem e realizar a detecção de rostos
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    faces = detect_faces(image)

    # Desenhar retângulos ao redor dos rostos detectados
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Salvar a imagem com os retângulos desenhados
    cv2.imwrite('static/detected_faces.jpg', image)

    return render_template('detect.html', detected_image='detected_faces.jpg')



if __name__ == '__main__':
    app.run(debug=True)
