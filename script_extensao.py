import pandas as pd
import random

# Função para gerar nomes fictícios
def gerar_nome(sexo):
    nomes_masculinos = ["Malik", "Pedro", "Fábio", "Rui", "Carlos", "Paulo", "Alex", "Miguel", "André", "Renato"]
    nomes_femininos = ["Joana", "Ana", "Maria", "Luciana", "Helena", "Rita", "Sara", "Beatriz", "Isabela", "Carla"]
    sobrenomes = ["Tavares", "Mendonça", "Silva", "Ramos", "Costa", "Santos", "Martins", "Alves", "Mota", "Rocha", "Sousa", "Monteiro", "Fernandes", "Teixeira", "Pereira", "Torres", "Carvalho", "Oliveira"]

    primeiro_nome = random.choice(nomes_masculinos if sexo == 1 else nomes_femininos)
    sobrenome = random.choice(sobrenomes)
    return f"{primeiro_nome} {sobrenome}"

# Lista para armazenar os registros
dados = []

# Gerando registros com IDs de 1 até 3000000
for i in range(1, 300000):
    sexo = random.choice([0, 1])  # 0 = Feminino, 1 = Masculino
    nome = gerar_nome(sexo)
    idade = random.randint(20, 60)
    tipo_conta = random.choice([1, 2, 3])  # Assume-se que há 3 tipos de conta
    funcionario = random.choice([0, 1])
    salario = round(random.uniform(10000, 30000), 2)
    saldo_conta = round(random.uniform(1000, 20000), 2)
    credito_imobiliario = random.choice([0, 1])
    emprestimo = random.choice([0, 1])
    cluster = random.choice([1, 2, 3])

    # Adicionando o registro à lista
    dados.append([i, nome, sexo, idade, tipo_conta, funcionario, salario, saldo_conta, credito_imobiliario, emprestimo, cluster])

# Convertendo a lista em um DataFrame do pandas
colunas = ["id", "nome", "sexo", "idade", "tipo_conta", "funcionario", "salario", "saldo_conta", "credito_imobiliario", "emprestimo", "cluster"]
df = pd.DataFrame(dados, columns=colunas)

# Salvar o DataFrame como CSV com codificação UTF-8
df.to_csv("C:/Users/manuel/Desktop/projecto inar/contas_bancarias.csv", index=False, encoding='utf-8')

print("Arquivo 'contas_bancarias.csv' gerado com sucesso com codificação UTF-8!")
