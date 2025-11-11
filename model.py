import pandas as pd
import numpy as np
import csv
import os
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, fbeta_score
from sklearn.model_selection import train_test_split

#models
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

def load_health_data(file_path='diabd_dataset.csv'):
    """
    Carrega o dataset DiaBD a partir do arquivo CSV especificado.

    Parâmetros:
    file_path (str): Caminho para o arquivo CSV do dataset.

    Retorna:
    pd.DataFrame: DataFrame contendo os dados carregados.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Arquivo '{file_path}' carregado com sucesso.")
        return df
    except FileNotFoundError:
        print(f"Erro: Arquivo '{file_path}' não encontrado.")
        return None



def standardize_numeric_columns(train, test):
    """
    Padroniza as colunas numéricas usando z-score.

    Parâmetros:
    train (pd.DataFrame): Conjunto de dados de treinamento.
    test (pd.DataFrame): Conjunto de dados de teste.
    numeric_columns (list): Lista de nomes das colunas numéricas a serem padronizadas.

    Retorna:
    tuple: Conjuntos de dados de treinamento e teste com colunas numéricas padronizadas.
    """
    for col in ['pulse_rate', 'pulse_pressure', 'glucose', 'bmi']:
        mean = train[col].mean()
        std = train[col].std()
        train[col] = (train[col] - mean) / std
        test[col] = (test[col] - mean) / std
    return train, test
    
def encode_categorical_columns(df):
    """
    Realiza one-hot encoding nas colunas categóricas.

    Parâmetros:
    df (pd.DataFrame): DataFrame contendo os dados.

    Retorna:
    pd.DataFrame: DataFrame com colunas categóricas codificadas.
    """
    mapping = {'No': 0, 'Yes': 1}
    df['diabetic'] = df['diabetic'].map(mapping)
    mapping = {'Male': 0, 'Female': 1}  
    df['gender'] = df['gender'].map(mapping)
    return df

def separate_features_and_target(df):
    """
    Separa as features (X) e o target (Y) do DataFrame.

    Parâmetros:
    df (pd.DataFrame): DataFrame contendo os dados.

    Retorna:
    tuple: Arrays numpy contendo as features (X) e o target (Y).
    """
    X = df.drop('diabetic', axis=1).values
    Y = df['diabetic'].values
    return X, Y

def smote_oversampling(X, Y):
    """
    Aplica SMOTE para balancear o dataset. Deve ser aplicado somente no conjunto de treinamento.

    Parâmetros:
    X (np.ndarray): Features.
    Y (np.ndarray): Target.

    Retorna:
    tuple: Arrays numpy contendo as features (X_res) e o target (Y_res) balanceados.
    """
    smote = SMOTE(random_state=42)
    X_res, Y_res = smote.fit_resample(X, Y)
    return X_res, Y_res

def drop_unnecessary_columns(df):
    """
    Remove colunas desnecessárias do DataFrame.

    Parâmetros:
    df (pd.DataFrame): DataFrame contendo os dados.

    Retorna:
    pd.DataFrame: DataFrame com colunas desnecessárias removidas.
    """
    columns_to_drop = ['weight', 'height', 'systolic_bp', 'diastolic_bp']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    return df

def calculate_pulse_pressure(df):
    """
    Calcula a pressão de pulso e adiciona como uma nova coluna no DataFrame.

    Parâmetros:
    df (pd.DataFrame): DataFrame contendo os dados.

    Retorna:
    pd.DataFrame: DataFrame com a nova coluna 'pulse_pressure'.
    """
    pp_values = df['systolic_bp'] - df['diastolic_bp']
    insert_loc = df.columns.get_loc('diastolic_bp') + 1
    df.insert(insert_loc, 'pulse_pressure', pp_values)
    return df

def evaluate_models(clf, X_train, Y_train, X_test, Y_test):
    """
    Treina e avalia o modelo fornecido, retornando as métricas de avaliação.

    Parâmetros:
    clf: Classificador a ser treinado e avaliado.
    X_train (np.ndarray): Features do conjunto de treinamento.
    Y_train (np.ndarray): Target do conjunto de treinamento.
    X_test (np.ndarray): Features do conjunto de teste.
    Y_test (np.ndarray): Target do conjunto de teste.

    Retorna:
    dict: Dicionário contendo as métricas de avaliação do modelo.
    """
    clf.fit(X_train, Y_train)
    preds = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, preds)
    report_dict = classification_report(Y_test, preds, output_dict=True)
    precision = report_dict.get('1', {}).get('precision', 0.0)
    recall = report_dict.get('1', {}).get('recall', 0.0)
    f2 = fbeta_score(Y_test, preds, beta=2, zero_division=0)

    return {
        'accuracy': accuracy,
        'F2-Score': f2,
        'Precision': precision,
        'Recall': recall
    }

def train_and_evaluate_models(X_train, X_test, Y_train, Y_test):

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    Y_train = np.asarray(Y_train)
    Y_test = np.asarray(Y_test)

    models = [
        ('GaussianNB', GaussianNB()),
        ('RandomForest', RandomForestClassifier(random_state=42)),
        # DecisionTree será adicionado depois com poda
        ('LogisticRegression', LogisticRegression(max_iter=2000, random_state=42)),
        ('MLP', MLPClassifier(max_iter=1000, random_state=42)),
        ('KNN-3', KNeighborsClassifier(n_neighbors=3)),
        ('KNN-5', KNeighborsClassifier(n_neighbors=5)),
        ('KNN-7', KNeighborsClassifier(n_neighbors=7)),
    ]

    # Decision Tree com seleção simples de ccp_alpha via split interno
    try:
        base_tree = DecisionTreeClassifier(random_state=42)
        base_tree.fit(X_train, Y_train)
        path = base_tree.cost_complexity_pruning_path(X_train, Y_train)
        alphas = [a for a in path.ccp_alphas if a > 0]
        chosen_alpha = 0.0
        if alphas:
            # pequena validação interna para escolher alpha
            X_tr, X_val, y_tr, y_val = train_test_split(X_train, Y_train, test_size=0.2, stratify=Y_train, random_state=42)
            best_alpha = alphas[0]
            best_acc = -1
            # testar até 5 candidatos espaçados
            candidates = np.linspace(min(alphas), max(alphas), min(len(alphas), 5))
            for a in candidates:
                clf_tmp = DecisionTreeClassifier(random_state=42, ccp_alpha=float(a))
                clf_tmp.fit(X_tr, y_tr)
                preds_val = clf_tmp.predict(X_val)
                acc = accuracy_score(y_val, preds_val)
                if acc > best_acc:
                    best_acc = acc
                    best_alpha = float(a)
            chosen_alpha = best_alpha

        pruned_tree = DecisionTreeClassifier(random_state=42, ccp_alpha=chosen_alpha)
        models.insert(2, ('DecisionTree-pruned', pruned_tree))
    except Exception as e:
        # fallback: árvore padrão
        models.insert(2, ('DecisionTree', DecisionTreeClassifier(random_state=42)))

    # treinar e mostrar relatório como no notebook
    models_results = []
    for name, clf in models:
        try:
            res = evaluate_models(clf, X_train, Y_train, X_test, Y_test)
            # adicionar nome do modelo ao dicionário de métricas
            res['model'] = name
            models_results.append(res)
        except Exception as e:
            print(f"Erro treinando/avaliando {name}: {e}")
    return models_results

def save_results_to_csv(models_results, repetition, fold, file_path='model_results.csv'):
    """
    Salva os resultados dos modelos em um arquivo CSV.

    Parâmetros:
    models_results (list): Lista de dicionários contendo os resultados dos modelos.
    repetition (int): Número da repetição atual.
    fold (int): Número do fold atual.
    file_path (str): Caminho para o arquivo CSV onde os resultados serão salvos.
    """

    try:
        file_exists = os.path.isfile(file_path)
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Model', 'Repetition', 'Fold', 'Accuracy', 'F2-Score', 'Precision', 'Recall'])
            for result in models_results:
                writer.writerow([
                    result['model'],
                    repetition,
                    fold,
                    result['accuracy'],
                    result['F2-Score'],
                    result['Precision'],
                    result['Recall']
                ])
        print(f"Resultados salvos em {file_path}")
    except Exception as e:
        print(f"Erro ao salvar resultados em {file_path}: {e}")

REPETITION_COUNT = 10

df = load_health_data()
df = encode_categorical_columns(df)
df = calculate_pulse_pressure(df)
df = drop_unnecessary_columns(df)
X, Y = separate_features_and_target(df)
for repetition in range(REPETITION_COUNT):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # enumerar os folds para passar o índice correto ao salvar
    for fold_idx, (train, test) in enumerate(skf.split(X, Y), start=1):
        X_train, X_test = X[train], X[test]
        Y_train, Y_test = Y[train], Y[test]
        X_train, X_test = standardize_numeric_columns(pd.DataFrame(X_train, columns=df.columns[:-1]), pd.DataFrame(X_test, columns=df.columns[:-1]))
        X_train, Y_train = smote_oversampling(X_train.values, Y_train)

        # Realizar treinamento e avaliação do modelo aqui (Random forest, Nayve Bayes, Decision Tree com poda, Rede Neural, Regressão logistica, Knn (3,5,7 vizinhos))
        # TODO calcular métricas de avalição do modelo (AUC F2-Score precisão recall)
        models_results = train_and_evaluate_models(X_train, X_test, Y_train, Y_test)
        # salvar métricas em um arquivo .csv para análise posterior
        save_results_to_csv(models_results, repetition, fold_idx)
        # necessitamos ter varias metricas para cada modelo para calcular desvio padrão e média e significancia estatistica 
        # TODO modelo header csv (Model, Repetition, Fold, AUC F2-Score, Precision, Recall)
