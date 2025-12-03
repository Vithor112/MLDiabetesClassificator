import pandas as pd
import numpy as np
import csv
import os
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, fbeta_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

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



def standardize_numeric_columns(train, test, numeric_indices):
    """
    Padroniza as colunas numéricas usando StandardScaler.

    Parâmetros:
    train (array numpy): Conjunto de dados de treinamento.
    test (array numpy): Conjunto de dados de teste.
    numeric_indices (list): Lista de índices das colunas numéricas a serem padronizadas
    Retorna:
    tuple: Conjuntos de dados de treinamento e teste com colunas numéricas padron
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_indices)
        ],
        remainder='passthrough'
    )

    X_train_scaled = preprocessor.fit_transform(train)
    X_test_scaled = preprocessor.transform(test)
    return (X_train_scaled, X_test_scaled)
    
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
    try:
        probs = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(Y_test, probs)
    except AttributeError:
        auc = 0.0 
    return {
        'Accuracy': accuracy,
        'F2-Score': f2,
        'Precision': precision,
        'Recall': recall,
        'AUC': auc
    }

def train_and_evaluate_models(X_train, X_test, Y_train, Y_test):

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    Y_train = np.asarray(Y_train)
    Y_test = np.asarray(Y_test)

    models = [
        ('GaussianNB', GaussianNB()),
        ('RandomForest', RandomForestClassifier(random_state=42)),
        ('LogisticRegression', LogisticRegression(max_iter=2000, random_state=42)),
        ('MLP', MLPClassifier(max_iter=1000, random_state=42)),
        ('KNN-3', KNeighborsClassifier(n_neighbors=3)),
        ('KNN-5', KNeighborsClassifier(n_neighbors=5)),
        ('KNN-7', KNeighborsClassifier(n_neighbors=7)),
    ]

    try:
        base_tree = DecisionTreeClassifier(random_state=42)
        base_tree.fit(X_train, Y_train)
        path = base_tree.cost_complexity_pruning_path(X_train, Y_train)
        alphas = [a for a in path.ccp_alphas if a > 0]
        chosen_alpha = 0.0
        if alphas:
            candidates = np.linspace(min(alphas), max(alphas), min(len(alphas), 5))
            best_alpha = candidates[0]
            best_score = -1
            # validação cruzada para cada alpha candidato
            for a in candidates:
                clf_tmp = DecisionTreeClassifier(random_state=42, ccp_alpha=float(a))
                scores = cross_val_score(clf_tmp, X_train, Y_train, cv=3)
                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_alpha = float(a)

            chosen_alpha = best_alpha
        # define o modelo final com o alpha escolhido
        pruned_tree = DecisionTreeClassifier(random_state=42, ccp_alpha=chosen_alpha)
        # nome dinâmico baseado no uso de poda
        name_tree = 'DecisionTree-pruned' if chosen_alpha > 0 else 'DecisionTree'
        models.insert(2, (name_tree, pruned_tree))
    except Exception as e:
        print(f"Erro durante seleção de ccp_alpha: {e}")
        models.insert(2, ('DecisionTree', DecisionTreeClassifier(random_state=42)))
    # treinar e avaliar todos os modelos
    models_results = []
    for name, clf in models:
        try:
            res = evaluate_models(clf, X_train, Y_train, X_test, Y_test)
            res['Model'] = name
            models_results.append(res)
        except Exception as e:
            print(f"Erro treinando/avaliando {name}: {e}")
    return models_results


def save_results_to_csv(models_results, repetition, fold, file_path='model_results.csv'):
    try:
        file_exists = os.path.isfile(file_path)
        with open(file_path, mode='a', newline='') as file:
            fieldnames = ['Model', 'Repetition', 'Fold', 'Accuracy', 'F2-Score', 'Precision', 'Recall', 'AUC']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            for result in models_results:
                row = result.copy()
                row['Repetition'] = repetition
                row['Fold'] = fold
                writer.writerow({k: row.get(k) for k in fieldnames})
                
    except Exception as e:
        print(f"Erro ao salvar: {e}")

REPETITION_COUNT = 10

df = load_health_data()
df = encode_categorical_columns(df)
df = calculate_pulse_pressure(df)
df = drop_unnecessary_columns(df)
target_numeric_cols = ['pulse_rate', 'glucose', 'bmi', 'pulse_pressure']
numeric_col_indices = [df.columns.get_loc(c) for c in df if c in target_numeric_cols]
X, Y = separate_features_and_target(df)
for repetition in range(REPETITION_COUNT):
    print("Repetition:", repetition + 1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + repetition)
    for fold_idx, (train, test) in enumerate(skf.split(X, Y), start=1):
        X_train, X_test = X[train], X[test]
        Y_train, Y_test = Y[train], Y[test]
        X_train, X_test = standardize_numeric_columns(X_train, X_test, numeric_col_indices)
        X_train, Y_train = smote_oversampling(X_train, Y_train)
        models_results = train_and_evaluate_models(X_train, X_test, Y_train, Y_test)
        save_results_to_csv(models_results, repetition, fold_idx)
print("Processo concluído.")