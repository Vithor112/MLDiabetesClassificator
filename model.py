import pandas as pd
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

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
    for col in ['pulse_rate', 'systolic_bp', 'diastolic_bp', 'glucose', 'height', 'weight', 'bmi']:
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

df = load_health_data()
df = encode_categorical_columns(df)
X, Y = separate_features_and_target(df)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train, test in skf.split(X, Y):
    X_train, X_test = X[train], X[test]
    Y_train, Y_test = Y[train], Y[test]
    X_train, X_test = standardize_numeric_columns(pd.DataFrame(X_train, columns=df.columns[:-1]), pd.DataFrame(X_test, columns=df.columns[:-1]))
    X_train, Y_train = smote_oversampling(X_train.values, Y_train)
