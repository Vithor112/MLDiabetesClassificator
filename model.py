import pandas as pd
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

def preprocess_health_data(df, categorical_to_numerical : bool = False):
    """
    Realiza pré-processamento básico com SMOTE, deve ser aplicado somente no conjunto de treino.

    Parâmetros:
    df (pd.DataFrame): DataFrame contendo os dados

    Retorna:
    pd.DataFrame: DataFrame pré-processado.
    """
    if df is None:
        return None
    
    # Padronizando colunas numéricas com zscore
    cols_to_norm = ['pulse_rate', 'systolic_bp', 'diastolic_bp', 'glucose', 'height', 'weight', 'bmi']
    df[cols_to_norm] = (df[cols_to_norm] - df[cols_to_norm].mean()) / df[cols_to_norm].std()

    # Mapeando categorias para valores numéricos
    if categorical_to_numerical:
        mapping = {'No': 0, 'Yes': 1}
        df['diabetic'] = df['diabetic'].map(mapping)
        mapping = {'Male': 0, 'Female': 1}  
        df['gender'] = df['gender'].map(mapping)

    X = df.drop(columns=['diabetic'])
    Y = df['diabetic']

    X_resampled, y_resampled = SMOTE().fit_resample(X, Y)
    return (X_resampled, y_resampled)


df = load_health_data()
x,y = preprocess_health_data(df, categorical_to_numerical=True)
