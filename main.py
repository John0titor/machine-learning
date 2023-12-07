from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def load_data(data):
    # Carrega os dados do repositório e separa as features e os targets
    X = data.data.features
    y = data.data.targets
    return X, y


def split_data(X, y):
    # Divide os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y.squeeze(), test_size=0.2, random_state=42)

    # Limpa os rótulos removendo caracteres indesejados
    y_train = y_train.str.strip('.')
    y_test = y_test.str.strip('.')
    return X_train, X_test, y_train, y_test


def create_transformers(X):
    # Identifica colunas numéricas e categóricas
    categorical_columns = X.select_dtypes(include=['object']).columns
    numeric_columns = X.select_dtypes(exclude=['object']).columns

    # Cria transformadores para dados numéricos e categóricos
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combina os transformadores
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_columns),
            ('cat', categorical_transformer, categorical_columns)
        ])
    return preprocessor


def preprocess_data(preprocessor, X_train, X_test):
    # Pré-processa os conjuntos de treinamento e teste
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)
    return X_train_preprocessed, X_test_preprocessed


def train_models(model, X_train_preprocessed, y_train):
    # Treina o modelo com os dados de treinamento
    model.fit(X_train_preprocessed, y_train)
    return model


def evaluate_models(model, X_test_preprocessed, y_test):
    # Avalia o modelo nos dados de teste e calcula métricas de desempenho
    y_pred = model.predict(X_test_preprocessed)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=1)
    return accuracy, report


def evaluate(model, name, X_test_preprocessed, y_test):
    # Exibe métricas de desempenho do modelo
    accuracy, report = evaluate_models(
        model, X_test_preprocessed, y_test)
    print(f'Acurácia ' + name + ': ' + str(accuracy))
    print('Relatório de Classificação ' + name + ':\n', report)


def main():
    # Carrega os dados do repositório
    data = fetch_ucirepo(id=2)

    # Inicializa modelos
    svm = SVC(kernel='rbf', C=1.0)
    rfc = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf=1)
    lr = LogisticRegression(C=1.0, solver='liblinear')
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

    # Carrega e divide os dados
    X, y = load_data(data)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Cria transformadores para pré-processamento
    preprocessor = create_transformers(X)
    X_train_preprocessed, X_test_preprocessed = preprocess_data(
        preprocessor, X_train, X_test)

    # Treina e avalia modelos
    lr_model = train_models(lr, X_train_preprocessed, y_train)
    evaluate(lr_model, "LR", X_test_preprocessed, y_test)

    svm_model = train_models(svm, X_train_preprocessed, y_train)
    evaluate(svm_model, "SVM", X_test_preprocessed, y_test)

    rfc_model = train_models(rfc, X_train_preprocessed, y_train)
    evaluate(rfc_model, "RFC", X_test_preprocessed, y_test)

    knn_model = train_models(knn, X_train_preprocessed, y_train)
    evaluate(knn_model, "KNN", X_test_preprocessed, y_test)


if __name__ == "__main__":
    main()
