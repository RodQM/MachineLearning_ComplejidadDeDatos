from sklearn.datasets import load_breast_cancer
import pandas as pd

# Cargar el dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)  # Features
y = pd.Series(data.target, name='target')  # Target (clase)

# Mostrar las primeras filas del dataset
print(X.head())
print(y.value_counts())  # Ver la distribución de clases

# Ver si hay valores perdidos
print(X.isnull().sum())

# Si hubiese valores perdidos, podrías eliminarlos o imputarlos
# X = X.dropna()  # Opción para eliminar valores nulos
# O imputar con la media, por ejemplo:
# X = X.fillna(X.mean())

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Dividimos en train y test antes de aplicar SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplicamos SMOTE solo en los datos de entrenamiento
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Verificamos la nueva distribución de clases
print(y_train_res.value_counts())  # Ahora debería estar balanceado

# Verificar nuevamente los valores perdidos
print(X_train_res.isnull().sum())

# Verificar balance de clases después de SMOTE
print(y_train_res.value_counts())


# Guardar el dataset transformado después de aplicar SMOTE en un archivo CSV
X_train_res['target'] = y_train_res  # Añadir la columna de la clase balanceada al dataset de entrenamiento
X_train_res.to_csv('dataset_transformado.csv', index=False)  # Guardar el dataset como CSV

print("Dataset transformado guardado como 'dataset_transformado.csv'")
