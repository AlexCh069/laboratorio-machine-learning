import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.combine import SMOTEENN
import seaborn as sns

# Supongamos que X e y son tus datos originales
# smote_enn = SMOTEENN(random_state=42)
# X_resampled, y_resampled = smote_enn.fit_resample(X, y)

# Contar la distribución antes y después
def plot_contrast_distribution(data:pd.DataFrame, 
                               data_resample:pd.DataFrame, 
                               target:str = None):

    if target is None:
        y = data[data.columns[-1]]
        y_resampled = data_resample[data_resample.columns[-1]]

    else: 
        y = data[target]
        y_resampled = data_resample[target]

    original_counts = Counter(y)
    resampled_counts = Counter(y_resampled)

    # Graficar las distribuciones
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].bar(original_counts.keys(), original_counts.values(), color='blue', alpha=0.7)
    ax[0].set_title("Distribución antes de SMOTEENN")
    ax[0].set_xlabel("Clases")
    ax[0].set_ylabel("Número de muestras")

    ax[1].bar(resampled_counts.keys(), resampled_counts.values(), color='green', alpha=0.7)
    ax[1].set_title("Distribución después de SMOTEENN")
    ax[1].set_xlabel("Clases")

    plt.show()

def contrast_stats(data:pd.DataFrame,
                       data_resampled:pd.DataFrame,
                       target:str = None):
    """MEJORAR MUCHO: SOLO ESTADISICOS RELEVANTES, NO TODOS"""

    # Comparar estadísticas
    stats_original = data.describe()
    stats_resampled = data_resampled.describe()

    # Mostrar comparaciones
    print("Estadísticas antes del muestreo:\n", stats_original)
    print("\nEstadísticas después del muestreo:\n", stats_resampled)

def constast_distributions(data:pd.DataFrame,
                           data_resampled:pd.DataFrame,
                           target:str = None):

    """Documentar y mejorar"""

    X = data.drop(data.columns[-1], axis = 1)

    
    for col in X.columns:
        sns.kdeplot(data[col], label="Original", fill=True)
        sns.kdeplot(data_resampled[col], label="Resampled", fill=True, alpha=0.5)
        plt.title(f"Distribución de {col} antes y después de SMOTEENN")
        plt.legend()
        plt.show()
        print(' ')