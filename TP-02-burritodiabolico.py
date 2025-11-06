import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
'''from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error'''
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif #ver si lo podemos usar

#%%

sns.set_theme(style="whitegrid")

#df_full = pd.read_csv(r"C:\Users\andre\OneDrive\Documents\TP2-Labo-de-datos\kuzushiji_full.csv", header = 0) #cada uno ponga su direcccion hasta q podamos hacer lo del directorio relativo
df_full = pd.read_csv(r"C:\Users\andre\OneDrive\Documents\TP2-Labo-de-datos\kuzushiji_full.csv")
#%%
# TP2 - PARTE 1: ANÁLISIS EXPLORATORIO DE DATOS 

# Separar datos (X) y clases (y)
y_data = df_full.iloc[:, 784]
X_data_raw = df_full.iloc[:,:-1] 

# Limpieza y Normalización
X_data_numeric = X_data_raw.apply(pd.to_numeric, errors='coerce').fillna(0)
# Normalizar los píxeles (escalar de 0-255 a 0-1)
X_data = X_data_numeric / 255.0

#%%
# Info para el informe
cantidad_datos = X_data.shape[0]
cantidad_atributos = X_data.shape[1]
tipo_atributo = X_data.iloc[0].dtype
clases_unicas = sorted(y_data.unique())
cantidad_clases = len(clases_unicas)


# Gráfico de balanceo de clases

plt.figure(figsize=(12, 7))
ax_balance = sns.countplot(
    x=y_data.astype(str), 
    order=[str(i) for i in range(10)], # Asegurar el orden de '0' a '9'
    palette='viridis'
)
ax_balance.set_title('Distribución de Clases (Dataset Completo)', fontsize=16, weight='bold')
ax_balance.set_xlabel('Clase (Etiqueta Numérica)', fontsize=12)
ax_balance.set_ylabel('Cantidad de Imágenes', fontsize=12)

# Añadir etiquetas de conteo
for p in ax_balance.patches:
    ax_balance.annotate(f'{p.get_height()}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points')
                        
plt.tight_layout()
plt.show()


#%%
# Gráfico para pregunta 1.a (Atributos Relevantes)

plt.figure(figsize=(6, 6))
average_image = X_data.mean().to_numpy().reshape(28, 28)

plt.imshow(average_image, cmap='hot') 
plt.title('1.a: Imagen Promedio (Atributos Relevantes)')
plt.colorbar() 
plt.axis('off')
plt.savefig('grafico_1a_atributos_relevantes.png')
plt.show() 
#%%
# Gráfico para Pregunta 1.b (Similitud entre Clases)

df_avg_class = X_data.groupby(y_data).mean()

plt.figure(figsize=(12, 6))
for i in range(10): 
    plt.subplot(2, 5, i + 1)     
    # Obtener la fila de la clase 'i' y reformatearla a 28x28
    imagen_promedio_clase = df_avg_class.loc[i].to_numpy().reshape(28, 28)
    plt.imshow(imagen_promedio_clase, cmap='gray')
    plt.title(f"Clase Promedio: {i}")
    plt.axis('off')

plt.suptitle('1.b: Similitud entre Clases (Imagen Promedio por Clase)', fontsize=18, weight='bold')
plt.subplots_adjust(hspace=0.4)
plt.savefig('grafico_1b_similitud_clases.png')
plt.show() 
#%%
# Gráfico para Pregunta 1.c (Similitud Intra-Clase)

CLASE_A_INSPECCIONAR = 8 

X_clase_8 = X_data[y_data == CLASE_A_INSPECCIONAR]

# Seleccion de16 índices aleatorios DE ESA CLASE
np.random.seed(42) 
indices_aleatorios_clase_8 = np.random.choice(X_clase_8.index, 16, replace=False)

plt.figure(figsize=(10, 10))
for i, idx in enumerate(indices_aleatorios_clase_8):
    plt.subplot(4, 4, i + 1)
    imagen = X_data.loc[idx].to_numpy().reshape(28, 28)
    plt.imshow(imagen, cmap='gray')
    plt.title(f"Ejemplo {i+1} (Clase {CLASE_A_INSPECCIONAR})")
    plt.axis('off')

plt.suptitle(f'1.c: Similitud Intra-Clase (Muestras Aleatorias Clase {CLASE_A_INSPECCIONAR})', fontsize=18, weight='bold')
plt.subplots_adjust(hspace=0.4, wspace=0.2) # Añadir espacio
plt.savefig('grafico_1c_similitud_intra_clase.png')
plt.show() 

#%%
# TP2 - PARTE 2: CLASIFICACiÓN BINARIA

df_binario = df_full[df_full['label'].isin([4, 5])]


# Separar datos Clase 4 y 5 
X_data4_5 = X_data[y_data.isin([4,5])]
y_data4_5 = y_data[y_data.isin([4,5])]

# Información para el informe
cantidad_de_muestras = y_data4_5.size
valores_clase_4 = y_data[y_data == 4].size
valores_clase_5 = y_data[y_data == 5].size

#print(cantidad_de_muestras)
#print(valores_clase_4)
#print(valores_clase_5)

# Separar los datos en conjuntos train y test
X_train, X_test, y_train, y_test = train_test_split(X_data4_5, y_data4_5, test_size=0.25, random_state=5)

# Seleccionar l pixeles mas significativos como atributos
l = 50

valores_k = [3,5,10]
'''
resultados = []

for k in valores_k:
    modelo = KNeighborsRegressor(n_neighbors=k)
    promedios = X_train.mean(axis=0)
    indices = np.argsort(promedios.values)[-l:][::-1]
    X_train0 = X_train.iloc[:, indices]
    X_test0 = X_test.iloc[:, indices]
    modelo.fit(X_train0, y_train)

    y_pred_train = modelo.predict(X_train0)
    y_pred_test = modelo.predict(X_test0)

    resultados.append({
        'k': k,
        'Train_RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'Test_RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'Train_MAE': mean_absolute_error(y_train, y_pred_train),
        'Test_MAE': mean_absolute_error(y_test, y_pred_test),
    })

# Mostrar resultados
df_resultados = pd.DataFrame(resultados)
print(df_resultados)'''


# Classifier 

resultados2= []

for k in valores_k:
    modelo = KNeighborsClassifier(n_neighbors=k)
    promedios = X_train.mean(axis=0)
    indices = np.argsort(promedios.values)[-l:][::-1]
    X_train0 = X_train.iloc[:, indices]
    X_test0 = X_test.iloc[:, indices]
    modelo.fit(X_train0, y_train)

    #y_pred_train = modelo.predict(X_train0)
    y_pred_test = modelo.predict(X_test0)

    resultados2.append({
        'k': k,
        'Accuracy': accuracy_score(y_test, y_pred_test),
        'Matriz Confusion': confusion_matrix(y_test, y_pred_test),
    })

df_resultados2 = pd.DataFrame(resultados2)
print(df_resultados2)

resultados3 = []

selector = SelectKBest(f_classif, k=l)
X_train_fs = selector.fit_transform(X_train, y_train)
X_test_fs = selector.transform(X_test) 

for k in valores_k:
    modelo = KNeighborsClassifier(n_neighbors=k)
    modelo.fit(X_train_fs, y_train)
    
    #y_pred_train = modelo.predict(X_train_fs)
    y_pred_test = modelo.predict(X_test_fs)
    
    resultados3.append({
        'k': k,
        'Accuracy': accuracy_score(y_test, y_pred_test),
        'Matriz Confusion': confusion_matrix(y_test, y_pred_test),
    })

df_resultados3 = pd.DataFrame(resultados3)
print(df_resultados3)

#%%
# TP2 - PARTE 3: ARBOLES DEDECISION
# Árbol de Decisión con "gini"

from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

# 1. Definir la metodología de Validación Cruzada
# -----------------------------------------------
nsplits = 5
kfold_cv = KFold(n_splits=nsplits, shuffle=True, random_state=42)

# Separar datos en Desarrollo (80%) y Evaluación (20%)
X_dev, X_eval, y_dev, y_eval = train_test_split(
    X_data, 
    y_data, 
    test_size=0.2, 
    random_state=42,
    stratify=y_data)

# 2. Experimentación con Árbol de Decisión usando 'gini'

# Probamos profundidades de 1 a 10
valores_depth_gini = [1,3,5,7,10] 

# Matriz de resultados: 1 fila por fold, 1 columna por 'max_depth'
resultados_dt_gini = np.zeros((nsplits, len(valores_depth_gini)))

# Loop K-Fold splits (usando X_dev, y_dev)
for i, (train_index, test_index) in enumerate(kfold_cv.split(X_dev)):
    
    # Separar los datos de este fold
    kf_X_train, kf_X_test = X_dev.iloc[train_index], X_dev.iloc[test_index]
    kf_y_train, kf_y_test = y_dev.iloc[train_index], y_dev.iloc[test_index]
    
    # Loop hiperparámetros
    for j, depth in enumerate(valores_depth_gini):
        
        dt_gini = DecisionTreeClassifier(
            criterion='gini',  # <-- CONSIGNA 'b' 
            max_depth=depth, 
            random_state=42
        )
        dt_gini.fit(kf_X_train, kf_y_train)
               
        pred = dt_gini.predict(kf_X_test)
        score = accuracy_score(kf_y_test, pred)
         
        resultados_dt_gini[i, j] = score

#%%
# RESULTADOS Y GRÁFICO ("gini")

# Calcular scores promedio
scores_promedio_dt_gini = resultados_dt_gini.mean(axis=0)

# Encontrar el mejor hiperparámetro
mejor_depth_index_gini = np.argmax(scores_promedio_dt_gini)
mejor_depth_gini = valores_depth_gini[mejor_depth_index_gini]
mejor_score_dt_gini = scores_promedio_dt_gini[mejor_depth_index_gini]

print(f"Mejor Hiperparámetro (max_depth con 'gini'): {mejor_depth_gini}")
print(f"Mejor Accuracy Promedio (CV con 'gini'): {mejor_score_dt_gini * 100:.2f}%")

#  Preparar DataFrame para el Boxplot
df_resultados_dt_gini = pd.DataFrame(resultados_dt_gini, columns=valores_depth_gini)
df_melted_dt_gini = df_resultados_dt_gini.melt(
    var_name='max_depth (Profundidad)', 
    value_name='Accuracy'
)

#  Generar Boxplot de DT (gini)
plt.figure(figsize=(12, 7))
sns.boxplot(data=df_melted_dt_gini, x='max_depth (Profundidad)', y='Accuracy', palette='plasma')
plt.title("Rendimiento de 'gini' vs. Profundidad Máxima (5-fold CV)", fontsize=16, weight='bold')
plt.xlabel('Profundidad Máxima ', fontsize=12)
plt.ylabel('Accuracy del Fold', fontsize=12)
plt.tight_layout()
plt.show()

#%%
# Árbol de Decisión con 'entropy'

valores_depth_entropy = [1,3,5,7,10]

# Matriz de resultados: 1 fila por fold, 1 columna por 'max_depth'
resultados_dt_entropy = np.zeros((nsplits, len(valores_depth_entropy)))
# Loop K-Fold splits (usando X_dev, y_dev)
for i, (train_index, test_index) in enumerate(kfold_cv.split(X_dev)):
    print(f"  DT (entropy): Fold {i+1}/{nsplits}...")
    # Separar los datos de este fold
    kf_X_train, kf_X_test = X_dev.iloc[train_index], X_dev.iloc[test_index]
    kf_y_train, kf_y_test = y_dev.iloc[train_index], y_dev.iloc[test_index]
    # Loop hiperparámetros
    for j, depth in enumerate(valores_depth_entropy):
    
        dt_entropy = DecisionTreeClassifier(
            criterion='entropy',  
            max_depth=depth, 
            random_state=42
        )
        dt_entropy.fit(kf_X_train, kf_y_train)
        
        pred = dt_entropy.predict(kf_X_test)
        score = accuracy_score(kf_y_test, pred)
        
        resultados_dt_entropy[i, j] = score

#%%
# RESULTADOS Y GRÁFICO (entropy)

# Calcular scores promedio
scores_promedio_dt_entropy = resultados_dt_entropy.mean(axis=0)

# Encontrar el mejor hiperparámetro
mejor_depth_index_entropy = np.argmax(scores_promedio_dt_entropy)
mejor_depth_entropy = valores_depth_entropy[mejor_depth_index_entropy]
mejor_score_dt_entropy = scores_promedio_dt_entropy[mejor_depth_index_entropy]

print(f"Mejor Hiperparámetro (max_depth con 'entropy'): {mejor_depth_entropy}")
print(f"Mejor Accuracy Promedio (CV con 'entropy'): {mejor_score_dt_entropy * 100:.2f}%")

# Preparar DataFrame para el Boxplot
df_resultados_dt_entropy = pd.DataFrame(resultados_dt_entropy, columns=valores_depth_entropy)
df_melted_dt_entropy = df_resultados_dt_entropy.melt(
    var_name='max_depth (Profundidad)', 
    value_name='Accuracy'
)

# Generar Boxplot de DT (entropy)
plt.figure(figsize=(12, 7))
sns.boxplot(data=df_melted_dt_entropy, x='max_depth (Profundidad)', y='Accuracy', palette='plasma')
plt.title("Rendimiento de 'entropy' vs. Profundidad Máxima (5-fold CV)", fontsize=16, weight='bold')
plt.xlabel('Profundidad Máxima (max_depth)', fontsize=12)
plt.ylabel('Accuracy del Fold', fontsize=12)
plt.tight_layout()
plt.show()

