import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

'''X = pd.read_csv(r"Datos/kuzushiji_full.csv")

img = np.array(X.iloc[12]).reshape((28,28))
plt.imshow(img, cmap='gray')
plt.show()'''

#%%

sns.set_theme(style="whitegrid")

df_full = pd.read_csv(r"C:/Users/Owen/Downloads/kuzushiji_full.csv", header = 0) #cada uno ponga su direcccion hasta q podamos hacer lo del directorio relativo

#%%
# TP2 - PARTE 1: ANÁLISIS EXPLORATORIO DE DATOS 

# 2. Separar datos (X) y etiquetas (y)
# y_data es la ultima columna (clase)
y_data = df_full.iloc[:, 784]
# X_data_raw son los píxeles (columnas 0 en adelante menos la ult)
X_data_raw = df_full.iloc[:,:-1] 


X_data_numeric = X_data_raw.apply(pd.to_numeric, errors='coerce').fillna(0)
# Normalizar los píxeles (escalar de 0-255 a 0-1)
X_data = X_data_numeric / 255.0




cantidad_datos = X_data.shape[0]


cantidad_atributos = X_data.shape[1]


tipo_atributo = X_data.iloc[0].dtype


clases_unicas = sorted(y_data.unique())
cantidad_clases = len(clases_unicas)
print(f"4. Cantidad de clases de la variable de interés: {cantidad_clases}")
print(f"   Clases: {clases_unicas}")
print("\n")



plt.figure(figsize=(12, 7))

ax_balance = sns.countplot(
    x=y_data.astype(str), 
    order=[str(i) for i in range(10)], 
    palette='viridis'
)
ax_balance.set_title(f'Distribución de Clases)', fontsize=16, weight='bold')
ax_balance.set_xlabel('Clase (Etiqueta Numérica)', fontsize=12)
ax_balance.set_ylabel('Cantidad de Imágenes', fontsize=12)


for p in ax_balance.patches:
    ax_balance.annotate(f'{p.get_hora()}', 
                        (p.get_x() + p.get_width() / 2., p.get_hora()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points')
                        
plt.tight_layout()