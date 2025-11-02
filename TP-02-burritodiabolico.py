import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#%%

sns.set_theme(style="whitegrid")

df_full = pd.read_csv(r"C:\Users\andre\OneDrive\Documents\TP2-Labo-de-datos\kuzushiji_full.csv", header = 0) #cada uno ponga su direcccion hasta q podamos hacer lo del directorio relativo

#%%
# TP2 - PARTE 1: ANÁLISIS EXPLORATORIO DE DATOS 


# 2. Separar datos (X) y etiquetas (y)
y_data = df_full.iloc[:, 784]
X_data_raw = df_full.iloc[:,:-1] 

# 3. Limpieza y Normalización
X_data_numeric = X_data_raw.apply(pd.to_numeric, errors='coerce').fillna(0)
# Normalizar los píxeles (escalar de 0-255 a 0-1)
X_data = X_data_numeric / 255.0

# 4. (AED) Información Básica (Para el informe)
cantidad_datos = X_data.shape[0]
cantidad_atributos = X_data.shape[1]
tipo_atributo = X_data.iloc[0].dtype
clases_unicas = sorted(y_data.unique())
cantidad_clases = len(clases_unicas)

# 5. (AED) Gráfico de Balanceo de Clases

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
plt.savefig('grafico_AED_balanceo_clases.png')
plt.show()
print("Gráfico 'grafico_AED_balanceo_clases.png' guardado.")


#%%
# (AED) Gráfico para Pregunta 1.a (Atributos Relevantes)
print("Generando Gráfico 1.a (Atributos Relevantes)...")
plt.figure(figsize=(6, 6))
average_image = X_data.mean().to_numpy().reshape(28, 28)

plt.imshow(average_image, cmap='hot') 
plt.title('1.a: Imagen Promedio (Atributos Relevantes)')
plt.colorbar() 
plt.axis('off')
plt.savefig('grafico_1a_atributos_relevantes.png')
plt.show() 
#%%
# (AED) Gráfico para Pregunta 1.b (Similitud entre Clases)
print("Generando Gráfico 1.b (Similitud entre Clases)...")
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
# (AED) Gráfico para Pregunta 1.c (Similitud Intra-Clase)
print("Generando Gráfico 1.c (Similitud Intra-Clase)...")
CLASE_A_INSPECCIONAR = 8 #

X_clase_8 = X_data[y_data == CLASE_A_INSPECCIONAR]

# Seleccionar 16 índices aleatorios DE ESA CLASE
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
