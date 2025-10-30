import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

X = pd.read_csv(r"Datos/kuzushiji_full.csv")

img = np.array(X.iloc[12]).reshape((28,28))
plt.imshow(img, cmap='gray')
plt.show()