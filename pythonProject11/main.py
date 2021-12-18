# Ulaş DEMİR
# https://github.com/ulasdemr
# Lojik kapıların Perceptron öğrenme algoritmasıyla modellenmesi

# credits for the Perceptron class: Aashir Javed
# Available: github.com/aashirjaved
# Perceptron-Machine-Learning-Using-Python-
# File: Perceptron.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

url = 'https://raw.githubusercontent.com/irhallac/SisLab/main/hw_data/data_0123.csv'
res = requests.get(url, allow_redirects=True)
with open('data_0123.csv','wb') as file:
    file.write(res.content)
df = pd.read_csv('data_0123.csv')

print(df)

cikis = df.iloc[0:100, 4].values
cikis = np.where(cikis == 'Iris-setosa', -1, 1)
giris = df.iloc[0:100, [0, 2]].values
plt.title('2D görünüm', fontsize=16)

plt.scatter(giris[:50, 0], giris[:50, 1], color='black', marker='o', label='setosa')
plt.scatter(giris[50:100, 0], giris[50:100, -1], color='green', marker='x', label='versicolor')
plt.xlabel('sapel length')
plt.ylabel('petal length')
plt.legend(loc='upper left')

plt.show()

class Perceptron(object):
    def __init__(self, ogrenme_orani=0.1, iter_sayisi=10):
        self.ogrenme_orani = ogrenme_orani
        self.iter_sayisi = iter_sayisi

    def ogren(self, X, y):
        self.w = np.zeros(1 + X.shape[1])

        self.hatalar = []
        for _ in range(self.iter_sayisi):
            hata = 0
            for xi, hedef in zip(X, y):
                degisim = self.ogrenme_orani * (hedef - self.tahmin(xi))
                self.w[1:] += degisim * xi
                self.w[0] += degisim
                hata += int(degisim != 0.0)
            self.hatalar.append(hata)
        return self

    def net_input(self, X):
        return np.dot(X, self.w[1:]) + self.w[0]

    def tahmin(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, -1)


siniflandirici = Perceptron(ogrenme_orani=0.1, iter_sayisi=10)
siniflandirici.ogren(giris, cikis)
siniflandirici.w
siniflandirici.hatalar