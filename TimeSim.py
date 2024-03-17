#Tempi Sul Giro ML#
import numpy as np
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#caricamento dati 
data = pd.read_csv("TempiSulGiro.csv")

#conversione tempo e divisione celle
data["Tempo sul Giro (min:sec)"] = data["Tempo sul Giro (min:sec)"].apply(lambda x: float(x.split(':')[0]) * 60 + float(x.split(':')[1]))
X = data.drop(["Tracciato","Tempo sul Giro (min:sec)"],axis=1)
y = data["Tempo sul Giro (min:sec)"]

#divisione training e set
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=0)

#modello regressione lineare 
model = LinearRegression()

#addestramento
model.fit(X_train, y_train)

#previsioni
y_pred = model.predict(X_test)

# utilizzo del modello addestrato per fare previsioni sul tempo sul giro
nuove_features = np.array([[11, 4932, 340, 136, 42, 6, 4, 7, 2, 1350, 0.0019]]) 
tempo_predetto = model.predict(nuove_features)

# Formatta il tempo
minuti = int(tempo_predetto.item() // 60)
secondi = int(tempo_predetto.item() % 60)
millesimi = int((tempo_predetto.item() % 1) * 1000)
tempo_predetto_formattato = f"{minuti:02d}:{secondi:02d}.{millesimi:03d}"

print("Tempo predetto sul giro:", tempo_predetto_formattato)