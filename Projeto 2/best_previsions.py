from model_tools import Model, load_csv
import csv
import numpy as np

X_test = load_csv('dados/Concreto - teste.csv')

model = Model(4)

with open('previsoes/best_model_predictions.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    predictions = model.predict(X_test)
    for pred in predictions:
        writer.writerow([pred])

print("Previsões salvas em 'previsoes/best_model_predictions.csv'")
print("Erro quadrático médio de treino:", model.mse)
