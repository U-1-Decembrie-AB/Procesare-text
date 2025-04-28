# Instructiuni de rulare in VS Code dupa deschidere terminal CTRL + SHIFT + `

# Creare mediu virtual
# python -m venv .venv

# Activare mediu
# ..venv\Scripts\Activate.ps1

# Actualizează pip 
# pip install --upgrade pip

# Instalează dependențele
# pip install tensorflow numpy matplotlib scikit-learn seaborn

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns  # pentru harta de caldura a matricei de confuzie mai frumoasa

# 1. Hiperparametri
VOCAB_SIZE = 20000  # dimensiunea vocabularului
MAXLEN = 200       # lungimea maxima a secventelor
EMB_DIM = 128      # dimensiunea embedding-ului
BATCH_SIZE = 64    # dimensiunea batch-ului

EPOCHS = 10        # numarul de epoci
LEARNING_RATE = 1e-3  # rata de invatare
MODEL_PATH = 'best_model.h5'  # calea pentru salvarea modelului

# 2. Incarcarea setului de date IMDB
print("Se incarca datele...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)

# 3. Impartirea setului de antrenare in antrenare + validare
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42
)

# 4. Preprocesare: completarea secventelor
print("Se preproceseaza datele...")
x_train = pad_sequences(x_train, maxlen=MAXLEN, padding='post', truncating='post')
x_val   = pad_sequences(x_val,   maxlen=MAXLEN, padding='post', truncating='post')
x_test  = pad_sequences(x_test,  maxlen=MAXLEN, padding='post', truncating='post')

# 5. Construirea modelului (LSTM bidirectional)
print("Se construiește modelul...")
model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMB_DIM, input_length=MAXLEN),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# 6. Compilare
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=LEARNING_RATE),
    metrics=['accuracy']
)
model.summary()

# 7. Antrenare cu checkpoint pentru salvarea celui mai bun model
checkpoint = ModelCheckpoint(
    filepath=MODEL_PATH,
    save_best_only=True,
    monitor='val_loss',
    verbose=1
)
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint]
)

# 8. Incarcarea celui mai bun model & evaluare
model.load_weights(MODEL_PATH)
print("Se evalueaza pe datele de test...")
loss, acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

# 9. Predictii & metrici
y_pred_prob = model.predict(x_test, batch_size=BATCH_SIZE)

y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)

print("Raport de clasificare:")
print(classification_report(y_test, y_pred, digits=4))

# Matricea de confuzie
tcm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(tcm, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice de confuzie')
plt.xlabel('Predictii')
plt.ylabel('Adevarat')
plt.show()

# 10. Plotarea istoricului de antrenament
plt.figure()
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss vs Epoci')
plt.xlabel('Epoca')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Acuratete vs Epoci')
plt.xlabel('Epoca')
plt.ylabel('Acuratete')
plt.legend()
plt.show()
