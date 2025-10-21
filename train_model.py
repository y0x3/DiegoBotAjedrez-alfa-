import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

# Cargar dataset
df = pd.read_csv("dataset_diegobot.csv")

# Convertir FEN a vector simple (solo ejemplo básico)
X = df['fen'].apply(lambda x: [ord(c) for c in x]).tolist()
y = df['move'].astype('category').cat.codes.tolist()  # Jugadas a números

# Padding para que todos tengan la misma longitud
max_len = max(len(xi) for xi in X)
X = [xi + [0]*(max_len - len(xi)) for xi in X]

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Split entrenamiento/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Definir modelo simple
class ChessNet(nn.Module):
    def __init__(self, input_size, num_moves):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_moves)
        )
    def forward(self, x):
        return self.model(x)

input_size = X_train.shape[1]
num_moves = len(df['move'].astype('category').cat.categories)
model = ChessNet(input_size, num_moves)

# Entrenamiento
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):  # Puedes aumentar después
    total_loss = 0
    for i in range(0, len(X_train), 64):
        xb = X_train[i:i+64]
        yb = y_train[i:i+64]

        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: loss = {total_loss/len(X_train):.4f}")

# Guardar modelo entrenado
torch.save(model.state_dict(), "diegobot_model.pth")
print("✅ Modelo guardado como diegobot_model.pth")
