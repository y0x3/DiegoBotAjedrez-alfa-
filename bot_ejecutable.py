import torch
import pandas as pd
import torch.nn as nn

# Cargar dataset para mapear índices a jugadas
df = pd.read_csv("dataset_diegobot.csv")
moves = df['move'].astype('category').cat.categories

# Definir la misma arquitectura del modelo
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

# Preparar un FEN de ejemplo (puedes cambiarlo)
fen_ejemplo = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3"

# Convertir FEN a vector como en el entrenamiento
X_input = [ord(c) for c in fen_ejemplo]
max_len = max(len(X_input), max(df['fen'].apply(len)))  # Para que coincida con entrenamiento
X_input += [0]*(max_len - len(X_input))
X_input = torch.tensor([X_input], dtype=torch.float32)

# Cargar modelo
input_size = len(X_input[0])
num_moves = len(moves)
model = ChessNet(input_size, num_moves)
model.load_state_dict(torch.load("diegobot_model.pth"))
model.eval()

# Predecir jugada
output = model(X_input)
pred_idx = torch.argmax(output).item()
predicted_move = moves[pred_idx]

print("✅ Predicción del bot para la posición FEN:")
print(fen_ejemplo)
print("Jugada sugerida:", predicted_move)
