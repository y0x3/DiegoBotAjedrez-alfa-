import chess
import torch
import pandas as pd
import torch.nn as nn

# ======= Cargar dataset para mapear índices a jugadas =======
df = pd.read_csv("dataset_diegobot.csv")
moves = df['move'].astype('category').cat.categories

# ======= Definir el modelo =======
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

# ======= Cargar pesos entrenados =======
sample_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
X_input_sample = [ord(c) for c in sample_fen]
max_len = max(len(X_input_sample), max(df['fen'].apply(len)))
X_input_sample += [0]*(max_len - len(X_input_sample))

input_size = len(X_input_sample)
num_moves = len(moves)

model = ChessNet(input_size, num_moves)
model.load_state_dict(torch.load("diegobot_model.pth"))
model.eval()

# ======= Función para predecir la jugada del bot =======
def predecir_jugada(fen):
    x = [ord(c) for c in fen]
    x += [0]*(max_len - len(x))
    x_tensor = torch.tensor([x], dtype=torch.float32)
    output = model(x_tensor)
    idx = torch.argmax(output).item()
    return moves[idx]

# ======= Función para mostrar tablero con coordenadas =======
def mostrar_tablero(board):
    print("  a b c d e f g h")  # Columnas
    for i in range(8, 0, -1):
        fila = str(i) + " "  # Número de fila
        for j in range(8):
            square = chess.square(j, i-1)
            piece = board.piece_at(square)
            fila += (piece.symbol() if piece else ".") + " "
        print(fila)
    print("  a b c d e f g h")  # Columnas de nuevo

# ======= Iniciar el tablero =======
board = chess.Board()

print("=== Juego de ajedrez contra tu bot ===")
print("Introduce tus jugadas en formato UCI, ej: e2e4")

while not board.is_game_over():
    print("\nTablero actual:")
    mostrar_tablero(board)

    # Tu turno
    while True:
        tu_jugada = input("Tu jugada: ")
        try:
            board.push_uci(tu_jugada)
            break
        except:
            print("Movimiento inválido, intenta de nuevo.")

    if board.is_game_over():
        break

    # Turno del bot
    fen = board.fen()
    bot_move = predecir_jugada(fen)
    print("Bot juega:", bot_move)
    board.push_uci(bot_move)

# ======= Final del juego =======
print("\n=== Juego terminado ===")
print("Resultado:", board.result())
mostrar_tablero(board)
