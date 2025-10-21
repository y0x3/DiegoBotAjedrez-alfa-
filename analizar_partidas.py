import chess.pgn
import pandas as pd
import os

mi_nombre = "D3spreci0"  # <-- tu usuario exacto
data = []
contador = 0

# Carpeta donde están todos los archivos .pgn
carpeta = "partidas"

# Recorre todos los archivos en la carpeta
for archivo in os.listdir(carpeta):
    if archivo.endswith(".pgn"):
        path = os.path.join(carpeta, archivo)
        print(f"Procesando {archivo}...")
        with open(path, encoding="utf-8") as pgn:
            while True:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break
                contador += 1

                white = game.headers.get("White", "")
                black = game.headers.get("Black", "")

                # Determinar si tú jugaste con blancas o negras
                if mi_nombre.lower() in white.lower():
                    color = "white"
                elif mi_nombre.lower() in black.lower():
                    color = "black"
                else:
                    continue  # si no aparece tu nombre, se salta

                board = game.board()

                # Recorre todas las jugadas principales
                for move in game.mainline_moves():
                    turno = "white" if board.turn else "black"

                    # Solo guardamos tus jugadas (no las del rival)
                    if turno == color:
                        data.append({"fen": board.fen(), "move": move.uci(), "game_id": contador})

                    board.push(move)

# Guardar todo en un CSV para el siguiente paso
df = pd.DataFrame(data)
df.to_csv("dataset_diegobot.csv", index=False)

print(f"✅ Se procesaron {contador} partidas.")
print(f"✅ Se guardaron {len(df)} jugadas de tu estilo en dataset_diegobot.csv")
