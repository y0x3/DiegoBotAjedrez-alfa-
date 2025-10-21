import pygame
import chess
import torch
import pandas as pd
import torch.nn as nn
import os
import chess.engine

# ======= Config =======
pygame.init()

# Obtener tamaño completo de pantalla
info = pygame.display.Info()
ANCHO_PANTALLA, ALTO_PANTALLA = info.current_w, info.current_h

# El tablero debe ser cuadrado, tomamos el lado más pequeño
LADO_TABLERO = min(ANCHO_PANTALLA, ALTO_PANTALLA - 40)  # -40 para barra superior visual
TAM_CASILLA = LADO_TABLERO // 8

# Dimensiones del tablero (multiplo de 8)
ANCHO = ALTURA = TAM_CASILLA * 8
FPS = 30

# Crear ventana en modo pantalla completa (esto no se debe reescribir después)
screen = pygame.display.set_mode((ANCHO_PANTALLA, ALTO_PANTALLA), pygame.FULLSCREEN)
pygame.display.set_caption("Mini Bot de Ajedrez")
clock = pygame.time.Clock()

# ======= Cargar dataset y modelo =======
df = pd.read_csv("dataset_diegobot.csv")
moves = df['move'].astype('category').cat.categories

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

# Preparar input_size y max_len (igual que antes)
sample_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
X_input_sample = [ord(c) for c in sample_fen]
max_len = max(len(X_input_sample), max(df['fen'].apply(len)))
X_input_sample += [0]*(max_len - len(X_input_sample))

input_size = len(X_input_sample)
num_moves = len(moves)
model = ChessNet(input_size, num_moves)
model.load_state_dict(torch.load("diegobot_model.pth"))
model.eval()

# ======= Inicializar Stockfish =======
engine = chess.engine.SimpleEngine.popen_uci(r"stockfish\stockfish-windows-x86-64")

# ======= Ajustar dificultad de Stockfish =======
engine.configure({"Skill Level": 5})
engine.configure({
    "UCI_LimitStrength": True,
    "UCI_Elo": 1520
})

# ======= Cargar imágenes de piezas =======
IMAGENES = {}
path = "imagenes"  # tu carpeta con wP.png, bK.png, etc.
for archivo in os.listdir(path):
    nombre = archivo.split('.')[0]  # wP, bK, etc.
    img = pygame.image.load(os.path.join(path, archivo)).convert_alpha()
    img = pygame.transform.scale(img, (TAM_CASILLA, TAM_CASILLA))
    IMAGENES[nombre] = img

# ======= Inicializar tablero =======
board = chess.Board()
jugando_blancas = True
seleccion = None  # Casilla seleccionada

# ======= Funciones =======

def dibujar_tablero(seleccion=None):
    """Dibuja tablero centrado en pantalla, con barra superior para el turno."""
    # Fondo tablero colores
    colores = [pygame.Color("white"), pygame.Color("gray")]

    # Calcular posición inicial para centrar el tablero
    offset_x = (ANCHO_PANTALLA - ANCHO) // 2
    offset_y = ((ALTO_PANTALLA - ALTURA) // 2) + 20  # 20 px para dejar espacio a la barra superior

    # Dibujar barra superior (indicador de turno) a lo ancho de la pantalla
    pygame.draw.rect(screen, (220, 220, 220), (0, 0, ANCHO_PANTALLA, 40))

    for i in range(8):
        for j in range(8):
            color = colores[(i + j) % 2]
            rect = pygame.Rect(
                offset_x + j * TAM_CASILLA,
                offset_y + i * TAM_CASILLA,
                TAM_CASILLA,
                TAM_CASILLA
            )
            pygame.draw.rect(screen, color, rect)

            # Resaltar casilla seleccionada
            if seleccion is not None:
                sel_col = chess.square_file(seleccion)
                sel_row = 7 - chess.square_rank(seleccion)
                if i == sel_row and j == sel_col:
                    pygame.draw.rect(screen, pygame.Color("yellow"), rect, 5)

            # Dibujar pieza si existe
            square = chess.square(j, 7 - i)
            pieza = board.piece_at(square)
            if pieza:
                key = f"{'w' if pieza.color else 'b'}{pieza.symbol().upper()}"
                if key in IMAGENES:
                    screen.blit(IMAGENES[key], (offset_x + j * TAM_CASILLA, offset_y + i * TAM_CASILLA))
                else:
                    # fallback sencillo: dibujar letra si no hay imagen
                    fuente = pygame.font.SysFont("arial", 20, bold=True)
                    texto = fuente.render(pieza.symbol(), True, pygame.Color("black"))
                    screen.blit(texto, (offset_x + j * TAM_CASILLA + 8, offset_y + i * TAM_CASILLA + 8))

def pos_to_casilla(pos):
    """
    Convierte coordenadas del ratón a un square de chess.
    Devuelve None si el click no cae dentro del área del tablero (incluida barra superior).
    """
    x, y = pos
    offset_x = (ANCHO_PANTALLA - ANCHO) // 2
    offset_y = ((ALTO_PANTALLA - ALTURA) // 2) + 20

    # Si está por encima o fuera del tablero, devolver None
    if x < offset_x or x >= offset_x + ANCHO or y < offset_y or y >= offset_y + ALTURA:
        return None

    col = (x - offset_x) // TAM_CASILLA
    row = (y - offset_y) // TAM_CASILLA
    # Convertir fila a coordenada de chess (0 abajo → 7 arriba)
    chess_row = 7 - row
    return chess.square(int(col), int(chess_row))

def predecir_jugada_segura(board):
    """Inferencia básica usando el modelo cargado, con torch.no_grad()"""
    fen = board.fen()
    x = [ord(c) for c in fen]
    if len(x) < max_len:
        x += [0]*(max_len - len(x))
    else:
        x = x[:max_len]

    x_tensor = torch.tensor([x], dtype=torch.float32)
    with torch.no_grad():
        output = model(x_tensor).squeeze(0)  # logits
    # argmax directo (mantengo tu lógica original, pero con no_grad)
    idx = int(torch.argmax(output).item())
    jugada_pred = moves[idx]

    # Manejo seguro de UCI -> Move
    try:
        move = chess.Move.from_uci(jugada_pred)
    except Exception as e:
        # fallback a engine si falla parsing
        try:
            result = engine.play(board, chess.engine.Limit(time=0.1))
            return result.move
        except:
            return list(board.legal_moves)[0]

    if move in board.legal_moves:
        return move

    # Si falla, usar Stockfish
    try:
        result = engine.play(board, chess.engine.Limit(time=0.1))
        if result.move in board.legal_moves:
            return result.move
    except:
        pass

    # Último recurso: elegir la primera legal
    return list(board.legal_moves)[0]

def mostrar_turno():
    """Muestra en la parte superior de la pantalla de quién es el turno"""
    fuente = pygame.font.SysFont("arial", 24, bold=True)
    texto_turno = "Turno: Blancas" if jugando_blancas else "Turno: Negras"
    color = pygame.Color("black") if jugando_blancas else pygame.Color("darkred")
    texto = fuente.render(texto_turno, True, color)
    rect = texto.get_rect(center=(ANCHO_PANTALLA // 2, 20))  # centrar en pantalla completa
    screen.blit(texto, rect)

def mostrar_mensaje_final(mensaje):
    """Muestra mensaje centrado de fin de partida"""
    overlay = pygame.Surface((ANCHO_PANTALLA, ALTO_PANTALLA))
    overlay.set_alpha(200)
    overlay.fill((0, 0, 0))
    screen.blit(overlay, (0, 0))
    fuente = pygame.font.SysFont("arial", 72, bold=True)
    texto = fuente.render(mensaje, True, pygame.Color("white"))
    rect = texto.get_rect(center=(ANCHO_PANTALLA // 2, ALTO_PANTALLA // 2))
    screen.blit(texto, rect)
    pygame.display.flip()
    pygame.time.wait(4000)

# ======= Loop principal =======
running = True
juego_terminado = False

while running:
    clock.tick(FPS)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Salir con ESC de manera segura
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

        elif event.type == pygame.MOUSEBUTTONDOWN and jugando_blancas and not juego_terminado:
            square = pos_to_casilla(pygame.mouse.get_pos())
            if square is not None:
                if seleccion is None:
                    seleccion = square
                else:
                    move = chess.Move(seleccion, square)
                    if move in board.legal_moves:
                        board.push(move)
                        jugando_blancas = False
                    seleccion = None

    # Turno del bot
    if not jugando_blancas and not board.is_game_over() and not juego_terminado:
        move = predecir_jugada_segura(board)
        board.push(move)
        jugando_blancas = True

    # Detectar fin de juego
    if board.is_game_over() and not juego_terminado:
        juego_terminado = True
        resultado = board.result()
        if resultado == "1-0":
            mostrar_mensaje_final("¡Ganaste! 😎")
        elif resultado == "0-1":
            mostrar_mensaje_final("Perdiste 😢")
        else:
            mostrar_mensaje_final("Empate 🤝")

    # Dibujar tablero y turno
    screen.fill(pygame.Color("black"))
    dibujar_tablero(seleccion)
    mostrar_turno()
    pygame.display.flip()

pygame.quit()
engine.quit()
