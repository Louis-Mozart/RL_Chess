"""
Configuration file for RL Chess Game
"""

# Window settings
WINDOW_WIDTH = 800
BOARD_SIZE = 800
WINDOW_HEIGHT = BOARD_SIZE + 100  # Extra space for info panel
SQUARE_SIZE = BOARD_SIZE // 8
FPS = 60

# Colors (RGB)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
HIGHLIGHT_COLOR = (186, 202, 68)
SELECTED_COLOR = (246, 246, 130)

# RL Agent settings
LEARNING_RATE = 0.01  # Increased for faster learning
GAMMA = 0.95  # Discount factor
EPSILON_START = 0.95  # Initial exploration rate
EPSILON_MIN = 0.05  # Lower minimum for better play
EPSILON_DECAY = 0.98  # Slower decay

# Experience replay
BATCH_SIZE = 32  # Smaller batches for more frequent updates
MEMORY_SIZE = 10000

# Training
TRAINING_ITERATIONS = 10  # Number of training passes per game

# Training
GAMES_PER_LEVEL = 1  # Number of wins before AI improves
MAX_AI_LEVEL = 10

# File paths
USER_DATA_DIR = "user_data"
MODELS_DIR = "models"
