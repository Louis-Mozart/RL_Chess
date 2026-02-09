# RL Chess - Learning AI Chess Game

An intelligent chess game where the AI learns and improves through Reinforcement Learning. The more you play, the stronger the AI becomes! Each user has their own AI that adapts to their playing style.

## Features

**Adaptive AI**: The AI learns from every game and gets progressively better
**Progressive Difficulty**: Win 3 games in a row, and the AI levels up (up to level 10)
**User Profiles**: Each player has their own profile with a personalized AI opponent
📊 **Statistics Tracking**: Track your wins, losses, draws, and progress
💾 **Data Persistence**: All progress is saved locally per user
🎮 **Clean GUI**: Simple and intuitive Pygame interface

## How It Works

### Learning System
- The AI uses a neural network to evaluate chess positions
- After each game, the AI trains on the moves from that game
- The AI's model is saved per user, so it remembers what it learned
- Each user's AI starts at level 1 (beginner) and can reach level 10 (advanced)

### Difficulty Progression
- **Level 1-3**: Beginner - Makes occasional mistakes, plays randomly sometimes
- **Level 4-6**: Intermediate - More consistent, fewer mistakes
- **Level 7-9**: Advanced - Strong tactical play
- **Level 10**: Expert - Maximum difficulty

### Leveling Up
- Win 3 games in a row → AI increases by 1 level
- Lose a game → Win streak resets (AI level stays the same)
- Draw → Win streak resets

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. **Clone or download this repository**

2. **Install required packages:**
```bash
pip install -r requirements.txt
```

The main dependencies are:
- `pygame` - For the GUI
- `python-chess` - For chess logic
- `torch` - For the neural network
- `numpy` - For numerical operations

## Usage

### Running the Game

```bash
python main.py
```

### First Time Playing

1. **Main Menu** - When you start the game, you'll see:
   - `N` - Create a new user
   - `S` - Select an existing user
   - `L` - List all users
   - `Q` - Quit

2. **Create a User**:
   - Press `N`
   - Enter a username (letters, numbers, underscore)
   - Press `ENTER`

3. **Start Playing**:
   - Click on your piece to select it
   - Click on a highlighted square to move
   - The AI (playing black) will automatically respond
   - The game follows standard chess rules

### In-Game Controls

- **Mouse Click**: Select and move pieces
- **ESC**: Return to main menu
- **SPACE** (after game ends): Start a new game

### Game Interface

- **Top**: Check indicator (if in check)
- **Bottom Bar**: 
  - Current turn
  - AI level
  - Your win/loss/draw record
  - Consecutive wins counter

## User Management

### Multiple Users
- Each user has their own:
  - Statistics (games played, won, lost, drawn)
  - AI difficulty level
  - Trained an AI model
  - Win streak progress

### Selecting Users
Press `S` in the main menu and enter an existing username to continue your progress.

### Listing Users
Press `L` in the main menu to see all registered users.

### Deleting User Data
To delete a user's data, you can manually remove:
- Their entry from `user_data/users.json`
- Their model file from `models/<username>_model.pth`

Or use Python:
```python
from user_manager import UserManager
um = UserManager()
um.delete_user("username")
```

## File Structure

```
RL_Chess/
├── main.py              # Main GUI and game loop
├── chess_game.py        # Chess game logic
├── chess_ai.py          # RL AI implementation
├── user_manager.py      # User profile management
├── config.py            # Configuration settings
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── user_data/          # User profiles (auto-created)
│   └── users.json
└── models/             # Saved AI models (auto-created)
    └── <username>_model.pth
```

## Configuration

You can modify settings in `config.py`:

### Display Settings
- `WINDOW_WIDTH`, `WINDOW_HEIGHT` - Window size
- Color schemes for the board
- `FPS` - Frame rate

### AI Settings
- `LEARNING_RATE` - How fast the AI learns
- `GAMMA` - Discount factor for future rewards
- `EPSILON_START/MIN/DECAY` - Exploration vs exploitation
- `GAMES_PER_LEVEL` - Wins needed to level up (default: 3)
- `MAX_AI_LEVEL` - Maximum difficulty level (default: 10)

## Technical Details

### Neural Network Architecture
- **Input**: 12-channel 8x8 board representation (6 piece types × 2 colors)
- **Layers**: 3 convolutional layers + 3 fully connected layers
- **Output**: Single value (position evaluation)

### Training Process
1. After each game, positions are stored with rewards
2. Rewards are assigned based on game outcome (win/loss/draw)
3. Neural network trains on these experiences using replay memory
4. Model weights are saved per user

### Learning Algorithm
- Uses Deep Q-Learning principles
- Experience replay for stable learning
- Epsilon-greedy exploration strategy
- Target network for stable training

## Tips for Playing

1. **Early Games**: The AI starts weak - expect easy wins at level 1
2. **Be Strategic**: The AI learns from patterns, so vary your strategies
3. **Patience**: The AI improves gradually - you'll notice it getting stronger
4. **Level 10**: This is the maximum difficulty - a true challenge!
5. **Reset Option**: Create a new user to start fresh with a beginner AI

## Troubleshooting

### Installation Issues

**PyTorch won't install:**
```bash
# For CPU-only version (smaller download)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**pygame issues on Linux:**
```bash
sudo apt-get install python3-pygame
```

### Runtime Issues

**"No module named X":**
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

**AI is too strong/weak:**
Adjust `GAMES_PER_LEVEL` in `config.py` or create a new user to reset.

**Game freezes:**
The AI may take a few seconds to think, especially at higher levels.

## Future Enhancements

Possible improvements:
- [ ] Opening book for better early game
- [ ] Multiple difficulty presets
- [ ] Online leaderboards
- [ ] Game replay system
- [ ] Hint system
- [ ] Move time limits
- [ ] Different AI training algorithms
- [ ] Piece movement animations

## License

This project is open source and available for educational purposes.

## Credits

Built with:
- [python-chess](https://github.com/niklasf/python-chess) - Chess logic
- [PyTorch](https://pytorch.org/) - Neural networks
- [Pygame](https://www.pygame.org/) - Graphics and UI

## Contributing

Feel free to fork and improve! Some areas that could use work:
- Better piece graphics
- More sophisticated AI evaluation
- Performance optimizations
- UI/UX improvements

---

**Enjoy the game and watch your AI opponent grow stronger!** 🎮♟️
