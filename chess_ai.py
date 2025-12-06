"""
Reinforcement Learning Chess AI Agent
Uses a neural network to evaluate positions and learn from games
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import chess
import random
import os
from collections import deque
from config import (LEARNING_RATE, GAMMA, EPSILON_START, EPSILON_MIN, 
                   EPSILON_DECAY, BATCH_SIZE, MEMORY_SIZE, MODELS_DIR)


class ChessNet(nn.Module):
    """Neural network for chess position evaluation"""
    def __init__(self):
        super(ChessNet, self).__init__()
        # Input: 8x8 board with 12 piece types (6 for each color)
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)  # Position evaluation
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        x = x.view(-1, 128 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class ChessAI:
    """AI agent that learns and improves through reinforcement learning"""
    def __init__(self, username=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChessNet().to(self.device)
        self.target_model = ChessNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()
        
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.username = username
        
        # Position-specific memory: tracks positions that led to losses
        self.bad_positions = {}  # position_hash -> (count, avg_penalty)
        self.good_positions = {}  # position_hash -> (count, avg_reward)
        self.losing_moves = {}  # (from_square, to_square) -> count
        
        # Load model if it exists for this user
        if username:
            self.load_model(username)
        
        self.update_target_model()
    
    def get_piece_value(self, piece_type):
        """Get the standard value of a piece"""
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        return piece_values.get(piece_type, 0)
    
    def calculate_material(self, board):
        """Calculate material balance using standard piece values"""
        white_material = 0
        black_material = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.get_piece_value(piece.piece_type)
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value
        
        # Return from black's perspective (AI plays black)
        return black_material - white_material
    
    def evaluate_move_material(self, board, move):
        """Evaluate the material outcome of a move (captures, trades)"""
        moving_piece = board.piece_at(move.from_square)
        if not moving_piece:
            return 0
        
        moving_value = self.get_piece_value(moving_piece.piece_type)
        captured_piece = board.piece_at(move.to_square)
        
        # Simple capture
        if captured_piece:
            captured_value = self.get_piece_value(captured_piece.piece_type)
            # Check if this square is defended
            board.push(move)
            attackers = board.attackers(chess.WHITE, move.to_square)
            board.pop()
            
            if attackers:
                # Square is defended, this is a trade
                # Net material = what we capture - what we'll lose
                return captured_value - moving_value
            else:
                # Free capture, no trade
                return captured_value
        
        return 0
    
    def board_to_tensor(self, board):
        """Convert chess board to tensor representation"""
        # Create 12 channels: 6 piece types x 2 colors
        tensor = np.zeros((12, 8, 8), dtype=np.float32)
        
        piece_map = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                piece_type = piece_map[piece.piece_type]
                channel = piece_type if piece.color == chess.WHITE else piece_type + 6
                tensor[channel][rank][file] = 1.0
        
        return torch.FloatTensor(tensor).unsqueeze(0).to(self.device)
    
    def evaluate_position(self, board):
        """Evaluate a chess position using neural network + material balance"""
        with torch.no_grad():
            board_tensor = self.board_to_tensor(board)
            nn_value = self.model(board_tensor).item()
        
        # Calculate material advantage (from black's perspective)
        material_value = self.calculate_material(board) * 0.5  # Scale material to reasonable range
        
        # Additional positional bonuses
        positional_value = 0
        
        # Bonus for center control
        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        for sq in center_squares:
            piece = board.piece_at(sq)
            if piece:
                if piece.color == chess.BLACK:
                    positional_value += 0.1
                else:
                    positional_value -= 0.1
        
        # Penalty for king exposure (simplified king safety)
        if board.is_check():
            if board.turn == chess.BLACK:
                positional_value -= 1.0  # AI is in check, bad!
            else:
                positional_value += 0.5  # Opponent in check, good!
        
        # Checkmate detection
        if board.is_checkmate():
            if board.turn == chess.BLACK:
                return -100  # AI is checkmated, worst outcome
            else:
                return 100  # AI checkmated opponent, best outcome
        
        # Combine all evaluations
        total_value = nn_value + material_value + positional_value
        
        return total_value
    
    def get_move(self, board, use_exploration=True):
        """
        Get the best move using:
        1. Baseline chess knowledge (material, captures)
        2. Learned experience (position memory)
        3. Neural network evaluation
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        # Separate moves by material impact
        winning_captures = []  # Capture more valuable piece or free capture
        neutral_moves = []      # No capture or even trade
        bad_moves = []          # Losing trades or known bad moves
        
        for move in legal_moves:
            material_gain = self.evaluate_move_material(board, move)
            move_key = (move.from_square, move.to_square)
            known_bad = self.losing_moves.get(move_key, 0)
            
            if known_bad >= 3:
                bad_moves.append(move)
            elif material_gain > 0:  # Good capture/trade
                winning_captures.append((move, material_gain))
            elif material_gain < -1:  # Bad trade (lose more than 1 point)
                bad_moves.append(move)
            else:
                neutral_moves.append(move)
        
        # BASELINE INTELLIGENCE: Always take winning captures unless learned otherwise
        if winning_captures:
            # Sort by material gain (best captures first)
            winning_captures.sort(key=lambda x: x[1], reverse=True)
            
            # First game: always take best capture (baseline intelligence)
            # Later games: use learned evaluation too
            if len(self.memory) < 50:  # Early games - pure material strategy
                return winning_captures[0][0]
            
            # Evaluate top captures with learned knowledge
            best_move = None
            best_score = float('-inf')
            
            for move, material_gain in winning_captures[:3]:  # Check top 3 captures
                board.push(move)
                pos_hash = board.fen().split()[0]
                
                # Start with material advantage
                score = material_gain * 2.0
                
                # Add learned adjustments
                if pos_hash in self.bad_positions:
                    count, penalty = self.bad_positions[pos_hash]
                    score -= penalty * (count / 5.0)
                if pos_hash in self.good_positions:
                    count, reward = self.good_positions[pos_hash]
                    score += reward * (count / 5.0)
                
                # Add neural network opinion (but material weighs more)
                nn_value = -self.evaluate_position(board)
                score += nn_value * 0.3
                
                board.pop()
                
                if score > best_score:
                    best_score = score
                    best_move = move
            
            return best_move if best_move else winning_captures[0][0]
        
        # No winning captures available - need to evaluate other moves
        
        # AVOID BAD MOVES: Never make moves we've learned are terrible
        if neutral_moves or (not neutral_moves and not bad_moves):
            candidate_moves = neutral_moves if neutral_moves else legal_moves
        else:
            # Only bad moves available - forced to choose one
            candidate_moves = bad_moves
        
        # Exploration: occasionally try random moves (but not terrible ones)
        if use_exploration and random.random() < self.epsilon and len(self.memory) < 100:
            safe_moves = [m for m in candidate_moves if self.losing_moves.get((m.from_square, m.to_square), 0) < 2]
            return random.choice(safe_moves if safe_moves else candidate_moves)
        
        # LEARNED EVALUATION: Use neural network + memory for non-capture moves
        best_move = None
        best_value = float('-inf')
        
        for move in candidate_moves:
            board.push(move)
            
            # Neural network evaluation
            nn_value = -self.evaluate_position(board)
            
            # Position memory
            pos_hash = board.fen().split()[0]
            memory_adjustment = 0
            
            if pos_hash in self.bad_positions:
                count, penalty = self.bad_positions[pos_hash]
                memory_adjustment -= penalty * (count / 10.0)
            
            if pos_hash in self.good_positions:
                count, reward = self.good_positions[pos_hash]
                memory_adjustment += reward * (count / 10.0)
            
            # Move memory penalty
            move_key = (move.from_square, move.to_square)
            if move_key in self.losing_moves:
                memory_adjustment -= self.losing_moves[move_key] * 0.5
            
            # Combined score
            value = nn_value + memory_adjustment
            
            board.pop()
            
            if value > best_value:
                best_value = value
                best_move = move
        
        return best_move if best_move else candidate_moves[0]
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train the model using experiences from memory"""
        if len(self.memory) < BATCH_SIZE:
            return
        
        batch = random.sample(self.memory, BATCH_SIZE)
        total_loss = 0
        
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                with torch.no_grad():
                    next_value = self.target_model(next_state).item()
                target = reward + GAMMA * next_value
            
            current_value = self.model(state)
            target_tensor = torch.FloatTensor([[target]]).to(self.device)
            
            loss = self.criterion(current_value, target_tensor)
            total_loss += loss.item()
            
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
        
        # Decay epsilon more gradually
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY
    
    def update_target_model(self):
        """Copy weights from model to target model"""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def save_model(self, username):
        """Save the model for a specific user"""
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
        
        model_path = os.path.join(MODELS_DIR, f"{username}_model.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'bad_positions': self.bad_positions,
            'good_positions': self.good_positions,
            'losing_moves': self.losing_moves
        }, model_path)
    
    def load_model(self, username):
        """Load the model for a specific user"""
        model_path = os.path.join(MODELS_DIR, f"{username}_model.pth")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', EPSILON_START)
            self.bad_positions = checkpoint.get('bad_positions', {})
            self.good_positions = checkpoint.get('good_positions', {})
            self.losing_moves = checkpoint.get('losing_moves', {})
            self.update_target_model()
    
    def train_from_game(self, moves, result):
        """
        Train the AI from a completed game
        result: 1 for AI win, 0 for draw, -1 for AI loss
        """
        if not moves:
            return
            
        board = chess.Board()
        states = []
        ai_moves = []  # Track AI's actual moves
        material_changes = []  # Track material gained/lost
        
        prev_material = self.calculate_material(board)
        
        # Collect all positions from the game
        for move in moves:
            state = self.board_to_tensor(board)
            pos_hash = board.fen().split()[0]
            
            board.push(move)
            curr_material = self.calculate_material(board)
            material_change = curr_material - prev_material
            
            states.append((state, board.turn == chess.WHITE, pos_hash, move, material_change))  # Note: turn before move
            
            # Track AI's moves
            if board.turn == chess.BLACK:  # After white's move
                ai_moves.append((move, pos_hash))
            
            prev_material = curr_material
        
        # Record position outcomes in memory
        for i, (state, was_whites_turn, pos_hash, move, material_change) in enumerate(states):
            if not was_whites_turn:  # AI's turn (black)
                progress = (i + 1) / len(states)
                time_weight = 0.3 + 0.7 * progress
                
                # Base reward on game result
                if result == 1:  # AI won
                    base_reward = time_weight * 2.0
                elif result == -1:  # AI lost
                    base_reward = -time_weight * 2.0
                else:  # Draw
                    base_reward = time_weight * 0.5
                
                # Add material-based immediate feedback
                # If AI gained material with this move, that's good
                # If AI lost material, that's bad
                material_reward = material_change * 0.3  # Scale material change
                
                reward = base_reward + material_reward
                
                # Record position memories
                if result == 1:  # AI won
                    if pos_hash in self.good_positions:
                        count, avg_reward = self.good_positions[pos_hash]
                        self.good_positions[pos_hash] = (count + 1, (avg_reward * count + reward) / (count + 1))
                    else:
                        self.good_positions[pos_hash] = (1, reward)
                        
                elif result == -1:  # AI lost
                    if pos_hash in self.bad_positions:
                        count, avg_penalty = self.bad_positions[pos_hash]
                        self.bad_positions[pos_hash] = (count + 1, (avg_penalty * count + abs(reward)) / (count + 1))
                    else:
                        self.bad_positions[pos_hash] = (1, abs(reward))
                    
                    # Mark this specific move as losing (especially if it lost material)
                    move_key = (move.from_square, move.to_square)
                    penalty_weight = 2 if material_change < -1 else 1  # Worse if lost material
                    self.losing_moves[move_key] = self.losing_moves.get(move_key, 0) + penalty_weight
                
                next_state = states[i + 1][0] if i + 1 < len(states) else state
                done = (i == len(states) - 1)
                
                self.remember(state, None, reward, next_state, done)
        
        # Train intensively on the experiences
        from config import TRAINING_ITERATIONS
        training_rounds = TRAINING_ITERATIONS if len(self.memory) >= BATCH_SIZE else 3
        
        for _ in range(training_rounds):
            self.replay()
        
        # Update target network periodically
        if len(self.memory) % 100 == 0:
            self.update_target_model()
        
        # Prune old memories to keep most relevant
        if len(self.bad_positions) > 1000:
            # Keep only positions seen multiple times
            self.bad_positions = {k: v for k, v in self.bad_positions.items() if v[0] > 1}
        if len(self.good_positions) > 1000:
            self.good_positions = {k: v for k, v in self.good_positions.items() if v[0] > 1}
        if len(self.losing_moves) > 500:
            self.losing_moves = {k: v for k, v in self.losing_moves.items() if v > 1}
        
        # Save the updated model after each game
        if self.username:
            self.save_model(self.username)
