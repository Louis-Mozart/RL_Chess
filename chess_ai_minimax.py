"""
Chess AI using Minimax search with Alpha-Beta pruning
Plus learning from game experience
"""
import chess
import random
import json
import os
from collections import defaultdict
from config import MODELS_DIR


class ChessAI:
    """AI that uses minimax search and learns position values"""
    
    def __init__(self, username=None):
        self.username = username
        
        # Piece values
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        # Piece-square tables for positional play
        self.pawn_table = [
            0,  0,  0,  0,  0,  0,  0,  0,
            50, 50, 50, 50, 50, 50, 50, 50,
            10, 10, 20, 30, 30, 20, 10, 10,
            5,  5, 10, 25, 25, 10,  5,  5,
            0,  0,  0, 20, 20,  0,  0,  0,
            5, -5,-10,  0,  0,-10, -5,  5,
            5, 10, 10,-20,-20, 10, 10,  5,
            0,  0,  0,  0,  0,  0,  0,  0
        ]
        
        self.knight_table = [
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50
        ]
        
        self.bishop_table = [
            -20,-10,-10,-10,-10,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5, 10, 10,  5,  0,-10,
            -10,  5,  5, 10, 10,  5,  5,-10,
            -10,  0, 10, 10, 10, 10,  0,-10,
            -10, 10, 10, 10, 10, 10, 10,-10,
            -10,  5,  0,  0,  0,  0,  5,-10,
            -20,-10,-10,-10,-10,-10,-10,-20
        ]
        
        self.rook_table = [
            0,  0,  0,  0,  0,  0,  0,  0,
            5, 10, 10, 10, 10, 10, 10,  5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            0,  0,  0,  5,  5,  0,  0,  0
        ]
        
        self.queen_table = [
            -20,-10,-10, -5, -5,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5,  5,  5,  5,  0,-10,
            -5,  0,  5,  5,  5,  5,  0, -5,
            0,  0,  5,  5,  5,  5,  0, -5,
            -10,  5,  5,  5,  5,  5,  0,-10,
            -10,  0,  5,  0,  0,  0,  0,-10,
            -20,-10,-10, -5, -5,-10,-10,-20
        ]
        
        self.king_table = [
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -20,-30,-30,-40,-40,-30,-30,-20,
            -10,-20,-20,-20,-20,-20,-20,-10,
            20, 20,  0,  0,  0,  0, 20, 20,
            20, 30, 10,  0,  0, 10, 30, 20
        ]
        
        # Learning components
        self.position_scores = defaultdict(lambda: {'wins': 0, 'losses': 0, 'draws': 0})
        self.opening_book = {}  # Learn good opening moves
        self.losing_moves = defaultdict(int)
        self.games_played = 0
        self.search_depth = 2  # Start at depth 2, increase as it learns
        
        # Load learned data
        if username:
            self.load_knowledge()
    
    def get_piece_square_value(self, piece, square):
        """Get positional value for a piece on a square"""
        tables = {
            chess.PAWN: self.pawn_table,
            chess.KNIGHT: self.knight_table,
            chess.BISHOP: self.bishop_table,
            chess.ROOK: self.rook_table,
            chess.QUEEN: self.queen_table,
            chess.KING: self.king_table
        }
        
        table = tables.get(piece.piece_type)
        if not table:
            return 0
        
        # Flip table for black pieces
        if piece.color == chess.BLACK:
            square = chess.square_mirror(square)
        
        return table[square]
    
    def evaluate_board(self, board):
        """Evaluate board position from black's perspective"""
        if board.is_checkmate():
            return -999999 if board.turn == chess.BLACK else 999999
        
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        score = 0
        
        # Material and position
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.piece_values[piece.piece_type]
                position_value = self.get_piece_square_value(piece, square)
                
                if piece.color == chess.BLACK:
                    score += value + position_value
                else:
                    score -= value + position_value
        
        # Mobility (number of legal moves)
        mobility = len(list(board.legal_moves))
        if board.turn == chess.BLACK:
            score += mobility * 10
        else:
            score -= mobility * 10
        
        # Check if we've learned something about this position
        pos_hash = board.fen().split()[0]
        if pos_hash in self.position_scores:
            stats = self.position_scores[pos_hash]
            total = stats['wins'] + stats['losses'] + stats['draws']
            if total > 0:
                win_rate = (stats['wins'] - stats['losses']) / total
                score += win_rate * 500  # Learned adjustment
        
        return score
    
    def minimax(self, board, depth, alpha, beta, maximizing):
        """Minimax with alpha-beta pruning"""
        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board)
        
        legal_moves = list(board.legal_moves)
        
        # Move ordering: captures first (better pruning)
        captures = [m for m in legal_moves if board.is_capture(m)]
        non_captures = [m for m in legal_moves if not board.is_capture(m)]
        ordered_moves = captures + non_captures
        
        if maximizing:
            max_eval = float('-inf')
            for move in ordered_moves:
                board.push(move)
                eval = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in ordered_moves:
                board.push(move)
                eval = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval
    
    def get_move(self, board, use_exploration=True):
        """Get best move using minimax search"""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        # Check opening book (first 10 moves)
        if board.fullmove_number <= 5 and self.opening_book:
            pos_hash = board.fen().split()[0]
            if pos_hash in self.opening_book:
                book_moves = self.opening_book[pos_hash]
                if book_moves:
                    return random.choice(book_moves)
        
        # Filter out known terrible moves
        good_moves = [m for m in legal_moves 
                     if self.losing_moves.get((m.from_square, m.to_square), 0) < 3]
        search_moves = good_moves if good_moves else legal_moves
        
        best_move = None
        best_value = float('-inf')
        
        # Search each move
        for move in search_moves:
            board.push(move)
            value = self.minimax(board, self.search_depth - 1, float('-inf'), float('inf'), False)
            board.pop()
            
            # Add small random factor for variety in equal positions
            value += random.uniform(-5, 5)
            
            if value > best_value:
                best_value = value
                best_move = move
        
        return best_move if best_move else legal_moves[0]
    
    def train_from_game(self, moves, result):
        """Learn from a completed game"""
        if not moves:
            return
        
        self.games_played += 1
        
        # Increase search depth as AI learns (up to depth 3)
        if self.games_played >= 10 and self.search_depth < 3:
            self.search_depth = 3
        
        board = chess.Board()
        
        # Record positions and outcomes
        for i, move in enumerate(moves):
            pos_hash = board.fen().split()[0]
            
            # Learn from positions
            if board.turn == chess.BLACK:
                if result == 1:  # AI won
                    self.position_scores[pos_hash]['wins'] += 1
                elif result == -1:  # AI lost
                    self.position_scores[pos_hash]['losses'] += 1
                    # Mark losing moves
                    move_key = (move.from_square, move.to_square)
                    self.losing_moves[move_key] += 1
                else:
                    self.position_scores[pos_hash]['draws'] += 1
            
            # Build opening book from winning games
            if result == 1 and board.fullmove_number <= 5 and board.turn == chess.BLACK:
                if pos_hash not in self.opening_book:
                    self.opening_book[pos_hash] = []
                if move not in self.opening_book[pos_hash]:
                    self.opening_book[pos_hash].append(move)
            
            board.push(move)
        
        # Prune old data
        if len(self.position_scores) > 5000:
            # Keep only positions seen multiple times
            self.position_scores = defaultdict(lambda: {'wins': 0, 'losses': 0, 'draws': 0},
                {k: v for k, v in self.position_scores.items() 
                 if sum(v.values()) > 1})
        
        if len(self.losing_moves) > 1000:
            self.losing_moves = defaultdict(int,
                {k: v for k, v in self.losing_moves.items() if v > 1})
        
        # Save learned knowledge
        if self.username:
            self.save_knowledge()
    
    def save_knowledge(self):
        """Save learned data to file"""
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
        
        # Convert chess.Move objects to UCI strings for JSON
        opening_book_json = {}
        for pos_hash, moves in self.opening_book.items():
            opening_book_json[pos_hash] = [move.uci() for move in moves]
        
        data = {
            'position_scores': dict(self.position_scores),
            'opening_book': opening_book_json,
            'losing_moves': {f"{k[0]},{k[1]}": v for k, v in self.losing_moves.items()},
            'games_played': self.games_played,
            'search_depth': self.search_depth
        }
        
        filepath = os.path.join(MODELS_DIR, f"{self.username}_knowledge.json")
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    def load_knowledge(self):
        """Load learned data from file"""
        filepath = os.path.join(MODELS_DIR, f"{self.username}_knowledge.json")
        if not os.path.exists(filepath):
            return
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.position_scores = defaultdict(lambda: {'wins': 0, 'losses': 0, 'draws': 0},
                                              data.get('position_scores', {}))
            
            # Convert UCI strings back to chess.Move objects
            opening_book_json = data.get('opening_book', {})
            self.opening_book = {}
            for pos_hash, uci_moves in opening_book_json.items():
                self.opening_book[pos_hash] = [chess.Move.from_uci(uci) for uci in uci_moves]
            
            losing_moves_data = data.get('losing_moves', {})
            self.losing_moves = defaultdict(int)
            for key, value in losing_moves_data.items():
                from_sq, to_sq = map(int, key.split(','))
                self.losing_moves[(from_sq, to_sq)] = value
            
            self.games_played = data.get('games_played', 0)
            self.search_depth = data.get('search_depth', 2)
        except Exception as e:
            print(f"Error loading knowledge: {e}")
    
    @property
    def epsilon(self):
        """Compatibility property for GUI display"""
        # Show learning progress based on games played
        return max(0.05, 1.0 - (self.games_played / 50.0))
    
    @property
    def memory(self):
        """Compatibility property for GUI display"""
        # Return a list-like object with length
        class FakeMemory:
            def __init__(self, size):
                self.size = size
            def __len__(self):
                return self.size
        return FakeMemory(len(self.position_scores))
    
    @property
    def bad_positions(self):
        """Compatibility property"""
        return {k: v for k, v in self.position_scores.items() if v['losses'] > v['wins']}
    
    @property
    def good_positions(self):
        """Compatibility property"""
        return {k: v for k, v in self.position_scores.items() if v['wins'] > v['losses']}
