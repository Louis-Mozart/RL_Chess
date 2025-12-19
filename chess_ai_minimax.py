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
        self.position_scores = defaultdict(lambda: {'wins': 0, 'losses': 0, 'draws': 0, 'visits': 0})
        self.move_scores = defaultdict(lambda: {'wins': 0, 'losses': 0, 'draws': 0, 'visits': 0})
        self.opening_book = defaultdict(list)  # Learn good opening moves
        self.losing_moves = defaultdict(int)
        self.games_played = 0
        self.search_depth = 3  # Start at depth 3 minimum
        self.max_depth = 5  # Can go deeper as it learns
        
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
        
        # Mobility (number of legal moves) - more important
        mobility = len(list(board.legal_moves))
        if board.turn == chess.BLACK:
            score += mobility * 15
        else:
            score -= mobility * 15
        
        # King safety - penalize exposed kings
        for color in [chess.BLACK, chess.WHITE]:
            king_square = board.king(color)
            if king_square:
                king_attackers = len(board.attackers(not color, king_square))
                king_defenders = len(board.attackers(color, king_square))
                safety_score = (king_defenders - king_attackers * 2) * 25
                if color == chess.BLACK:
                    score += safety_score
                else:
                    score -= safety_score
        
        # Control of center squares (e4, e5, d4, d5)
        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        for sq in center_squares:
            black_control = len(board.attackers(chess.BLACK, sq))
            white_control = len(board.attackers(chess.WHITE, sq))
            score += (black_control - white_control) * 15
        
        # Check if we've learned something about this position
        pos_hash = board.fen().split()[0]
        if pos_hash in self.position_scores:
            stats = self.position_scores[pos_hash]
            visits = stats['visits']
            if visits >= 3:  # Only trust positions we've seen multiple times
                win_rate = (stats['wins'] - stats['losses']) / visits
                # Much stronger learning signal that scales with confidence
                confidence = min(visits / 10.0, 1.0)
                score += win_rate * 2000 * confidence
        
        return score
    
    def order_moves(self, board, moves):
        """Order moves for better alpha-beta pruning"""
        scored_moves = []
        for move in moves:
            score = 0
            
            # Prioritize captures
            if board.is_capture(move):
                captured = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if captured and attacker:
                    # MVV-LVA: Most Valuable Victim - Least Valuable Attacker
                    score += self.piece_values[captured.piece_type] * 10 - self.piece_values[attacker.piece_type]
            
            # Prioritize checks
            board.push(move)
            if board.is_check():
                score += 5000
            board.pop()
            
            # Learn from history
            move_key = f"{move.from_square},{move.to_square}"
            if move_key in self.move_scores:
                stats = self.move_scores[move_key]
                visits = stats['visits']
                if visits >= 2:
                    win_rate = (stats['wins'] - stats['losses']) / visits
                    score += win_rate * 3000
            
            # Penalize known losing moves
            if self.losing_moves.get((move.from_square, move.to_square), 0) >= 5:
                score -= 10000
            
            scored_moves.append((score, move))
        
        scored_moves.sort(reverse=True, key=lambda x: x[0])
        return [move for score, move in scored_moves]
    
    def quiescence_search(self, board, alpha, beta, depth_left=2):
        """Search captures to avoid horizon effect"""
        stand_pat = self.evaluate_board(board)
        
        if depth_left == 0:
            return stand_pat
        
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
        
        # Only search captures
        captures = [m for m in board.legal_moves if board.is_capture(m)]
        for move in self.order_moves(board, captures)[:5]:  # Limit for performance
            board.push(move)
            score = -self.quiescence_search(board, -beta, -alpha, depth_left - 1)
            board.pop()
            
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        
        return alpha
    
    def minimax(self, board, depth, alpha, beta, maximizing):
        """Minimax with alpha-beta pruning and quiescence search"""
        if depth == 0:
            return self.quiescence_search(board, alpha, beta)
        
        if board.is_game_over():
            return self.evaluate_board(board)
        
        legal_moves = list(board.legal_moves)
        ordered_moves = self.order_moves(board, legal_moves)
        
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
        
        # Dynamic depth based on position complexity
        depth = self.search_depth
        if len(legal_moves) < 15:  # Simple position, search deeper
            depth = min(self.max_depth, depth + 1)
        
        # Check opening book (first 8 moves)
        if board.fullmove_number <= 8 and self.opening_book:
            pos_hash = board.fen().split()[0]
            if pos_hash in self.opening_book:
                book_moves = self.opening_book[pos_hash]
                # Filter to valid moves
                valid_book_moves = [m for m in book_moves if m in legal_moves]
                if valid_book_moves:
                    # Choose best from book based on learned scores
                    best_book_move = max(valid_book_moves, 
                                       key=lambda m: self.move_scores.get(f"{m.from_square},{m.to_square}", 
                                                                          {'wins': 0, 'losses': 0, 'visits': 0})['wins'])
                    if random.random() > 0.2:  # 80% use book, 20% explore
                        return best_book_move
        
        # Filter out known terrible moves
        good_moves = [m for m in legal_moves 
                     if self.losing_moves.get((m.from_square, m.to_square), 0) < 5]
        search_moves = good_moves if good_moves else legal_moves
        
        best_move = None
        best_value = float('-inf')
        
        # Search each move with move ordering
        ordered_moves = self.order_moves(board, search_moves)
        for move in ordered_moves:
            board.push(move)
            value = self.minimax(board, depth - 1, float('-inf'), float('inf'), False)
            board.pop()
            
            # Smaller random factor for more consistent play
            value += random.uniform(-2, 2)
            
            if value > best_value:
                best_value = value
                best_move = move
        
        return best_move if best_move else legal_moves[0]
    
    def train_from_game(self, moves, result):
        """Learn from a completed game - improved learning mechanism"""
        if not moves:
            return
        
        self.games_played += 1
        
        # Gradually increase search depth as AI learns
        if self.games_played >= 20 and self.search_depth < 4:
            self.search_depth = 4
        elif self.games_played >= 50 and self.search_depth < 5:
            self.search_depth = 5
            self.max_depth = 6
        
        board = chess.Board()
        ai_moves = []
        
        # First pass: collect all AI moves
        for i, move in enumerate(moves):
            if board.turn == chess.BLACK:
                ai_moves.append((move, board.fen().split()[0]))
            board.push(move)
        
        # Second pass: learn from AI positions with temporal credit assignment
        board = chess.Board()
        for i, move in enumerate(moves):
            pos_hash = board.fen().split()[0]
            move_key = f"{move.from_square},{move.to_square}"
            
            # Learn from ALL positions (helps understand opponent patterns too)
            self.position_scores[pos_hash]['visits'] += 1
            self.move_scores[move_key]['visits'] += 1
            
            # Only update win/loss for AI's moves
            if board.turn == chess.BLACK:
                # Temporal credit assignment: later moves get more credit/blame
                move_index = len([m for m in ai_moves if moves.index(m[0]) <= i])
                weight = 0.5 + (0.5 * move_index / max(1, len(ai_moves)))
                
                if result == 1:  # AI won
                    self.position_scores[pos_hash]['wins'] += weight
                    self.move_scores[move_key]['wins'] += weight
                elif result == -1:  # AI lost
                    self.position_scores[pos_hash]['losses'] += weight
                    self.move_scores[move_key]['losses'] += weight
                    # Especially blame moves near the end
                    if i >= len(moves) * 0.7:
                        move_tuple_key = (move.from_square, move.to_square)
                        self.losing_moves[move_tuple_key] += 2
                else:
                    self.position_scores[pos_hash]['draws'] += weight
                    self.move_scores[move_key]['draws'] += weight
            
            # Build opening book from good games (wins and draws)
            if result >= 0 and board.fullmove_number <= 8 and board.turn == chess.BLACK:
                if move not in self.opening_book[pos_hash]:
                    self.opening_book[pos_hash].append(move)
            
            board.push(move)
        
        # Learn from player's successful patterns (opponent modeling)
        if result == -1:  # AI lost, learn from player
            board = chess.Board()
            for i, move in enumerate(moves):
                if board.turn == chess.WHITE and board.fullmove_number <= 8:
                    pos_hash = board.fen().split()[0]
                    # Store good player moves to consider in similar positions
                    if move not in self.opening_book[pos_hash]:
                        self.opening_book[pos_hash].append(move)
                board.push(move)
        
        # Prune old data more aggressively to keep best information
        if len(self.position_scores) > 8000:
            self.position_scores = defaultdict(lambda: {'wins': 0, 'losses': 0, 'draws': 0, 'visits': 0},
                {k: v for k, v in self.position_scores.items() if v['visits'] >= 2})
        
        if len(self.move_scores) > 3000:
            self.move_scores = defaultdict(lambda: {'wins': 0, 'losses': 0, 'draws': 0, 'visits': 0},
                {k: v for k, v in self.move_scores.items() if v['visits'] >= 2})
        
        if len(self.losing_moves) > 1000:
            self.losing_moves = defaultdict(int,
                {k: v for k, v in self.losing_moves.items() if v >= 3})
        
        # Prune opening book to keep best moves
        for pos_hash in list(self.opening_book.keys()):
            if len(self.opening_book[pos_hash]) > 5:
                # Keep top 5 moves by score
                scored = [(m, self.move_scores.get(f"{m.from_square},{m.to_square}", 
                                                   {'wins': 0, 'losses': 0, 'visits': 0})) 
                         for m in self.opening_book[pos_hash]]
                scored.sort(key=lambda x: x[1]['wins'] - x[1]['losses'], reverse=True)
                self.opening_book[pos_hash] = [m for m, _ in scored[:5]]
        
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
            'move_scores': dict(self.move_scores),
            'opening_book': opening_book_json,
            'losing_moves': {f"{k[0]},{k[1]}": v for k, v in self.losing_moves.items()},
            'games_played': self.games_played,
            'search_depth': self.search_depth,
            'max_depth': self.max_depth
        }
        
        filepath = os.path.join(MODELS_DIR, f"{self.username}_knowledge.json")
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_knowledge(self):
        """Load learned data from file"""
        filepath = os.path.join(MODELS_DIR, f"{self.username}_knowledge.json")
        if not os.path.exists(filepath):
            return
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.position_scores = defaultdict(lambda: {'wins': 0, 'losses': 0, 'draws': 0, 'visits': 0})
            for k, v in data.get('position_scores', {}).items():
                # Ensure all fields exist
                if 'visits' not in v:
                    v['visits'] = v.get('wins', 0) + v.get('losses', 0) + v.get('draws', 0)
                self.position_scores[k] = v
            
            self.move_scores = defaultdict(lambda: {'wins': 0, 'losses': 0, 'draws': 0, 'visits': 0})
            for k, v in data.get('move_scores', {}).items():
                if 'visits' not in v:
                    v['visits'] = v.get('wins', 0) + v.get('losses', 0) + v.get('draws', 0)
                self.move_scores[k] = v
            
            # Convert UCI strings back to chess.Move objects
            opening_book_json = data.get('opening_book', {})
            self.opening_book = defaultdict(list)
            for pos_hash, uci_moves in opening_book_json.items():
                try:
                    self.opening_book[pos_hash] = [chess.Move.from_uci(uci) for uci in uci_moves]
                except:
                    pass  # Skip invalid moves
            
            losing_moves_data = data.get('losing_moves', {})
            self.losing_moves = defaultdict(int)
            for key, value in losing_moves_data.items():
                try:
                    from_sq, to_sq = map(int, key.split(','))
                    self.losing_moves[(from_sq, to_sq)] = value
                except:
                    pass
            
            self.games_played = data.get('games_played', 0)
            self.search_depth = data.get('search_depth', 3)
            self.max_depth = data.get('max_depth', 5)
            
            print(f"Loaded knowledge: {self.games_played} games, depth {self.search_depth}, "
                  f"{len(self.position_scores)} positions, {len(self.opening_book)} opening positions")
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
