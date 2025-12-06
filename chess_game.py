"""
Chess game logic using python-chess library
"""
import chess
from typing import List, Tuple, Optional


class ChessGame:
    def __init__(self):
        self.board = chess.Board()
        self.move_history = []
        self.selected_square = None
        
    def reset(self):
        """Reset the game to initial state"""
        self.board = chess.Board()
        self.move_history = []
        self.selected_square = None
    
    def get_legal_moves_from_square(self, square: int) -> List[chess.Move]:
        """Get all legal moves from a specific square"""
        return [move for move in self.board.legal_moves 
                if move.from_square == square]
    
    def make_move(self, move: chess.Move) -> bool:
        """Make a move if it's legal"""
        if move in self.board.legal_moves:
            self.board.push(move)
            self.move_history.append(move)
            return True
        return False
    
    def make_move_uci(self, uci_string: str) -> bool:
        """Make a move from UCI string (e.g., 'e2e4')"""
        try:
            move = chess.Move.from_uci(uci_string)
            return self.make_move(move)
        except:
            return False
    
    def undo_move(self):
        """Undo the last move"""
        if self.board.move_stack:
            move = self.board.pop()
            self.move_history.pop()
            return move
        return None
    
    def is_game_over(self) -> bool:
        """Check if the game is over"""
        return self.board.is_game_over()
    
    def get_result(self) -> Optional[str]:
        """Get the game result"""
        if not self.board.is_game_over():
            return None
        
        if self.board.is_checkmate():
            # Return winner
            return "black" if self.board.turn == chess.WHITE else "white"
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            return "draw"
        elif self.board.is_fifty_moves() or self.board.is_repetition():
            return "draw"
        return "draw"
    
    def get_board_state(self):
        """Get current board state as FEN"""
        return self.board.fen()
    
    def get_piece_at(self, square: int):
        """Get the piece at a specific square"""
        return self.board.piece_at(square)
    
    def get_all_legal_moves(self) -> List[chess.Move]:
        """Get all legal moves in current position"""
        return list(self.board.legal_moves)
    
    def is_check(self) -> bool:
        """Check if current player is in check"""
        return self.board.is_check()
    
    def get_current_turn(self) -> str:
        """Get whose turn it is"""
        return "white" if self.board.turn == chess.WHITE else "black"
    
    def square_to_coords(self, square: int) -> Tuple[int, int]:
        """Convert chess square index to board coordinates"""
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        return (file, 7 - rank)  # Flip rank for display
    
    def coords_to_square(self, file: int, rank: int) -> int:
        """Convert board coordinates to chess square index"""
        return chess.square(file, 7 - rank)
