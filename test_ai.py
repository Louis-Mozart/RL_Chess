"""
Simple script to test the chess AI without GUI
"""
import chess
from chess_ai import ChessAI
from user_manager import UserManager


def play_test_game():
    """Play a simple test game to verify everything works"""
    print("Testing Chess AI...")
    
    # Create test user
    user_manager = UserManager()
    test_username = "test_player"
    
    if test_username not in user_manager.list_users():
        user_manager.create_user(test_username)
    
    user_manager.select_user(test_username)
    
    # Initialize AI
    stats = user_manager.get_user_stats()
    ai = ChessAI(test_username, stats['ai_level'])
    
    # Play a few moves
    board = chess.Board()
    print("\nStarting position:")
    print(board)
    
    moves_count = 0
    max_moves = 10
    
    while not board.is_game_over() and moves_count < max_moves:
        if board.turn == chess.WHITE:
            # Simple player move (just pick first legal move for testing)
            legal_moves = list(board.legal_moves)
            move = legal_moves[0]
            print(f"\nPlayer moves: {move}")
        else:
            # AI move
            move = ai.get_move(board, use_exploration=True)
            print(f"\nAI moves: {move}")
        
        board.push(move)
        print(board)
        moves_count += 1
    
    print("\n✓ Test completed successfully!")
    print(f"Moves played: {moves_count}")
    print(f"AI Level: {ai.ai_level}")


if __name__ == "__main__":
    play_test_game()
