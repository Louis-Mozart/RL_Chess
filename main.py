"""
Pygame GUI for the Chess game
"""
import pygame
import chess
from chess_game import ChessGame
from chess_ai_minimax import ChessAI  # Using new minimax AI
from user_manager import UserManager
from config import (WINDOW_WIDTH, WINDOW_HEIGHT, BOARD_SIZE, SQUARE_SIZE, FPS,
                   WHITE, BLACK, LIGHT_SQUARE, DARK_SQUARE, 
                   HIGHLIGHT_COLOR, SELECTED_COLOR)


class ChessGUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("RL Chess - Learning AI")
        self.clock = pygame.time.Clock()
        
        self.game = ChessGame()
        self.user_manager = UserManager()
        self.ai = None
        
        self.selected_square = None
        self.valid_moves = []
        self.player_color = chess.WHITE
        self.game_over = False
        self.game_result = None
        
        # Load piece images
        self.load_pieces()
        
        # Font for text
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # State management
        self.state = "menu"  # menu, select_user, game, game_over
        self.input_text = ""
        self.message = ""
        self.message_timer = 0
        
    def load_pieces(self):
        """Load chess piece images"""
        self.pieces = {}
        piece_names = ['p', 'n', 'b', 'r', 'q', 'k']
        
        for piece_name in piece_names:
            # White pieces (uppercase)
            self.pieces[piece_name.upper()] = self.create_piece_surface(
                piece_name, True)
            # Black pieces (lowercase)
            self.pieces[piece_name] = self.create_piece_surface(
                piece_name, False)
    
    def create_piece_surface(self, piece_type, is_white):
        """Create beautiful chess pieces with clear distinction"""
        surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
        
        # Use proper Unicode chess symbols - filled for white, outlined for black
        if is_white:
            symbols = {
                'p': '♙',  # White Pawn
                'n': '♘',  # White Knight  
                'b': '♗',  # White Bishop
                'r': '♖',  # White Rook
                'q': '♕',  # White Queen
                'k': '♔'   # White King
            }
            piece_color = (255, 255, 255)  # White
            outline_color = (0, 0, 0)  # Black outline
        else:
            symbols = {
                'p': '♟',  # Black Pawn
                'n': '♞',  # Black Knight
                'b': '♝',  # Black Bishop
                'r': '♜',  # Black Rook
                'q': '♛',  # Black Queen
                'k': '♚'   # Black King
            }
            piece_color = (40, 40, 40)  # Very dark gray
            outline_color = (255, 255, 255)  # White outline
        
        # Use a nice font that supports chess symbols
        try:
            font = pygame.font.SysFont('segoeuisymbol,dejavusans,noto', 68)
        except:
            font = pygame.font.Font(None, 68)
        
        # Draw multiple outline layers for better contrast
        for offset in [(0, -3), (0, 3), (-3, 0), (3, 0), (-2, -2), (2, -2), (-2, 2), (2, 2)]:
            outline = font.render(symbols[piece_type], True, outline_color)
            outline_rect = outline.get_rect(center=(SQUARE_SIZE // 2 + offset[0], SQUARE_SIZE // 2 + offset[1]))
            surface.blit(outline, outline_rect)
        
        # Draw the main piece on top
        text = font.render(symbols[piece_type], True, piece_color)
        text_rect = text.get_rect(center=(SQUARE_SIZE // 2, SQUARE_SIZE // 2))
        surface.blit(text, text_rect)
        
        return surface
    
    def draw_board(self):
        """Draw the chess board"""
        for rank in range(8):
            for file in range(8):
                color = LIGHT_SQUARE if (rank + file) % 2 == 0 else DARK_SQUARE
                pygame.draw.rect(self.screen, color,
                               (file * SQUARE_SIZE, rank * SQUARE_SIZE,
                                SQUARE_SIZE, SQUARE_SIZE))
    
    def draw_pieces(self):
        """Draw all pieces on the board"""
        for square in chess.SQUARES:
            piece = self.game.get_piece_at(square)
            if piece:
                file, rank = self.game.square_to_coords(square)
                piece_symbol = piece.symbol()
                if piece_symbol in self.pieces:
                    self.screen.blit(self.pieces[piece_symbol],
                                   (file * SQUARE_SIZE, rank * SQUARE_SIZE))
    
    def draw_highlights(self):
        """Draw highlights for selected square and valid moves"""
        if self.selected_square is not None:
            file, rank = self.game.square_to_coords(self.selected_square)
            # Draw yellow border for selected square
            border_rect = pygame.Rect(file * SQUARE_SIZE, rank * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(self.screen, (255, 255, 0), border_rect, 4)
            
            # Subtle highlight overlay
            s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE))
            s.set_alpha(50)
            s.fill(SELECTED_COLOR)
            self.screen.blit(s, (file * SQUARE_SIZE, rank * SQUARE_SIZE))
        
        for move in self.valid_moves:
            file, rank = self.game.square_to_coords(move.to_square)
            center_x = file * SQUARE_SIZE + SQUARE_SIZE // 2
            center_y = rank * SQUARE_SIZE + SQUARE_SIZE // 2
            
            # Check if target square has a piece (capture move)
            if self.game.get_piece_at(move.to_square):
                # Draw a red circle border for captures
                pygame.draw.circle(self.screen, (220, 50, 50), (center_x, center_y), SQUARE_SIZE // 2 - 5, 4)
            else:
                # Draw a smaller filled circle for normal moves
                pygame.draw.circle(self.screen, HIGHLIGHT_COLOR, (center_x, center_y), SQUARE_SIZE // 7)
    
    def draw_game_info(self):
        """Draw game information panel"""
        # Status bar at bottom
        info_rect = pygame.Rect(0, BOARD_SIZE, WINDOW_WIDTH, 100)
        pygame.draw.rect(self.screen, (50, 50, 50), info_rect)
        
        # Current turn
        turn_text = f"Turn: {'White (You)' if self.game.get_current_turn() == 'white' else 'Black (AI)'}"
        turn_surface = self.small_font.render(turn_text, True, WHITE)
        self.screen.blit(turn_surface, (10, BOARD_SIZE + 10))
        
        # Games played and AI learning progress
        if self.user_manager.current_user:
            stats = self.user_manager.get_user_stats()
            games_text = f"Games Played: {stats['games_played']}"
            games_surface = self.small_font.render(games_text, True, WHITE)
            self.screen.blit(games_surface, (10, BOARD_SIZE + 35))
            
            # Show AI learning metrics
            if self.ai:
                epsilon_pct = int((1 - self.ai.epsilon) * 100)
                learning_text = f"AI Intelligence: {epsilon_pct}%"
                learning_color = (100 + epsilon_pct, 255 - epsilon_pct, 100)
                learning_surface = self.small_font.render(learning_text, True, learning_color)
                self.screen.blit(learning_surface, (10, BOARD_SIZE + 60))
        
        # User stats
        if self.user_manager.current_user:
            stats = self.user_manager.get_user_stats()
            stats_text = f"W:{stats['games_won']} L:{stats['games_lost']} D:{stats['games_drawn']}"
            stats_surface = self.small_font.render(stats_text, True, WHITE)
            self.screen.blit(stats_surface, (WINDOW_WIDTH - 200, BOARD_SIZE + 10))
            
            user_text = f"Player: {self.user_manager.current_user}"
            user_surface = self.small_font.render(user_text, True, WHITE)
            self.screen.blit(user_surface, (WINDOW_WIDTH - 300, BOARD_SIZE + 35))
        
        # Check indicator
        if self.game.is_check():
            check_text = "CHECK!"
            check_surface = self.font.render(check_text, True, (255, 0, 0))
            # Draw on a semi-transparent background
            text_bg = pygame.Surface((200, 50))
            text_bg.set_alpha(180)
            text_bg.fill((0, 0, 0))
            self.screen.blit(text_bg, (WINDOW_WIDTH // 2 - 100, BOARD_SIZE + 60))
            check_rect = check_surface.get_rect(center=(WINDOW_WIDTH // 2, BOARD_SIZE + 85))
            self.screen.blit(check_surface, check_rect)
        
        # Message display
        if self.message and self.message_timer > 0:
            msg_surface = self.small_font.render(self.message, True, (255, 255, 0))
            msg_rect = msg_surface.get_rect(center=(WINDOW_WIDTH // 2, BOARD_SIZE + 70))
            self.screen.blit(msg_surface, msg_rect)
            self.message_timer -= 1
    
    def draw_menu(self):
        """Draw main menu"""
        self.screen.fill((40, 40, 40))
        
        title = self.font.render("RL Chess - Learning AI", True, WHITE)
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, 100))
        self.screen.blit(title, title_rect)
        
        subtitle = self.small_font.render("The AI learns from every game!", True, (200, 200, 200))
        subtitle_rect = subtitle.get_rect(center=(WINDOW_WIDTH // 2, 150))
        self.screen.blit(subtitle, subtitle_rect)
        
        # Menu options
        options = [
            ("N - New User", 250),
            ("S - Select User", 300),
            ("L - List Users", 350),
            ("Q - Quit", 400)
        ]
        
        for text, y in options:
            surface = self.small_font.render(text, True, WHITE)
            rect = surface.get_rect(center=(WINDOW_WIDTH // 2, y))
            self.screen.blit(surface, rect)
        
        if self.message:
            msg_surface = self.small_font.render(self.message, True, (255, 255, 0))
            msg_rect = msg_surface.get_rect(center=(WINDOW_WIDTH // 2, 500))
            self.screen.blit(msg_surface, msg_rect)
    
    def draw_user_input(self):
        """Draw user input screen"""
        self.screen.fill((40, 40, 40))
        
        prompt = self.font.render("Enter Username:", True, WHITE)
        prompt_rect = prompt.get_rect(center=(WINDOW_WIDTH // 2, 200))
        self.screen.blit(prompt, prompt_rect)
        
        # Input box
        input_box = pygame.Rect(WINDOW_WIDTH // 2 - 150, 250, 300, 50)
        pygame.draw.rect(self.screen, WHITE, input_box, 2)
        
        input_surface = self.font.render(self.input_text, True, WHITE)
        self.screen.blit(input_surface, (input_box.x + 10, input_box.y + 10))
        
        instruction = self.small_font.render("Press ENTER to confirm, ESC to cancel", True, (200, 200, 200))
        inst_rect = instruction.get_rect(center=(WINDOW_WIDTH // 2, 350))
        self.screen.blit(instruction, inst_rect)
        
        if self.message:
            msg_surface = self.small_font.render(self.message, True, (255, 100, 100))
            msg_rect = msg_surface.get_rect(center=(WINDOW_WIDTH // 2, 400))
            self.screen.blit(msg_surface, msg_rect)
    
    def draw_game_over(self):
        """Draw game over screen"""
        overlay = pygame.Surface((BOARD_SIZE, BOARD_SIZE))
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        result_text = ""
        if self.game_result == "white":
            result_text = "You Win!"
        elif self.game_result == "black":
            result_text = "AI Wins!"
        else:
            result_text = "Draw!"
        
        result_surface = self.font.render(result_text, True, WHITE)
        result_rect = result_surface.get_rect(center=(WINDOW_WIDTH // 2, BOARD_SIZE // 2 - 50))
        self.screen.blit(result_surface, result_rect)
        
        # Show learning message
        learning_msg = "AI is learning from this game..."
        learning_surface = self.small_font.render(learning_msg, True, (100, 200, 255))
        learning_rect = learning_surface.get_rect(center=(WINDOW_WIDTH // 2, BOARD_SIZE // 2))
        self.screen.blit(learning_surface, learning_rect)
        
        instruction = self.small_font.render("Press SPACE for new game, ESC for menu", True, WHITE)
        inst_rect = instruction.get_rect(center=(WINDOW_WIDTH // 2, BOARD_SIZE // 2 + 50))
        self.screen.blit(instruction, inst_rect)
    
    def handle_square_click(self, pos):
        """Handle mouse click on a square"""
        if self.game_over or self.game.get_current_turn() != "white":
            return
        
        # Check if click is within board bounds
        if pos[0] < 0 or pos[0] >= BOARD_SIZE or pos[1] < 0 or pos[1] >= BOARD_SIZE:
            return
        
        file = pos[0] // SQUARE_SIZE
        rank = pos[1] // SQUARE_SIZE
        
        # Double check bounds
        if file < 0 or file > 7 or rank < 0 or rank > 7:
            return
        
        clicked_square = self.game.coords_to_square(file, rank)
        
        # If a square is already selected
        if self.selected_square is not None:
            # Try to make a move
            move_made = False
            for move in self.valid_moves:
                if move.to_square == clicked_square:
                    self.game.make_move(move)
                    self.user_manager.increment_moves()
                    move_made = True
                    break
            
            self.selected_square = None
            self.valid_moves = []
            
            if move_made:
                # Check for game over
                if self.game.is_game_over():
                    self.handle_game_over()
                return
        
        # Select a new square
        piece = self.game.get_piece_at(clicked_square)
        if piece and piece.color == self.player_color:
            self.selected_square = clicked_square
            self.valid_moves = self.game.get_legal_moves_from_square(clicked_square)
    
    def make_ai_move(self):
        """Make AI move"""
        if self.game.get_current_turn() == "black" and not self.game_over:
            move = self.ai.get_move(self.game.board, use_exploration=True)
            if move:
                self.game.make_move(move)
                
                if self.game.is_game_over():
                    self.handle_game_over()
    
    def handle_game_over(self):
        """Handle game over state"""
        self.game_over = True
        result = self.game.get_result()
        self.game_result = result
        
        # Update user statistics
        if result == "white":
            self.user_manager.update_game_result("win")
            self.message = "You won! Great job!"
            ai_result = -1
        elif result == "black":
            self.user_manager.update_game_result("loss")
            self.message = "AI won! Better luck next time!"
            ai_result = 1
        else:
            self.user_manager.update_game_result("draw")
            self.message = "It's a draw!"
            ai_result = 0
        
        # Train AI from this game
        print(f"\nTraining AI on {len(self.game.move_history)} moves (result: {ai_result})...")
        self.ai.train_from_game(self.game.move_history, ai_result)
        print(f"AI trained! Epsilon: {self.ai.epsilon:.4f}")
        print(f"Memory - Bad positions: {len(self.ai.bad_positions)}, Good positions: {len(self.ai.good_positions)}, Losing moves: {len(self.ai.losing_moves)}")
        print(f"Experience replay buffer: {len(self.ai.memory)} positions\n")
        
        self.state = "game_over"
    
    def start_new_game(self):
        """Start a new game"""
        self.game.reset()
        self.selected_square = None
        self.valid_moves = []
        self.game_over = False
        self.game_result = None
        self.state = "game"
        self.message = ""
    
    def run(self):
        """Main game loop"""
        running = True
        
        while running:
            self.clock.tick(FPS)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if self.state == "menu":
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_n:
                            self.state = "new_user"
                            self.input_text = ""
                            self.message = ""
                        elif event.key == pygame.K_s:
                            self.state = "select_user"
                            self.input_text = ""
                            self.message = ""
                        elif event.key == pygame.K_l:
                            users = self.user_manager.list_users()
                            if users:
                                self.message = f"Users: {', '.join(users)}"
                            else:
                                self.message = "No users found"
                        elif event.key == pygame.K_q:
                            running = False
                
                elif self.state in ["new_user", "select_user"]:
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_RETURN and self.input_text:
                            if self.state == "new_user":
                                if self.user_manager.create_user(self.input_text):
                                    self.user_manager.select_user(self.input_text)
                                    self.ai = ChessAI(self.input_text)
                                    self.start_new_game()
                                else:
                                    self.message = "User already exists!"
                            else:  # select_user
                                if self.user_manager.select_user(self.input_text):
                                    self.ai = ChessAI(self.input_text)
                                    self.start_new_game()
                                else:
                                    self.message = "User not found!"
                        elif event.key == pygame.K_ESCAPE:
                            self.state = "menu"
                            self.input_text = ""
                            self.message = ""
                        elif event.key == pygame.K_BACKSPACE:
                            self.input_text = self.input_text[:-1]
                        elif event.unicode.isalnum() or event.unicode == '_':
                            if len(self.input_text) < 20:
                                self.input_text += event.unicode
                
                elif self.state == "game":
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        self.handle_square_click(event.pos)
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.state = "menu"
                
                elif self.state == "game_over":
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            self.start_new_game()
                        elif event.key == pygame.K_ESCAPE:
                            self.state = "menu"
            
            # AI move logic
            if self.state == "game" and not self.game_over:
                if self.game.get_current_turn() == "black":
                    pygame.time.wait(500)  # Small delay for better UX
                    self.make_ai_move()
            
            # Drawing
            if self.state == "menu":
                self.draw_menu()
            elif self.state in ["new_user", "select_user"]:
                self.draw_user_input()
            elif self.state in ["game", "game_over"]:
                self.draw_board()
                self.draw_highlights()
                self.draw_pieces()
                self.draw_game_info()
                if self.state == "game_over":
                    self.draw_game_over()
            
            pygame.display.flip()
        
        pygame.quit()


if __name__ == "__main__":
    gui = ChessGUI()
    gui.run()
