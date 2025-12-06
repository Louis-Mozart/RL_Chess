"""
User management system for tracking individual player progress
"""
import json
import os
from datetime import datetime
from config import USER_DATA_DIR, MODELS_DIR


class UserManager:
    def __init__(self):
        self.users_file = os.path.join(USER_DATA_DIR, "users.json")
        self.ensure_directories()
        self.users = self.load_users()
        self.current_user = None
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        os.makedirs(USER_DATA_DIR, exist_ok=True)
        os.makedirs(MODELS_DIR, exist_ok=True)
    
    def load_users(self):
        """Load user data from JSON file"""
        if os.path.exists(self.users_file):
            with open(self.users_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_users(self):
        """Save user data to JSON file"""
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f, indent=4)
    
    def create_user(self, username):
        """Create a new user profile"""
        if username in self.users:
            return False
        
        self.users[username] = {
            "created_at": datetime.now().isoformat(),
            "games_played": 0,
            "games_won": 0,
            "games_lost": 0,
            "games_drawn": 0,
            "total_moves": 0,
            "last_played": None
        }
        self.save_users()
        return True
    
    def select_user(self, username):
        """Select a user as the current player"""
        if username in self.users:
            self.current_user = username
            return True
        return False
    
    def get_user_stats(self, username=None):
        """Get statistics for a user"""
        user = username or self.current_user
        if user and user in self.users:
            return self.users[user]
        return None
    
    def update_game_result(self, result):
        """
        Update user statistics after a game
        result: 'win', 'loss', or 'draw'
        """
        if not self.current_user:
            return
        
        user_data = self.users[self.current_user]
        user_data["games_played"] += 1
        user_data["last_played"] = datetime.now().isoformat()
        
        if result == "win":
            user_data["games_won"] += 1
        elif result == "loss":
            user_data["games_lost"] += 1
        else:  # draw
            user_data["games_drawn"] += 1
        
        self.save_users()
    
    def delete_user(self, username):
        """Delete a user profile and their model"""
        if username in self.users:
            del self.users[username]
            self.save_users()
            
            # Delete user's model file if it exists
            model_file = os.path.join(MODELS_DIR, f"{username}_model.pth")
            if os.path.exists(model_file):
                os.remove(model_file)
            return True
        return False
    
    def list_users(self):
        """Get list of all usernames"""
        return list(self.users.keys())
    
    def increment_moves(self):
        """Increment the move counter for current user"""
        if self.current_user and self.current_user in self.users:
            self.users[self.current_user]["total_moves"] += 1
            self.save_users()
