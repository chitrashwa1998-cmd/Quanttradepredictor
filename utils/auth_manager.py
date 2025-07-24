
import os
import hashlib
import secrets
import streamlit as st
from typing import Optional, Dict, Any
from utils.database_adapter import DatabaseAdapter

class AuthManager:
    """Authentication manager for admin login without Replit dependency."""
    
    def __init__(self):
        """Initialize authentication manager."""
        self.db = DatabaseAdapter()
        self._ensure_auth_table()
        
    def _ensure_auth_table(self):
        """Create authentication table if it doesn't exist."""
        try:
            with self.db.db.conn.cursor() as cursor:
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS admin_users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(255) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    salt VARCHAR(255) NOT NULL,
                    role VARCHAR(50) DEFAULT 'admin',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE
                )
                """)
                
                # Check if admin user exists, if not create default one
                cursor.execute("SELECT COUNT(*) FROM admin_users WHERE username = 'admin'")
                if cursor.fetchone()[0] == 0:
                    self._create_default_admin()
                    
        except Exception as e:
            print(f"Error creating auth table: {e}")
    
    def _create_default_admin(self):
        """Create default admin user."""
        default_password = "TribexAdmin2024!"
        self.create_user("admin", default_password)
        print("🔐 Default admin user created:")
        print(f"   Username: admin")
        print(f"   Password: {default_password}")
        print("   ⚠️ Please change this password after first login!")
    
    def _hash_password(self, password: str, salt: str = None) -> tuple:
        """Hash password with salt."""
        if salt is None:
            salt = secrets.token_hex(32)
        
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        ).hex()
        
        return password_hash, salt
    
    def create_user(self, username: str, password: str, role: str = "admin") -> bool:
        """Create a new admin user."""
        try:
            password_hash, salt = self._hash_password(password)
            
            with self.db.db.conn.cursor() as cursor:
                cursor.execute("""
                INSERT INTO admin_users (username, password_hash, salt, role)
                VALUES (%s, %s, %s, %s)
                """, (username, password_hash, salt, role))
                
            return True
        except Exception as e:
            print(f"Error creating user: {e}")
            return False
    
    def authenticate(self, username: str, password: str) -> bool:
        """Authenticate user credentials."""
        try:
            with self.db.db.conn.cursor() as cursor:
                cursor.execute("""
                SELECT password_hash, salt, is_active 
                FROM admin_users 
                WHERE username = %s
                """, (username,))
                
                result = cursor.fetchone()
                if not result:
                    return False
                
                stored_hash, salt, is_active = result
                
                if not is_active:
                    return False
                
                # Hash the provided password with the stored salt
                password_hash, _ = self._hash_password(password, salt)
                
                if password_hash == stored_hash:
                    # Update last login
                    cursor.execute("""
                    UPDATE admin_users 
                    SET last_login = CURRENT_TIMESTAMP 
                    WHERE username = %s
                    """, (username,))
                    return True
                
            return False
        except Exception as e:
            print(f"Authentication error: {e}")
            return False
    
    def change_password(self, username: str, old_password: str, new_password: str) -> bool:
        """Change user password."""
        try:
            # First verify old password
            if not self.authenticate(username, old_password):
                return False
            
            # Hash new password
            password_hash, salt = self._hash_password(new_password)
            
            with self.db.db.conn.cursor() as cursor:
                cursor.execute("""
                UPDATE admin_users 
                SET password_hash = %s, salt = %s 
                WHERE username = %s
                """, (password_hash, salt, username))
                
            return True
        except Exception as e:
            print(f"Error changing password: {e}")
            return False
    
    def get_user_info(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user information."""
        try:
            with self.db.db.conn.cursor() as cursor:
                cursor.execute("""
                SELECT username, role, created_at, last_login, is_active
                FROM admin_users 
                WHERE username = %s
                """, (username,))
                
                result = cursor.fetchone()
                if result:
                    return {
                        'username': result[0],
                        'role': result[1],
                        'created_at': result[2],
                        'last_login': result[3],
                        'is_active': result[4]
                    }
            return None
        except Exception as e:
            print(f"Error getting user info: {e}")
            return None

def check_authentication():
    """Check if user is authenticated in current session."""
    return st.session_state.get('authenticated', False)

def get_current_user():
    """Get current authenticated user."""
    return st.session_state.get('current_user', None)

def logout():
    """Logout current user."""
    st.session_state.authenticated = False
    st.session_state.current_user = None
    st.rerun()

def show_login_page():
    """Display login page."""
    st.markdown("""
    <div style="max-width: 400px; margin: 2rem auto; padding: 2rem; 
         background: rgba(255,255,255,0.05); border-radius: 10px; 
         border: 1px solid rgba(255,255,255,0.1);">
        <h2 style="text-align: center; color: #00d4ff; margin-bottom: 2rem;">
            🔐 TribexAlpha Admin Login
        </h2>
    """, unsafe_allow_html=True)
    
    auth_manager = AuthManager()
    
    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter admin username")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            login_button = st.form_submit_button("🔑 Login", type="primary", use_container_width=True)
        with col2:
            if st.form_submit_button("ℹ️ Help"):
                st.info("""
                **Default Credentials:**
                - Username: admin
                - Password: TribexAdmin2024!
                
                ⚠️ Change password after first login!
                """)
    
    if login_button:
        if username and password:
            if auth_manager.authenticate(username, password):
                st.session_state.authenticated = True
                st.session_state.current_user = username
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid username or password")
        else:
            st.error("❌ Please enter both username and password")
    
    st.markdown("</div>", unsafe_allow_html=True)

def require_authentication():
    """Decorator function to require authentication for pages."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not check_authentication():
                show_login_page()
                return
            return func(*args, **kwargs)
        return wrapper
    return decorator
