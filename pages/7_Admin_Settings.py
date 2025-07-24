
import streamlit as st
from utils.auth_manager import AuthManager, check_authentication, get_current_user, logout, require_authentication

st.set_page_config(page_title="Admin Settings", page_icon="⚙️", layout="wide")

# Load custom CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

@require_authentication()
def show_admin_settings():
    """Show admin settings page."""
    
    st.markdown("""
    <div class="trading-header">
        <h1 style="margin:0;">⚙️ ADMIN SETTINGS</h1>
        <p style="font-size: 1.2rem; margin: 1rem 0 0 0; color: rgba(255,255,255,0.8);">
            Account Management & Security
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    auth_manager = AuthManager()
    current_user = get_current_user()
    
    # User Info Section
    st.header("👤 Account Information")
    
    user_info = auth_manager.get_user_info(current_user)
    if user_info:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Username", user_info['username'])
            st.metric("Role", user_info['role'].title())
            
        with col2:
            if user_info['created_at']:
                st.metric("Account Created", user_info['created_at'].strftime('%Y-%m-%d %H:%M'))
            if user_info['last_login']:
                st.metric("Last Login", user_info['last_login'].strftime('%Y-%m-%d %H:%M'))
    
    st.markdown("---")
    
    # Password Change Section
    st.header("🔒 Change Password")
    
    with st.form("change_password_form"):
        st.write("**Update your admin password:**")
        
        old_password = st.text_input("Current Password", type="password", key="old_pass")
        new_password = st.text_input("New Password", type="password", key="new_pass")
        confirm_password = st.text_input("Confirm New Password", type="password", key="confirm_pass")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            change_password_button = st.form_submit_button("🔄 Update Password", type="primary")
        
        if change_password_button:
            if not all([old_password, new_password, confirm_password]):
                st.error("❌ Please fill in all password fields")
            elif new_password != confirm_password:
                st.error("❌ New passwords do not match")
            elif len(new_password) < 8:
                st.error("❌ New password must be at least 8 characters long")
            elif auth_manager.change_password(current_user, old_password, new_password):
                st.success("✅ Password updated successfully!")
            else:
                st.error("❌ Current password is incorrect")
    
    # Password Security Tips
    with st.expander("🛡️ Password Security Tips"):
        st.markdown("""
        **Create a strong password:**
        - Use at least 8 characters
        - Include uppercase and lowercase letters
        - Include numbers and special characters
        - Avoid common words or personal information
        - Don't reuse passwords from other accounts
        
        **Example strong password:** `TribexSecure2024!@#`
        """)
    
    st.markdown("---")
    
    # Session Management
    st.header("🚪 Session Management")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🚪 Logout", type="secondary"):
            logout()
    
    with col2:
        st.info("Click logout to end your current session securely.")

# Run the main function
show_admin_settings()
