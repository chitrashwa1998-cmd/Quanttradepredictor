# Upstox WebSocket Connection Troubleshooting Guide

## Problem: WebSocket keeps disconnecting ("WebSocket onclose")

### Root Cause
The most common cause is an **expired Upstox access token**. Upstox tokens typically expire after 24 hours and need to be refreshed.

### Quick Fix Steps

1. **Go to your Streamlit app** in the browser
2. **Click on "Upstox Data"** in the sidebar
3. **Look for the Token Status section** - it will show if your token is expired
4. **Click "Refresh Token"** button
5. **Click "Login to Upstox"** button (this will appear after refresh)
6. **Complete the Upstox authentication** in the popup/new tab
7. **Return to the app** - you should now see "Token is valid and working!"
8. **Click "Start WebSocket Stream"** to connect

### How to Verify the Fix

- Token Status should show: ✅ "Token is valid and working!"
- Current NIFTY price should be displayed
- WebSocket connection should start successfully
- You should see live price updates in the stream

### Common Error Messages and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| "Token is expired or invalid" | Token expired | Refresh token using steps above |
| "Failed to get WebSocket URL" | Authentication issue | Get fresh token |
| "WebSocket onclose" repeatedly | Token expired | Refresh token |
| "401 Unauthorized" | Invalid credentials | Check API keys and refresh token |
| "403 Forbidden" | Token lacks permissions | Ensure token has WebSocket access |

### Technical Details

- **Token Validity**: Upstox tokens expire every 24 hours
- **WebSocket URL**: Generated fresh for each session
- **Authentication**: Uses Bearer token in WebSocket headers
- **Auto-Save**: Valid tokens are automatically saved to `.upstox_token` file
- **Connection Check**: App tests token validity before allowing WebSocket connection

### Prevention

- Bookmark the Upstox Data page for easy token refresh
- Monitor the Token Status section before starting WebSocket
- Set up daily token refresh routine if using the app regularly

### Still Having Issues?

1. Check that UPSTOX_API_KEY and UPSTOX_API_SECRET are set in environment
2. Verify your Upstox account has WebSocket API access
3. Test with console script: `python3 test_websocket.py`
4. Check network connectivity and firewall settings