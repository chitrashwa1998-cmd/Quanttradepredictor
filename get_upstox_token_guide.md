# How to Get a New Upstox Access Token

## Step-by-Step Instructions

### 1. Open Your Streamlit App
- Go to your app URL: http://0.0.0.0:5000
- Navigate to "Upstox Data" page in the sidebar

### 2. Clear Old Token (if needed)
- If you see "Token is expired or invalid" message
- Click the "Refresh Token" button
- This will clear your old expired token

### 3. Start Authentication Process
- Click the "Login to Upstox" button
- This will redirect you to Upstox's official login page

### 4. Complete Upstox Authentication
- Enter your Upstox credentials (username/password)
- Complete any 2FA if required
- Grant permission to the app when prompted

### 5. Return to App
- You'll be redirected back to your Streamlit app
- The app will automatically exchange the code for an access token
- You should see "Successfully authenticated with Upstox!" message

### 6. Verify Token
- Check the "Token Status" section
- Should show "Token is valid and working!"
- Current NIFTY price should be displayed

### 7. Start WebSocket
- Click "Start WebSocket Stream"
- Connection should now work without disconnections

## Troubleshooting

### If Login Button Doesn't Work
- Check that UPSTOX_API_KEY and UPSTOX_API_SECRET are set in your environment
- Refresh the page and try again

### If Redirect Fails
- Make sure you're using the correct app URL
- Check that there are no popup blockers preventing the redirect

### If Token Still Shows Invalid
- Try the authentication process again
- Ensure you're using the correct Upstox account

### If WebSocket Still Disconnects
- Verify token status shows "valid and working"
- Check market hours (WebSocket only works during trading hours)
- Look for any error messages in the app

## Token Validity
- Upstox tokens expire after 24 hours
- You'll need to repeat this process daily
- The app will show token status to help you know when to refresh

## Need Help?
If you're still having issues, let me know exactly what step isn't working and I can provide more specific guidance.