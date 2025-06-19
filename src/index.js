
import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

// Ensure DOM is ready before mounting React
const mountReactApp = () => {
  const rootElement = document.getElementById('root');
  
  if (!rootElement) {
    console.error('Root element not found! Make sure there is a div with id="root" in your HTML.');
    return;
  }

  try {
    const root = ReactDOM.createRoot(rootElement);
    root.render(
      <React.StrictMode>
        <App />
      </React.StrictMode>
    );
    console.log('✅ React app mounted successfully');
  } catch (error) {
    console.error('Error mounting React app:', error);
  }
};

// Wait for DOM to be ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', mountReactApp);
} else {
  mountReactApp();
}
