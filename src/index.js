import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

// Ensure DOM is ready before mounting
const renderApp = () => {
  try {
    const rootElement = document.getElementById('root');
    if (rootElement && rootElement.nodeType === Node.ELEMENT_NODE) {
      // Clear any existing content
      rootElement.innerHTML = '';

      const root = ReactDOM.createRoot(rootElement);
      root.render(
        <React.StrictMode>
          <App />
        </React.StrictMode>
      );
      console.log('✅ React app mounted successfully');
    } else {
      console.error('❌ Root element not found or invalid');
      // Retry with exponential backoff
      setTimeout(renderApp, 200);
    }
  } catch (error) {
    console.error('❌ React mounting error:', error);
    setTimeout(renderApp, 500);
  }
};

// Wait for DOM to be completely ready
const waitForDOM = () => {
  if (document.readyState === 'complete') {
    setTimeout(renderApp, 100); // Small delay to ensure DOM is fully rendered
  } else {
    setTimeout(waitForDOM, 50);
  }
};

// Start the initialization process
waitForDOM();