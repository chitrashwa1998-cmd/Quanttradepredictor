import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

// Ensure DOM is ready before mounting
const renderApp = () => {
  try {
    const rootElement = document.getElementById('root');
    if (rootElement && rootElement.nodeType === Node.ELEMENT_NODE) {
      const root = ReactDOM.createRoot(rootElement);
      root.render(
        <React.StrictMode>
          <App />
        </React.StrictMode>
      );
      console.log('✅ React app mounted successfully');
    } else {
      console.error('❌ Root element not found or invalid');
      if (document.readyState === 'complete') {
        setTimeout(renderApp, 500); // Retry after 500ms if DOM is complete
      }
    }
  } catch (error) {
    console.error('❌ React mounting error:', error);
    setTimeout(renderApp, 1000); // Retry after error
  }
};

// Wait for DOM to be fully ready
const initApp = () => {
  if (document.readyState === 'complete') {
    renderApp();
  } else {
    setTimeout(initApp, 100);
  }
};

// Start initialization
initApp();