import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

// Ensure DOM is ready before mounting
const renderApp = () => {
  try {
    const rootElement = document.getElementById('root');
    
    // More thorough DOM element validation
    if (rootElement && 
        rootElement.nodeType === Node.ELEMENT_NODE && 
        rootElement.tagName === 'DIV') {
      
      console.log('✅ Root element found, mounting React app...');
      
      // Clear any existing content safely
      while (rootElement.firstChild) {
        rootElement.removeChild(rootElement.firstChild);
      }

      const root = ReactDOM.createRoot(rootElement);
      root.render(
        <React.StrictMode>
          <App />
        </React.StrictMode>
      );
      console.log('✅ React app mounted successfully');
    } else {
      console.error('❌ Root element not found or invalid:', rootElement);
      // Retry with shorter interval
      setTimeout(renderApp, 100);
    }
  } catch (error) {
    console.error('❌ React mounting error:', error);
    setTimeout(renderApp, 300);
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