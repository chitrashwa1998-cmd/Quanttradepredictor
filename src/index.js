import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

// Ensure DOM is ready before mounting
const renderApp = () => {
  const rootElement = document.getElementById('root');
  if (rootElement) {
    const root = ReactDOM.createRoot(rootElement);
    root.render(
      <React.StrictMode>
        <App />
      </React.StrictMode>
    );
    console.log('✅ React app mounted successfully');
  } else {
    console.error('❌ Root element not found - check if public/index.html exists');
    setTimeout(renderApp, 100); // Retry in 100ms
  }
};

// Multiple safety checks for DOM readiness
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', renderApp);
} else if (document.readyState === 'interactive' || document.readyState === 'complete') {
  renderApp();
} else {
  window.addEventListener('load', renderApp);
}