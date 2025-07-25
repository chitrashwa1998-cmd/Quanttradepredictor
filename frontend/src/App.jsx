/**
 * Main App component for TribexAlpha Trading Platform
 * React + FastAPI architecture
 */

import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout/Layout';
import ErrorBoundary from './components/common/ErrorBoundary';

// Import page components
import Home from './pages/Home';
import SafeHome from './pages/SafeHome';
import Dashboard from './pages/Dashboard';
import DataUpload from './pages/DataUpload';
import ModelTraining from './pages/ModelTraining';
import Predictions from './pages/Predictions';
import LiveTrading from './pages/LiveTrading';
import Backtesting from './pages/Backtesting';
import DatabaseManager from './pages/DatabaseManager';

function App() {
  return (
    <ErrorBoundary>
      <Router>
        <Layout>
          <Routes>
            <Route path="/" element={<SafeHome />} />
            <Route path="/dashboard" element={<ErrorBoundary><Dashboard /></ErrorBoundary>} />
            <Route path="/upload" element={<ErrorBoundary><DataUpload /></ErrorBoundary>} />
            <Route path="/training" element={<ErrorBoundary><ModelTraining /></ErrorBoundary>} />
            <Route path="/predictions" element={<ErrorBoundary><Predictions /></ErrorBoundary>} />
            <Route path="/live" element={<ErrorBoundary><LiveTrading /></ErrorBoundary>} />
            <Route path="/backtesting" element={<ErrorBoundary><Backtesting /></ErrorBoundary>} />
            <Route path="/database" element={<ErrorBoundary><DatabaseManager /></ErrorBoundary>} />
          </Routes>
        </Layout>
      </Router>
    </ErrorBoundary>
  );
}

export default App;
