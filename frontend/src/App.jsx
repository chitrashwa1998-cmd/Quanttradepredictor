/**
 * Main App component for TribexAlpha Trading Platform
 * React + FastAPI architecture
 */

import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout/Layout';
import ErrorBoundary from './components/common/ErrorBoundary';

// Import page components
import Home from './pages/Home';
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
            <Route path="/" element={<Home />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/upload" element={<DataUpload />} />
            <Route path="/training" element={<ModelTraining />} />
            <Route path="/predictions" element={<Predictions />} />
            <Route path="/live" element={<LiveTrading />} />
            <Route path="/backtesting" element={<Backtesting />} />
            <Route path="/database" element={<DatabaseManager />} />
          </Routes>
        </Layout>
      </Router>
    </ErrorBoundary>
  );
}

export default App;
