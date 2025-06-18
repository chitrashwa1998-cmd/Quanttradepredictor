
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import DataUpload from './pages/DataUpload';
import ModelTraining from './pages/ModelTraining';
import Predictions from './pages/Predictions';
import Backtesting from './pages/Backtesting';
import RealtimeData from './pages/RealtimeData';
import DatabaseManager from './pages/DatabaseManager';

function App() {
  return (
    <Router>
      <div className="App">
        <div className="app-layout">
          <Navbar />
          <main className="main-content">
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/data-upload" element={<DataUpload />} />
              <Route path="/model-training" element={<ModelTraining />} />
              <Route path="/predictions" element={<Predictions />} />
              <Route path="/backtesting" element={<Backtesting />} />
              <Route path="/realtime-data" element={<RealtimeData />} />
              <Route path="/database" element={<DatabaseManager />} />
            </Routes>
          </main>
        </div>
      </div>
    </Router>
  );
}

export default App;
