
import React, { useState, useEffect } from 'react';
import axios from 'axios';

const ModelTraining = () => {
  const [modelsStatus, setModelsStatus] = useState({});
  const [training, setTraining] = useState(false);
  const [selectedModels, setSelectedModels] = useState([
    'direction', 'magnitude', 'profit_prob', 'volatility', 
    'trend_sideways', 'reversal', 'trading_signal'
  ]);

  useEffect(() => {
    fetchModelsStatus();
  }, []);

  const fetchModelsStatus = async () => {
    try {
      const response = await axios.get('/api/models/status');
      setModelsStatus(response.data.models);
    } catch (error) {
      console.error('Failed to fetch models status:', error);
    }
  };

  const handleTraining = async () => {
    if (selectedModels.length === 0) {
      alert('Please select at least one model to train.');
      return;
    }

    setTraining(true);
    try {
      const response = await axios.post('/api/models/train', selectedModels);
      alert('✅ Model training completed successfully!');
      fetchModelsStatus();
    } catch (error) {
      alert('❌ Training failed: ' + (error.response?.data?.detail || error.message));
    } finally {
      setTraining(false);
    }
  };

  const modelOptions = [
    { id: 'direction', name: 'Direction Prediction', desc: 'Predict if price will go up or down' },
    { id: 'magnitude', name: 'Magnitude Prediction', desc: 'Predict the size of price moves' },
    { id: 'profit_prob', name: 'Profit Probability', desc: 'Predict probability of profitable trades' },
    { id: 'volatility', name: 'Volatility Forecasting', desc: 'Forecast future volatility' },
    { id: 'trend_sideways', name: 'Trend Classification', desc: 'Classify trending vs sideways markets' },
    { id: 'reversal', name: 'Reversal Points', desc: 'Identify potential trend reversals' },
    { id: 'trading_signal', name: 'Trading Signals', desc: 'Generate buy/sell/hold recommendations' }
  ];

  return (
    <div className="container">
      <div className="header">
        <h1>🔬 ML TRAINING LAB</h1>
        <p>Advanced Machine Learning Model Training</p>
      </div>

      {Object.keys(modelsStatus).length > 0 && (
        <div className="card">
          <h3>📊 Existing Models Status</h3>
          <div className="grid grid-3">
            {Object.entries(modelsStatus).map(([modelName, modelInfo]) => (
              <div key={modelName} className="metric-card">
                <h4 style={{color: '#00ffff'}}>{modelInfo.name}</h4>
                <div className="metric-value" style={{color: '#00ff41', fontSize: '1rem'}}>
                  ✅ Trained
                </div>
                <div className="metric-label">
                  {modelInfo.task_type} | {modelInfo.trained_at}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="card">
        <h3>🎯 Models to Train</h3>
        <p style={{marginBottom: '2rem', color: '#b8bcc8'}}>
          Select which prediction models you want to train:
        </p>

        <div className="grid grid-2">
          {modelOptions.map((model) => (
            <div key={model.id} style={{
              padding: '1rem',
              border: selectedModels.includes(model.id) ? '2px solid #00ffff' : '1px solid rgba(0, 255, 255, 0.3)',
              borderRadius: '8px',
              background: selectedModels.includes(model.id) ? 'rgba(0, 255, 255, 0.1)' : 'rgba(0, 255, 255, 0.05)',
              cursor: 'pointer'
            }} onClick={() => {
              if (selectedModels.includes(model.id)) {
                setSelectedModels(selectedModels.filter(m => m !== model.id));
              } else {
                setSelectedModels([...selectedModels, model.id]);
              }
            }}>
              <div style={{display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem'}}>
                <input
                  type="checkbox"
                  checked={selectedModels.includes(model.id)}
                  onChange={() => {}}
                />
                <strong style={{color: '#00ffff'}}>{model.name}</strong>
              </div>
              <p style={{color: '#b8bcc8', fontSize: '0.9rem', margin: 0}}>
                {model.desc}
              </p>
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <h3>⚙️ Training Configuration</h3>
        
        <div className="grid grid-3" style={{marginBottom: '2rem'}}>
          <div>
            <label style={{display: 'block', marginBottom: '0.5rem', color: '#00ffff'}}>
              Training Data Split
            </label>
            <select style={{
              width: '100%',
              padding: '0.5rem',
              background: 'rgba(25, 25, 45, 0.9)',
              border: '1px solid rgba(0, 255, 255, 0.3)',
              borderRadius: '4px',
              color: '#ffffff'
            }}>
              <option value="0.8">80% Training</option>
              <option value="0.7">70% Training</option>
              <option value="0.9">90% Training</option>
            </select>
          </div>

          <div>
            <label style={{display: 'block', marginBottom: '0.5rem', color: '#00ffff'}}>
              XGBoost Max Depth
            </label>
            <select style={{
              width: '100%',
              padding: '0.5rem',
              background: 'rgba(25, 25, 45, 0.9)',
              border: '1px solid rgba(0, 255, 255, 0.3)',
              borderRadius: '4px',
              color: '#ffffff'
            }}>
              <option value="6">6</option>
              <option value="5">5</option>
              <option value="7">7</option>
              <option value="8">8</option>
            </select>
          </div>

          <div>
            <label style={{display: 'block', marginBottom: '0.5rem', color: '#00ffff'}}>
              Number of Estimators
            </label>
            <select style={{
              width: '100%',
              padding: '0.5rem',
              background: 'rgba(25, 25, 45, 0.9)',
              border: '1px solid rgba(0, 255, 255, 0.3)',
              borderRadius: '4px',
              color: '#ffffff'
            }}>
              <option value="100">100</option>
              <option value="150">150</option>
              <option value="200">200</option>
              <option value="50">50</option>
            </select>
          </div>
        </div>

        <button
          className="btn btn-primary"
          onClick={handleTraining}
          disabled={training || selectedModels.length === 0}
          style={{
            opacity: (training || selectedModels.length === 0) ? 0.5 : 1,
            width: '100%',
            padding: '1rem',
            fontSize: '1.1rem'
          }}
        >
          {training ? '🔄 Training Models...' : '🚀 Train All Selected Models'}
        </button>

        {selectedModels.length === 0 && (
          <div className="alert alert-warning" style={{marginTop: '1rem'}}>
            ⚠️ Please select at least one model to train.
          </div>
        )}
      </div>

      <div className="card">
        <h3>ℹ️ Training Information</h3>
        <div style={{color: '#b8bcc8', lineHeight: 1.8}}>
          <p>• Training uses XGBoost algorithms for optimal performance</p>
          <p>• Models are automatically saved to the database after training</p>
          <p>• Training time depends on data size and model complexity</p>
          <p>• Full dataset is used for maximum accuracy</p>
          <p>• Technical indicators are calculated automatically during training</p>
        </div>
      </div>
    </div>
  );
};

export default ModelTraining;
