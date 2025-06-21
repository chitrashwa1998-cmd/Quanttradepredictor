import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { API_BASE_URL } from '../config/api';

const ModelTraining = () => {
  const [modelsStatus, setModelsStatus] = useState({});
  const [training, setTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState({});
  const [trainingResults, setTrainingResults] = useState({});
  const [selectedModels, setSelectedModels] = useState([
    'direction', 'magnitude', 'profit_prob', 'volatility', 
    'trend_sideways', 'reversal', 'trading_signal'
  ]);

  useEffect(() => {
    fetchModelsStatus();
  }, []);

  const fetchModelsStatus = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/models/status`);
      setModelsStatus(response.data.data.trained_models || {});
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
    setTrainingProgress({});
    setTrainingResults({});

    // Initialize progress for selected models
    const initialProgress = {};
    const initialResults = {};
    selectedModels.forEach(model => {
      initialProgress[model] = { status: 'waiting', progress: 0 };
      initialResults[model] = null;
    });
    setTrainingProgress(initialProgress);
    setTrainingResults(initialResults);

    try {
      // Simulate progress updates during training
      const progressInterval = setInterval(() => {
        setTrainingProgress(prev => {
          const updated = { ...prev };
          selectedModels.forEach(model => {
            if (updated[model].status === 'waiting') {
              updated[model] = { status: 'training', progress: 10 };
            } else if (updated[model].status === 'training' && updated[model].progress < 90) {
              updated[model].progress = Math.min(updated[model].progress + 10, 90);
            }
          });
          return updated;
        });
      }, 2000);

      const response = await axios.post(`${API_BASE_URL}/train-models`, { models: selectedModels });
      
      clearInterval(progressInterval);

      // Update final results
      if (response.data.success && response.data.data.results) {
        const results = response.data.data.results;
        const finalProgress = {};
        const finalResults = {};

        selectedModels.forEach(model => {
          if (results[model] && results[model].status === 'success') {
            finalProgress[model] = { status: 'completed', progress: 100 };
            finalResults[model] = {
              status: 'success',
              accuracy: results[model].accuracy,
              task_type: results[model].task_type
            };
          } else {
            finalProgress[model] = { status: 'failed', progress: 0 };
            finalResults[model] = {
              status: 'failed',
              error: results[model]?.error || 'Unknown error'
            };
          }
        });

        setTrainingProgress(finalProgress);
        setTrainingResults(finalResults);
      }

      fetchModelsStatus();
    } catch (error) {
      // Clear progress interval on error
      setTrainingProgress(prev => {
        const updated = { ...prev };
        selectedModels.forEach(model => {
          updated[model] = { status: 'failed', progress: 0 };
        });
        return updated;
      });
      
      alert('❌ Training failed: ' + (error.response?.data?.error || error.message));
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

        {/* Training Progress Section */}
        {training && Object.keys(trainingProgress).length > 0 && (
          <div style={{marginTop: '2rem'}}>
            <h4 style={{color: '#00ffff', marginBottom: '1rem'}}>Training Progress</h4>
            {Object.entries(trainingProgress).map(([modelName, progress]) => {
              const modelInfo = modelOptions.find(m => m.id === modelName);
              return (
                <div key={modelName} style={{
                  marginBottom: '1rem',
                  padding: '1rem',
                  border: '1px solid rgba(0, 255, 255, 0.3)',
                  borderRadius: '8px',
                  background: 'rgba(0, 255, 255, 0.05)'
                }}>
                  <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem'}}>
                    <span style={{color: '#00ffff', fontWeight: 'bold'}}>
                      {modelInfo?.name || modelName}
                    </span>
                    <span style={{
                      color: progress.status === 'completed' ? '#00ff41' : 
                             progress.status === 'failed' ? '#ff4444' : 
                             progress.status === 'training' ? '#ffaa00' : '#888888'
                    }}>
                      {progress.status === 'waiting' ? '⏳ Waiting' :
                       progress.status === 'training' ? '🔄 Training' :
                       progress.status === 'completed' ? '✅ Completed' :
                       progress.status === 'failed' ? '❌ Failed' : ''}
                    </span>
                  </div>
                  <div style={{
                    width: '100%',
                    height: '8px',
                    background: 'rgba(255, 255, 255, 0.1)',
                    borderRadius: '4px',
                    overflow: 'hidden'
                  }}>
                    <div style={{
                      width: `${progress.progress}%`,
                      height: '100%',
                      background: progress.status === 'completed' ? '#00ff41' :
                                 progress.status === 'failed' ? '#ff4444' :
                                 progress.status === 'training' ? '#ffaa00' : '#888888',
                      transition: 'width 0.3s ease'
                    }}></div>
                  </div>
                  <div style={{fontSize: '0.8rem', color: '#b8bcc8', marginTop: '0.25rem'}}>
                    {progress.progress}%
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* Training Summary */}
        {!training && Object.keys(trainingResults).length > 0 && (
          <div style={{marginTop: '2rem'}}>
            <div className="grid grid-3" style={{marginBottom: '2rem'}}>
              <div className="metric-card">
                <div className="metric-value" style={{color: '#00ff41'}}>
                  {Object.values(trainingResults).filter(r => r?.status === 'success').length}
                </div>
                <div className="metric-label">Successful</div>
              </div>
              <div className="metric-card">
                <div className="metric-value" style={{color: '#ff4444'}}>
                  {Object.values(trainingResults).filter(r => r?.status === 'failed').length}
                </div>
                <div className="metric-label">Failed</div>
              </div>
              <div className="metric-card">
                <div className="metric-value" style={{color: '#00ffff'}}>
                  {Object.keys(trainingResults).length}
                </div>
                <div className="metric-label">Total Models</div>
              </div>
            </div>
            <h4 style={{color: '#00ffff', marginBottom: '1rem'}}>Detailed Results</h4>
            {Object.entries(trainingResults).map(([modelName, result]) => {
              const modelInfo = modelOptions.find(m => m.id === modelName);
              return (
                <div key={modelName} style={{
                  marginBottom: '1rem',
                  padding: '1rem',
                  border: `1px solid ${result?.status === 'success' ? 'rgba(0, 255, 65, 0.3)' : 'rgba(255, 68, 68, 0.3)'}`,
                  borderRadius: '8px',
                  background: result?.status === 'success' ? 'rgba(0, 255, 65, 0.05)' : 'rgba(255, 68, 68, 0.05)'
                }}>
                  <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
                    <span style={{color: '#00ffff', fontWeight: 'bold'}}>
                      {modelInfo?.name || modelName}
                    </span>
                    <span style={{
                      color: result?.status === 'success' ? '#00ff41' : '#ff4444'
                    }}>
                      {result?.status === 'success' ? '✅ Success' : '❌ Failed'}
                    </span>
                  </div>
                  {result?.status === 'success' && (
                    <div style={{color: '#b8bcc8', fontSize: '0.9rem', marginTop: '0.5rem'}}>
                      {result.task_type === 'classification' ? 
                        `Accuracy: ${(result.accuracy * 100).toFixed(1)}%` :
                        `RMSE: ${result.accuracy.toFixed(4)}`
                      }
                    </div>
                  )}
                  {result?.status === 'failed' && (
                    <div style={{color: '#ff4444', fontSize: '0.9rem', marginTop: '0.5rem'}}>
                      Error: {result.error}
                    </div>
                  )}
                </div>
              );
            })}
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