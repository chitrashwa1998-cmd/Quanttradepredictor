/**
 * Predictions page - Complete Streamlit functionality migration with 4 model tabs
 */

import { useState, useEffect, useCallback } from 'react';
import Card from '../components/common/Card';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { dataAPI, predictionsAPI } from '../services/api';

const Predictions = () => {
  const [datasets, setDatasets] = useState([]);
  const [modelStatus, setModelStatus] = useState({});
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('volatility');
  const [predictions, setPredictions] = useState({});
  const [predictionStatus, setPredictionStatus] = useState('');
  const [freshData, setFreshData] = useState(null);
  const [livePredictionsAvailable, setLivePredictionsAvailable] = useState(false);

  // Load initial data
  const loadInitialData = useCallback(async () => {
    try {
      setLoading(true);
      
      // Load datasets and model status in parallel
      const [datasetsResponse, modelsResponse] = await Promise.all([
        dataAPI.getDatasets(),
        predictionsAPI.getModelsStatus()
      ]);

      setDatasets(datasetsResponse.data || []);
      setModelStatus(modelsResponse.data || {});

      // Try to load fresh data from preferred datasets
      const datasetList = datasetsResponse.data || [];
      if (datasetList.length > 0) {
        // Prefer training_dataset or livenifty50
        const preferredDatasets = ['training_dataset', 'livenifty50'];
        let datasetToLoad = null;

        for (const preferred of preferredDatasets) {
          const found = datasetList.find(d => d.name === preferred);
          if (found) {
            datasetToLoad = preferred;
            break;
          }
        }

        // If no preferred dataset, use the first available
        if (!datasetToLoad && datasetList.length > 0) {
          datasetToLoad = datasetList[0].name;
        }

        if (datasetToLoad) {
          const dataResponse = await dataAPI.loadDataset(datasetToLoad);
          setFreshData(dataResponse.data);
          setPredictionStatus(`✅ Using authentic data with ${dataResponse.data?.length || 0} records from ${datasetToLoad}`);
        }
      }

      // Check for live predictions
      try {
        const liveStatus = await predictionsAPI.getLivePredictionStatus();
        if (liveStatus.data?.pipeline_active && liveStatus.data?.instruments_with_predictions > 0) {
          setLivePredictionsAvailable(true);
        }
      } catch (error) {
        // Live predictions not available, which is fine
        setLivePredictionsAvailable(false);
      }

    } catch (error) {
      setPredictionStatus(`❌ Error initializing: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadInitialData();
  }, [loadInitialData]);

  // Generate predictions for a specific model
  const generatePredictions = async (modelType) => {
    if (!freshData || freshData.length === 0) {
      setPredictionStatus('❌ No data available for predictions');
      return;
    }

    if (!modelStatus[modelType]?.loaded) {
      setPredictionStatus(`❌ ${modelType} model not trained. Please train the model first.`);
      return;
    }

    try {
      setLoading(true);
      setPredictionStatus(`🔮 Generating ${modelType} predictions...`);

      const response = await predictionsAPI.generatePredictions({
        model_type: modelType,
        data: freshData.slice(-100) // Use last 100 records for prediction
      });

      setPredictions(prev => ({
        ...prev,
        [modelType]: response.data
      }));

      setPredictionStatus(`✅ ${modelType.charAt(0).toUpperCase() + modelType.slice(1)} predictions generated successfully!`);
    } catch (error) {
      setPredictionStatus(`❌ Prediction failed: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Clear cached data
  const clearCache = async () => {
    try {
      setLoading(true);
      setPredictionStatus('🗑️ Clearing all cached data...');
      
      // Clear local state
      setPredictions({});
      setFreshData(null);
      
      // Reload fresh data
      await loadInitialData();
      
      setPredictionStatus('✅ Cleared all cached data. Page reloaded with fresh database data.');
    } catch (error) {
      setPredictionStatus(`❌ Error clearing cache: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const modelTabs = [
    { 
      id: 'volatility', 
      name: 'Volatility Predictions', 
      icon: '📊', 
      description: 'Forecasts the magnitude of price movements without predicting direction.',
      color: '#00ffff'
    },
    { 
      id: 'direction', 
      name: 'Direction Predictions', 
      icon: '📈', 
      description: 'Predicts whether prices will move up or down.',
      color: '#00ff41'
    },
    { 
      id: 'profit_probability', 
      name: 'Profit Probability', 
      icon: '💰', 
      description: 'Estimates the likelihood of profitable trades with risk assessment.',
      color: '#ff0080'
    },
    { 
      id: 'reversal', 
      name: 'Reversal Detection', 
      icon: '🔄', 
      description: 'Identifies potential market reversal points.',
      color: '#8b5cf6'
    }
  ];

  return (
    <div className="container mx-auto px-6 py-8">
      {/* Header */}
      <div className="trading-header mb-8">
        <h1 style={{
          margin: '0',
          background: 'var(--gradient-primary)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          backgroundClip: 'text',
          fontFamily: 'var(--font-display)',
          fontSize: '2.5rem'
        }}>
          🔮 REAL-TIME PREDICTIONS
        </h1>
        <p style={{
          fontSize: '1.2rem',
          margin: '1rem 0 0 0',
          color: 'rgba(255,255,255,0.8)',
          fontFamily: 'var(--font-primary)'
        }}>
          Advanced ML Model Predictions - Authentic Data Only
        </p>
      </div>

      {/* Live Predictions Banner */}
      {livePredictionsAvailable && (
        <Card style={{ marginBottom: '2rem', background: 'rgba(0, 255, 0, 0.05)', border: '1px solid rgba(0, 255, 0, 0.2)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <h3 style={{ color: '#51cf66', margin: '0 0 0.5rem 0', fontSize: '1.2rem' }}>
                🎯 Live Predictions Available!
              </h3>
              <p style={{ color: 'var(--text-primary)', margin: '0' }}>
                Real-time direction predictions are being generated from live market data.
              </p>
            </div>
            <button
              onClick={() => window.location.href = '/live'}
              style={{
                padding: '0.75rem 1.5rem',
                background: 'var(--gradient-primary)',
                border: 'none',
                borderRadius: '8px',
                color: 'white',
                fontFamily: 'var(--font-primary)',
                fontWeight: '600',
                cursor: 'pointer'
              }}
            >
              📡 View Live Data Page
            </button>
          </div>
        </Card>
      )}

      {/* Controls */}
      <Card style={{ marginBottom: '2rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <h3 style={{ color: 'var(--accent-cyan)', margin: '0 0 0.5rem 0' }}>
              Data Status
            </h3>
            <p style={{ color: 'var(--text-secondary)', margin: '0', fontSize: '0.9rem' }}>
              {freshData ? `${freshData.length} records loaded` : 'No data loaded'}
            </p>
          </div>
          <button
            onClick={clearCache}
            disabled={loading}
            style={{
              padding: '0.75rem 1.5rem',
              background: loading ? 'var(--bg-secondary)' : '#ff6b6b',
              border: '1px solid #ff6b6b',
              borderRadius: '8px',
              color: 'white',
              fontFamily: 'var(--font-primary)',
              fontWeight: '600',
              cursor: loading ? 'not-allowed' : 'pointer'
            }}
            title="Click if you see synthetic datetime warnings"
          >
            {loading ? '⏳ Clearing...' : '🗑️ Clear All Cached Data'}
          </button>
        </div>
      </Card>

      {/* Model Status Overview */}
      <Card style={{ marginBottom: '2rem' }}>
        <h3 style={{ color: 'var(--accent-cyan)', marginBottom: '1rem' }}>
          🤖 Model Status
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {modelTabs.map((tab) => (
            <div key={tab.id} style={{
              background: modelStatus[tab.id]?.loaded 
                ? 'rgba(0, 255, 0, 0.05)' 
                : 'rgba(255, 0, 0, 0.05)',
              border: `1px solid ${modelStatus[tab.id]?.loaded ? '#00ff00' : '#ff0000'}`,
              borderRadius: '8px',
              padding: '1rem',
              textAlign: 'center'
            }}>
              <div style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>{tab.icon}</div>
              <div style={{
                color: modelStatus[tab.id]?.loaded ? '#51cf66' : '#ff6b6b',
                fontWeight: '600',
                fontSize: '0.9rem'
              }}>
                {modelStatus[tab.id]?.loaded ? '✅ Ready' : '❌ Not Trained'}
              </div>
              <div style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginTop: '0.25rem' }}>
                {tab.name}
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Prediction Tabs */}
      <Card>
        {/* Tab Navigation */}
        <div style={{
          display: 'flex',
          borderBottom: '2px solid var(--border)',
          marginBottom: '2rem',
          overflowX: 'auto'
        }}>
          {modelTabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                padding: '1rem 1.5rem',
                background: activeTab === tab.id ? tab.color + '20' : 'transparent',
                border: 'none',
                borderBottom: activeTab === tab.id ? `3px solid ${tab.color}` : '3px solid transparent',
                color: activeTab === tab.id ? tab.color : 'var(--text-primary)',
                fontFamily: 'var(--font-primary)',
                fontWeight: '600',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                whiteSpace: 'nowrap'
              }}
            >
              {tab.icon} {tab.name}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        {modelTabs.map((tab) => (
          activeTab === tab.id && (
            <div key={tab.id}>
              <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                <div className="lg:col-span-3">
                  <h3 style={{
                    color: tab.color,
                    fontFamily: 'var(--font-display)',
                    fontSize: '1.5rem',
                    marginBottom: '1rem'
                  }}>
                    {tab.icon} {tab.name}
                  </h3>
                  <p style={{
                    color: 'var(--text-primary)',
                    marginBottom: '1.5rem',
                    lineHeight: '1.6'
                  }}>
                    {tab.description}
                  </p>

                  {/* Model Status Check */}
                  {!modelStatus[tab.id]?.loaded && (
                    <div style={{
                      background: 'rgba(255, 0, 0, 0.05)',
                      border: '1px solid #ff0000',
                      borderRadius: '8px',
                      padding: '1.5rem',
                      marginBottom: '1.5rem',
                      textAlign: 'center'
                    }}>
                      <div style={{ color: '#ff6b6b', fontSize: '1.2rem', marginBottom: '0.5rem' }}>
                        ⚠️ Model Not Trained
                      </div>
                      <p style={{ color: 'var(--text-secondary)', margin: '0 0 1rem 0' }}>
                        Please train the {tab.name.toLowerCase()} first in the Model Training page.
                      </p>
                      <button
                        onClick={() => window.location.href = '/training'}
                        style={{
                          padding: '0.75rem 1.5rem',
                          background: 'var(--gradient-primary)',
                          border: 'none',
                          borderRadius: '8px',
                          color: 'white',
                          fontFamily: 'var(--font-primary)',
                          fontWeight: '600',
                          cursor: 'pointer'
                        }}
                      >
                        🧠 Go to Model Training
                      </button>
                    </div>
                  )}

                  {/* Predictions Results */}
                  {predictions[tab.id] && modelStatus[tab.id]?.loaded && (
                    <div style={{
                      background: `${tab.color}10`,
                      border: `1px solid ${tab.color}40`,
                      borderRadius: '12px',
                      padding: '1.5rem',
                      marginBottom: '1.5rem'
                    }}>
                      <h4 style={{ color: tab.color, marginBottom: '1rem' }}>
                        📊 Latest {tab.name}
                      </h4>
                      
                      {/* Prediction Visualization */}
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                        {predictions[tab.id].summary && Object.entries(predictions[tab.id].summary).map(([key, value]) => (
                          <div key={key} style={{
                            background: 'rgba(255, 255, 255, 0.05)',
                            borderRadius: '8px',
                            padding: '1rem',
                            textAlign: 'center'
                          }}>
                            <div style={{
                              color: tab.color,
                              fontSize: '1.5rem',
                              fontWeight: '700',
                              marginBottom: '0.25rem'
                            }}>
                              {typeof value === 'number' ? value.toFixed(4) : value}
                            </div>
                            <div style={{
                              color: 'var(--text-secondary)',
                              fontSize: '0.9rem',
                              textTransform: 'capitalize'
                            }}>
                              {key.replace('_', ' ')}
                            </div>
                          </div>
                        ))}
                      </div>

                      {/* Prediction Table */}
                      {predictions[tab.id].data && (
                        <div style={{ overflowX: 'auto' }}>
                          <table style={{
                            width: '100%',
                            borderCollapse: 'collapse',
                            fontFamily: 'var(--font-mono)',
                            fontSize: '0.9rem'
                          }}>
                            <thead>
                              <tr style={{ background: `${tab.color}20` }}>
                                <th style={{
                                  padding: '0.75rem',
                                  textAlign: 'left',
                                  color: tab.color,
                                  borderBottom: '1px solid var(--border)'
                                }}>
                                  Timestamp
                                </th>
                                <th style={{
                                  padding: '0.75rem',
                                  textAlign: 'left',
                                  color: tab.color,
                                  borderBottom: '1px solid var(--border)'
                                }}>
                                  Prediction
                                </th>
                                <th style={{
                                  padding: '0.75rem',
                                  textAlign: 'left',
                                  color: tab.color,
                                  borderBottom: '1px solid var(--border)'
                                }}>
                                  Confidence
                                </th>
                              </tr>
                            </thead>
                            <tbody>
                              {predictions[tab.id].data.slice(0, 10).map((row, index) => (
                                <tr key={index}>
                                  <td style={{
                                    padding: '0.75rem',
                                    color: 'var(--text-primary)',
                                    borderBottom: '1px solid rgba(255, 255, 255, 0.1)'
                                  }}>
                                    {row.timestamp}
                                  </td>
                                  <td style={{
                                    padding: '0.75rem',
                                    color: tab.color,
                                    borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
                                    fontWeight: '600'
                                  }}>
                                    {row.prediction}
                                  </td>
                                  <td style={{
                                    padding: '0.75rem',
                                    color: 'var(--text-primary)',
                                    borderBottom: '1px solid rgba(255, 255, 255, 0.1)'
                                  }}>
                                    {row.confidence}
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                <div>
                  <button
                    onClick={() => generatePredictions(tab.id)}
                    disabled={loading || !modelStatus[tab.id]?.loaded || !freshData}
                    style={{
                      width: '100%',
                      padding: '1rem',
                      background: loading || !modelStatus[tab.id]?.loaded || !freshData
                        ? 'var(--bg-secondary)' 
                        : tab.color,
                      border: `2px solid ${tab.color}`,
                      borderRadius: '8px',
                      color: 'white',
                      fontFamily: 'var(--font-primary)',
                      fontWeight: '600',
                      fontSize: '1rem',
                      cursor: loading || !modelStatus[tab.id]?.loaded || !freshData ? 'not-allowed' : 'pointer',
                      transition: 'all 0.3s ease'
                    }}
                  >
                    {loading ? '⏳ Generating...' : `🔮 Generate ${tab.name}`}
                  </button>

                  {predictions[tab.id] && (
                    <div style={{
                      marginTop: '1rem',
                      padding: '1rem',
                      background: `${tab.color}10`,
                      border: `1px solid ${tab.color}40`,
                      borderRadius: '8px',
                      textAlign: 'center'
                    }}>
                      <div style={{
                        color: tab.color,
                        fontSize: '1.5rem',
                        marginBottom: '0.5rem'
                      }}>
                        ✅
                      </div>
                      <div style={{
                        color: tab.color,
                        fontSize: '0.9rem',
                        fontWeight: '600'
                      }}>
                        Predictions Generated
                      </div>
                      <div style={{
                        color: 'var(--text-secondary)',
                        fontSize: '0.8rem',
                        marginTop: '0.25rem'
                      }}>
                        {predictions[tab.id].data?.length || 0} predictions
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )
        ))}
      </Card>

      {/* Status Message */}
      {predictionStatus && (
        <div style={{
          position: 'fixed',
          bottom: '2rem',
          right: '2rem',
          padding: '1rem 1.5rem',
          background: predictionStatus.includes('❌') 
            ? 'rgba(255, 0, 0, 0.9)' 
            : predictionStatus.includes('✅')
            ? 'rgba(0, 255, 0, 0.9)'
            : 'rgba(0, 255, 255, 0.9)',
          border: `1px solid ${
            predictionStatus.includes('❌') ? '#ff0000' : 
            predictionStatus.includes('✅') ? '#00ff00' : '#00ffff'
          }`,
          borderRadius: '8px',
          color: 'white',
          fontFamily: 'var(--font-primary)',
          fontWeight: '600',
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
          zIndex: 1000,
          maxWidth: '400px'
        }}>
          {predictionStatus}
        </div>
      )}
    </div>
  );
};

export default Predictions;