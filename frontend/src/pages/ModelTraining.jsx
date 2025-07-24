/**
 * Model Training page - Complete Streamlit functionality migration
 */

import { useState, useEffect, useCallback } from 'react';
import Card from '../components/common/Card';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { dataAPI, modelsAPI } from '../services/api';

const ModelTraining = () => {
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState('');
  const [currentData, setCurrentData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('volatility');
  const [trainingConfig, setTrainingConfig] = useState({
    train_split: 0.8,
    max_depth: 6,
    n_estimators: 100
  });
  const [trainingResults, setTrainingResults] = useState({});
  const [trainingStatus, setTrainingStatus] = useState('');
  const [featuresCalculated, setFeaturesCalculated] = useState(false);

  // Load datasets on component mount
  const loadDatasets = useCallback(async () => {
    try {
      setLoading(true);
      const response = await dataAPI.getDatasets();
      const datasetList = response.data || [];
      setDatasets(datasetList);
      
      // Auto-select training_dataset if available
      const trainingDataset = datasetList.find(d => d.name === 'training_dataset');
      if (trainingDataset) {
        setSelectedDataset('training_dataset');
        await loadDataset('training_dataset');
      } else if (datasetList.length > 0) {
        setSelectedDataset(datasetList[0].name);
        await loadDataset(datasetList[0].name);
      }
    } catch (error) {
      console.error('Error loading datasets:', error);
      setDatasets([]);
    } finally {
      setLoading(false);
    }
  }, []);

  // Load specific dataset
  const loadDataset = async (datasetName) => {
    try {
      setLoading(true);
      const response = await dataAPI.loadDataset(datasetName);
      setCurrentData(response.data);
      setTrainingStatus(`✅ Loaded ${datasetName}: ${response.data?.length || 0} rows`);
      setFeaturesCalculated(false);
    } catch (error) {
      setTrainingStatus(`❌ Failed to load ${datasetName}: ${error.response?.data?.detail || error.message}`);
      setCurrentData(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDatasets();
  }, [loadDatasets]);

  // Handle dataset selection
  const handleDatasetChange = (datasetName) => {
    setSelectedDataset(datasetName);
    loadDataset(datasetName);
  };

  // Calculate technical indicators
  const calculateFeatures = async () => {
    if (!currentData) {
      setTrainingStatus('❌ No data loaded. Please select a dataset first.');
      return;
    }

    try {
      setLoading(true);
      setTrainingStatus('🔧 Calculating technical indicators...');
      
      const response = await modelsAPI.calculateFeatures(selectedDataset);
      setFeaturesCalculated(true);
      setTrainingStatus('✅ Technical indicators calculated successfully!');
    } catch (error) {
      setTrainingStatus(`❌ Error calculating features: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Train specific model
  const trainModel = async (modelType) => {
    if (!featuresCalculated) {
      setTrainingStatus('❌ Please calculate features first.');
      return;
    }

    try {
      setLoading(true);
      setTrainingStatus(`🚀 Training ${modelType} model...`);

      const response = await modelsAPI.trainModel({
        model_type: modelType,
        dataset_name: selectedDataset,
        config: trainingConfig
      });

      setTrainingResults(prev => ({
        ...prev,
        [modelType]: response.data
      }));

      setTrainingStatus(`✅ ${modelType.charAt(0).toUpperCase() + modelType.slice(1)} model trained successfully!`);
    } catch (error) {
      setTrainingStatus(`❌ Training failed: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const modelTabs = [
    { id: 'volatility', name: 'Volatility Model', icon: '📈', description: 'Predicts future market volatility using technical indicators.' },
    { id: 'direction', name: 'Direction Model', icon: '🎯', description: 'Predicts whether prices will move up or down.' },
    { id: 'profit_probability', name: 'Profit Probability Model', icon: '💰', description: 'Estimates the likelihood of profitable trades.' },
    { id: 'reversal', name: 'Reversal Model', icon: '🔄', description: 'Identifies potential market reversal points.' }
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
          🧠 MODEL TRAINING CENTER
        </h1>
        <p style={{
          fontSize: '1.2rem',
          margin: '1rem 0 0 0',
          color: 'rgba(255,255,255,0.8)',
          fontFamily: 'var(--font-primary)'
        }}>
          Train prediction models using your processed data
        </p>
      </div>

      {/* Dataset Selection */}
      <Card style={{ marginBottom: '2rem' }}>
        <h2 style={{
          color: 'var(--accent-cyan)',
          fontFamily: 'var(--font-display)',
          fontSize: '1.5rem',
          marginBottom: '1.5rem'
        }}>
          📊 Dataset Selection
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div>
            <label style={{
              display: 'block',
              color: 'var(--text-primary)',
              marginBottom: '0.5rem',
              fontWeight: '500'
            }}>
              Select Dataset for Training:
            </label>
            <select
              value={selectedDataset}
              onChange={(e) => handleDatasetChange(e.target.value)}
              style={{
                width: '100%',
                padding: '0.75rem',
                background: 'var(--bg-secondary)',
                border: '2px solid var(--border)',
                borderRadius: '8px',
                color: 'var(--text-primary)',
                fontFamily: 'var(--font-primary)'
              }}
            >
              <option value="">Select a dataset...</option>
              {datasets.map((dataset, index) => (
                <option key={index} value={dataset.name}>
                  {dataset.name} ({dataset.rows || 0} rows)
                </option>
              ))}
            </select>
          </div>
          <div>
            <button
              onClick={() => selectedDataset && loadDataset(selectedDataset)}
              disabled={loading || !selectedDataset}
              style={{
                padding: '0.75rem 1.5rem',
                background: loading || !selectedDataset 
                  ? 'var(--bg-secondary)' 
                  : 'var(--gradient-primary)',
                border: '2px solid var(--border-hover)',
                borderRadius: '8px',
                color: 'white',
                fontFamily: 'var(--font-primary)',
                fontWeight: '600',
                cursor: loading || !selectedDataset ? 'not-allowed' : 'pointer',
                marginTop: '1.5rem'
              }}
            >
              🔄 Load Selected Dataset
            </button>
          </div>
        </div>

        {currentData && (
          <div style={{
            background: 'rgba(0, 255, 255, 0.05)',
            border: '1px solid rgba(0, 255, 255, 0.2)',
            borderRadius: '8px',
            padding: '1rem',
            marginTop: '1rem'
          }}>
            <p style={{ color: 'var(--accent-cyan)', margin: '0' }}>
              📈 Current dataset: {currentData.length || 0} rows loaded
            </p>
          </div>
        )}
      </Card>

      {/* Training Configuration */}
      <Card style={{ marginBottom: '2rem' }}>
        <h2 style={{
          color: 'var(--accent-cyan)',
          fontFamily: 'var(--font-display)',
          fontSize: '1.5rem',
          marginBottom: '1.5rem'
        }}>
          ⚙️ Training Configuration
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label style={{
              display: 'block',
              color: 'var(--text-primary)',
              marginBottom: '0.5rem',
              fontWeight: '500'
            }}>
              Training Split: {(trainingConfig.train_split * 100).toFixed(0)}%
            </label>
            <input
              type="range"
              min="0.6"
              max="0.9"
              step="0.05"
              value={trainingConfig.train_split}
              onChange={(e) => setTrainingConfig(prev => ({
                ...prev,
                train_split: parseFloat(e.target.value)
              }))}
              style={{
                width: '100%',
                accentColor: 'var(--accent-cyan)'
              }}
            />
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              fontSize: '0.8rem',
              color: 'var(--text-secondary)',
              marginTop: '0.25rem'
            }}>
              <span>Training: {(trainingConfig.train_split * 100).toFixed(0)}%</span>
              <span>Testing: {((1 - trainingConfig.train_split) * 100).toFixed(0)}%</span>
            </div>
          </div>

          <div>
            <label style={{
              display: 'block',
              color: 'var(--text-primary)',
              marginBottom: '0.5rem',
              fontWeight: '500'
            }}>
              Max Depth:
            </label>
            <select
              value={trainingConfig.max_depth}
              onChange={(e) => setTrainingConfig(prev => ({
                ...prev,
                max_depth: parseInt(e.target.value)
              }))}
              style={{
                width: '100%',
                padding: '0.5rem',
                background: 'var(--bg-secondary)',
                border: '2px solid var(--border)',
                borderRadius: '8px',
                color: 'var(--text-primary)',
                fontFamily: 'var(--font-primary)'
              }}
            >
              {[4, 6, 8, 10, 12].map(depth => (
                <option key={depth} value={depth}>{depth}</option>
              ))}
            </select>
          </div>

          <div>
            <label style={{
              display: 'block',
              color: 'var(--text-primary)',
              marginBottom: '0.5rem',
              fontWeight: '500'
            }}>
              Number of Estimators:
            </label>
            <select
              value={trainingConfig.n_estimators}
              onChange={(e) => setTrainingConfig(prev => ({
                ...prev,
                n_estimators: parseInt(e.target.value)
              }))}
              style={{
                width: '100%',
                padding: '0.5rem',
                background: 'var(--bg-secondary)',
                border: '2px solid var(--border)',
                borderRadius: '8px',
                color: 'var(--text-primary)',
                fontFamily: 'var(--font-primary)'
              }}
            >
              {[50, 100, 150, 200, 250, 300, 350, 400, 450, 500].map(est => (
                <option key={est} value={est}>{est}</option>
              ))}
            </select>
          </div>
        </div>
      </Card>

      {/* Feature Engineering */}
      <Card style={{ marginBottom: '2rem' }}>
        <h2 style={{
          color: 'var(--accent-cyan)',
          fontFamily: 'var(--font-display)',
          fontSize: '1.5rem',
          marginBottom: '1.5rem'
        }}>
          🔧 Feature Engineering
        </h2>

        {!featuresCalculated ? (
          <div style={{ textAlign: 'center', padding: '2rem' }}>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>
              ⚠️ Technical indicators not calculated yet.
            </p>
            <button
              onClick={calculateFeatures}
              disabled={loading || !currentData}
              style={{
                padding: '1rem 2rem',
                background: loading || !currentData 
                  ? 'var(--bg-secondary)' 
                  : 'var(--gradient-primary)',
                border: '2px solid var(--border-hover)',
                borderRadius: '8px',
                color: 'white',
                fontFamily: 'var(--font-primary)',
                fontWeight: '600',
                fontSize: '1rem',
                cursor: loading || !currentData ? 'not-allowed' : 'pointer'
              }}
            >
              {loading ? '⏳ Calculating...' : '🔧 Calculate Technical Indicators'}
            </button>
          </div>
        ) : (
          <div style={{
            background: 'rgba(0, 255, 0, 0.05)',
            border: '1px solid rgba(0, 255, 0, 0.2)',
            borderRadius: '8px',
            padding: '1rem',
            textAlign: 'center'
          }}>
            <p style={{ color: '#51cf66', margin: '0', fontWeight: '600' }}>
              ✅ Technical indicators calculated and ready for training
            </p>
          </div>
        )}
      </Card>

      {/* Model Training Tabs */}
      <Card>
        <h2 style={{
          color: 'var(--accent-cyan)',
          fontFamily: 'var(--font-display)',
          fontSize: '1.5rem',
          marginBottom: '1.5rem'
        }}>
          🎯 Model Selection
        </h2>

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
                background: activeTab === tab.id ? 'var(--gradient-primary)' : 'transparent',
                border: 'none',
                borderBottom: activeTab === tab.id ? '3px solid var(--accent-cyan)' : '3px solid transparent',
                color: activeTab === tab.id ? 'white' : 'var(--text-primary)',
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
                    color: 'var(--accent-gold)',
                    fontFamily: 'var(--font-display)',
                    fontSize: '1.3rem',
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

                  {/* Training Results */}
                  {trainingResults[tab.id] && (
                    <div style={{
                      background: 'rgba(0, 255, 0, 0.05)',
                      border: '1px solid rgba(0, 255, 0, 0.2)',
                      borderRadius: '12px',
                      padding: '1.5rem',
                      marginBottom: '1.5rem'
                    }}>
                      <h4 style={{ color: '#51cf66', marginBottom: '1rem' }}>
                        📊 Training Results
                      </h4>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        {trainingResults[tab.id].metrics && Object.entries(trainingResults[tab.id].metrics).map(([key, value]) => (
                          <div key={key} style={{ textAlign: 'center' }}>
                            <div style={{
                              color: 'var(--accent-gold)',
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
                    </div>
                  )}
                </div>

                <div>
                  <button
                    onClick={() => trainModel(tab.id)}
                    disabled={loading || !featuresCalculated}
                    style={{
                      width: '100%',
                      padding: '1rem',
                      background: loading || !featuresCalculated 
                        ? 'var(--bg-secondary)' 
                        : 'var(--gradient-primary)',
                      border: '2px solid var(--border-hover)',
                      borderRadius: '8px',
                      color: 'white',
                      fontFamily: 'var(--font-primary)',
                      fontWeight: '600',
                      fontSize: '1rem',
                      cursor: loading || !featuresCalculated ? 'not-allowed' : 'pointer',
                      transition: 'all 0.3s ease'
                    }}
                  >
                    {loading ? '⏳ Training...' : `🚀 Train ${tab.name}`}
                  </button>

                  {trainingResults[tab.id] && (
                    <div style={{
                      marginTop: '1rem',
                      padding: '1rem',
                      background: 'rgba(0, 255, 255, 0.05)',
                      border: '1px solid rgba(0, 255, 255, 0.2)',
                      borderRadius: '8px',
                      textAlign: 'center'
                    }}>
                      <div style={{
                        color: '#51cf66',
                        fontSize: '1.5rem',
                        marginBottom: '0.5rem'
                      }}>
                        ✅
                      </div>
                      <div style={{
                        color: '#51cf66',
                        fontSize: '0.9rem',
                        fontWeight: '600'
                      }}>
                        Model Trained Successfully
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
      {trainingStatus && (
        <div style={{
          position: 'fixed',
          bottom: '2rem',
          right: '2rem',
          padding: '1rem 1.5rem',
          background: trainingStatus.includes('❌') 
            ? 'rgba(255, 0, 0, 0.9)' 
            : trainingStatus.includes('✅')
            ? 'rgba(0, 255, 0, 0.9)'
            : 'rgba(0, 255, 255, 0.9)',
          border: `1px solid ${
            trainingStatus.includes('❌') ? '#ff0000' : 
            trainingStatus.includes('✅') ? '#00ff00' : '#00ffff'
          }`,
          borderRadius: '8px',
          color: 'white',
          fontFamily: 'var(--font-primary)',
          fontWeight: '600',
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
          zIndex: 1000,
          maxWidth: '400px'
        }}>
          {trainingStatus}
        </div>
      )}
    </div>
  );
};

export default ModelTraining;