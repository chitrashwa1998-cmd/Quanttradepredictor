
/**
 * Model Training page - Complete Streamlit UI replication
 */

import { useState, useEffect } from 'react';
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
  const [trainingStatus, setTrainingStatus] = useState('');
  const [trainingResults, setTrainingResults] = useState({});
  const [featuresCalculated, setFeaturesCalculated] = useState({
    volatility: false,
    direction: false,
    profit_probability: false,
    reversal: false
  });
  const [featureStatus, setFeatureStatus] = useState('');

  // Load datasets on component mount
  useEffect(() => {
    loadDatasets();
  }, []);

  // Load datasets
  const loadDatasets = async () => {
    try {
      setLoading(true);
      const response = await dataAPI.getDatasets();
      
      if (response?.data?.success && response.data.datasets) {
        const datasetList = response.data.datasets;
        setDatasets(datasetList);

        // Auto-select training_dataset if available, otherwise main_dataset, otherwise first
        const trainingDataset = datasetList.find(d => d.name === 'training_dataset');
        const mainDataset = datasetList.find(d => d.name === 'main_dataset');
        
        if (trainingDataset) {
          setSelectedDataset('training_dataset');
        } else if (mainDataset) {
          setSelectedDataset('main_dataset');
        } else if (datasetList.length > 0) {
          setSelectedDataset(datasetList[0].name);
        }
      } else {
        setTrainingStatus('❌ Failed to load datasets');
      }
    } catch (error) {
      console.error('Error loading datasets:', error);
      setTrainingStatus(`❌ Error loading datasets: ${error?.message || 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  };

  // Load selected dataset
  const loadSelectedDataset = async () => {
    if (!selectedDataset) {
      setTrainingStatus('❌ Please select a dataset first');
      return;
    }

    try {
      setLoading(true);
      setTrainingStatus(`🔄 Loading ${selectedDataset}...`);
      
      const response = await dataAPI.getDataset(selectedDataset);
      if (response?.data?.success && response.data.data) {
        setCurrentData(response.data.data);
        setTrainingStatus(`✅ Loaded ${selectedDataset}: ${response.data.data.length} rows`);
      } else {
        setTrainingStatus(`❌ Failed to load ${selectedDataset}`);
      }
    } catch (error) {
      console.error('Error loading dataset:', error);
      setTrainingStatus(`❌ Error loading dataset: ${error?.message || 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  };

  // Calculate features for specific model
  const calculateFeatures = async (modelType) => {
    if (!selectedDataset) {
      setFeatureStatus('❌ Please select a dataset first');
      return;
    }

    try {
      setLoading(true);
      setFeatureStatus(`🔧 Calculating ${modelType} features...`);

      // Call backend API to calculate features
      const response = await modelsAPI.calculateFeatures({
        dataset_name: selectedDataset,
        model_type: modelType
      });

      if (response?.success) {
        setFeaturesCalculated(prev => ({
          ...prev,
          [modelType]: true
        }));
        setFeatureStatus(`✅ ${modelType} features calculated successfully!`);
      } else {
        setFeatureStatus(`❌ Failed to calculate ${modelType} features: ${response?.message || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Feature calculation error:', error);
      setFeatureStatus(`❌ Error calculating features: ${error?.response?.data?.detail || error?.message || 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  };

  // Train model
  const trainModel = async () => {
    if (!selectedDataset) {
      setTrainingStatus('❌ Please select a dataset first');
      return;
    }

    if (!featuresCalculated[activeTab]) {
      setTrainingStatus('❌ Please calculate features first');
      return;
    }

    try {
      setLoading(true);
      setTrainingStatus(`🚀 Training ${activeTab} model...`);

      const response = await modelsAPI.trainModel({
        dataset_name: selectedDataset,
        model_type: activeTab,
        config: trainingConfig
      });

      if (response?.success) {
        setTrainingResults(prev => ({
          ...prev,
          [activeTab]: response
        }));
        setTrainingStatus(`✅ ${activeTab} model trained successfully!`);
      } else {
        setTrainingStatus(`❌ Failed to train ${activeTab} model: ${response?.message || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Training error:', error);
      setTrainingStatus(`❌ Training error: ${error?.response?.data?.detail || error?.message || 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  };

  const modelTabs = [
    { 
      id: 'volatility', 
      name: 'Volatility Model', 
      icon: '📈',
      description: 'Predicts future market volatility using technical indicators.'
    },
    { 
      id: 'direction', 
      name: 'Direction Model', 
      icon: '🎯',
      description: 'Predicts whether price will move up or down.'
    },
    { 
      id: 'profit_probability', 
      name: 'Profit Probability Model', 
      icon: '💰',
      description: 'Predicts the likelihood of profitable trades within the next 5 periods.'
    },
    { 
      id: 'reversal', 
      name: 'Reversal Model', 
      icon: '🔄',
      description: 'Predicts market reversal points using specialized technical indicators.'
    }
  ];

  const currentModel = modelTabs.find(tab => tab.id === activeTab);

  return (
    <div style={{ backgroundColor: '#0a0a0f', minHeight: '100vh', color: '#ffffff', fontFamily: 'Space Grotesk, sans-serif' }}>
      <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '2rem' }}>
        
        {/* Header */}
        <div className="trading-header" style={{
          background: 'linear-gradient(145deg, rgba(25, 25, 45, 0.9), rgba(35, 35, 55, 0.6))',
          border: '2px solid rgba(0, 255, 255, 0.2)',
          borderRadius: '20px',
          padding: '2rem',
          marginBottom: '2rem',
          boxShadow: '0 0 20px rgba(0, 255, 255, 0.1)'
        }}>
          <h1 style={{ margin: 0, fontSize: '2.5rem', fontFamily: 'Orbitron, monospace' }}>
            🧠 MODEL TRAINING
          </h1>
          <p style={{ fontSize: '1.2rem', margin: '1rem 0 0 0', color: 'rgba(255,255,255,0.8)' }}>
            Train prediction models using your processed data.
          </p>
        </div>

        {/* Status Messages */}
        {(trainingStatus || featureStatus) && (
          <div style={{
            padding: '1rem',
            borderRadius: '8px',
            marginBottom: '2rem',
            backgroundColor: (trainingStatus?.includes('✅') || featureStatus?.includes('✅')) ? 'rgba(0, 255, 65, 0.1)' : 
                           (trainingStatus?.includes('❌') || featureStatus?.includes('❌')) ? 'rgba(255, 0, 128, 0.1)' : 'rgba(0, 255, 255, 0.1)',
            border: `1px solid ${(trainingStatus?.includes('✅') || featureStatus?.includes('✅')) ? 'rgba(0, 255, 65, 0.3)' : 
                              (trainingStatus?.includes('❌') || featureStatus?.includes('❌')) ? 'rgba(255, 0, 128, 0.3)' : 'rgba(0, 255, 255, 0.3)'}`,
            color: (trainingStatus?.includes('✅') || featureStatus?.includes('✅')) ? '#00ff41' : 
                   (trainingStatus?.includes('❌') || featureStatus?.includes('❌')) ? '#ff0080' : '#00ffff'
          }}>
            {trainingStatus && <div>{trainingStatus}</div>}
            {featureStatus && <div>{featureStatus}</div>}
          </div>
        )}

        {/* Dataset Selection */}
        <div style={{ marginBottom: '2rem' }}>
          <h2 style={{ fontSize: '1.8rem', marginBottom: '1rem' }}>📊 Dataset Selection</h2>
          
          <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
            <div>
              <label style={{ display: 'block', marginBottom: '0.5rem', color: '#b8bcc8' }}>
                Select Dataset for Training:
              </label>
              <select
                value={selectedDataset}
                onChange={(e) => setSelectedDataset(e.target.value)}
                style={{
                  width: '100%',
                  padding: '0.75rem',
                  backgroundColor: 'rgba(25, 25, 45, 0.5)',
                  border: '1px solid rgba(0, 255, 255, 0.3)',
                  borderRadius: '4px',
                  color: '#ffffff',
                  fontSize: '1rem'
                }}
                disabled={loading}
              >
                <option value="">Choose which dataset to use for model training</option>
                {datasets.map((dataset, index) => (
                  <option key={index} value={dataset?.name || ''}>
                    {dataset?.name || 'Unknown'} ({dataset?.rows || 0} rows)
                  </option>
                ))}
              </select>
            </div>
            
            <div style={{ display: 'flex', alignItems: 'end' }}>
              <button
                onClick={loadSelectedDataset}
                style={{
                  backgroundColor: selectedDataset ? '#00ffff' : '#666666',
                  color: '#0a0a0f',
                  border: 'none',
                  padding: '0.75rem 1.5rem',
                  borderRadius: '4px',
                  fontSize: '1rem',
                  fontWeight: 'bold',
                  cursor: selectedDataset ? 'pointer' : 'not-allowed',
                  width: '100%'
                }}
                disabled={loading || !selectedDataset}
              >
                🔄 Load Selected Dataset
              </button>
            </div>
          </div>

          {/* Show dataset list */}
          {datasets.length > 0 && (
            <div style={{
              backgroundColor: 'rgba(0, 255, 255, 0.1)',
              border: '1px solid rgba(0, 255, 255, 0.3)',
              borderRadius: '4px',
              padding: '1rem',
              marginBottom: '1rem'
            }}>
              <h4 style={{ margin: '0 0 0.5rem 0', color: '#00ffff' }}>Available Datasets:</h4>
              {datasets.map((dataset, index) => (
                <div key={index} style={{ color: '#b8bcc8', marginBottom: '0.25rem' }}>
                  • {dataset.name}: {dataset.rows} rows
                  {dataset.start_date && dataset.end_date && 
                    ` (${dataset.start_date} to ${dataset.end_date})`
                  }
                </div>
              ))}
            </div>
          )}

          {currentData && (
            <div style={{
              backgroundColor: 'rgba(0, 255, 65, 0.1)',
              border: '1px solid rgba(0, 255, 65, 0.3)',
              borderRadius: '4px',
              padding: '0.75rem',
              color: '#00ff41'
            }}>
              📈 Current dataset: {selectedDataset} loaded ({currentData.length} rows)
            </div>
          )}
        </div>

        {/* Training Configuration */}
        <div style={{ marginBottom: '2rem' }}>
          <h2 style={{ fontSize: '1.8rem', marginBottom: '1rem' }}>⚙️ Training Configuration</h2>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', marginBottom: '1rem' }}>
            <div>
              <label style={{ display: 'block', marginBottom: '0.5rem', color: '#b8bcc8' }}>
                Training Split
              </label>
              <select
                value={trainingConfig.train_split}
                onChange={(e) => setTrainingConfig(prev => ({
                  ...prev,
                  train_split: parseFloat(e.target.value)
                }))}
                style={{
                  width: '100%',
                  padding: '0.75rem',
                  backgroundColor: 'rgba(25, 25, 45, 0.5)',
                  border: '1px solid rgba(0, 255, 255, 0.3)',
                  borderRadius: '4px',
                  color: '#ffffff'
                }}
                disabled={loading}
              >
                <option value={0.6}>60% Training</option>
                <option value={0.65}>65% Training</option>
                <option value={0.7}>70% Training</option>
                <option value={0.75}>75% Training</option>
                <option value={0.8}>80% Training</option>
                <option value={0.85}>85% Training</option>
                <option value={0.9}>90% Training</option>
              </select>
            </div>
            
            <div>
              <label style={{ display: 'block', marginBottom: '0.5rem', color: '#b8bcc8' }}>
                Max Depth
              </label>
              <select
                value={trainingConfig.max_depth}
                onChange={(e) => setTrainingConfig(prev => ({
                  ...prev,
                  max_depth: parseInt(e.target.value)
                }))}
                style={{
                  width: '100%',
                  padding: '0.75rem',
                  backgroundColor: 'rgba(25, 25, 45, 0.5)',
                  border: '1px solid rgba(0, 255, 255, 0.3)',
                  borderRadius: '4px',
                  color: '#ffffff'
                }}
                disabled={loading}
              >
                <option value={4}>4</option>
                <option value={6}>6</option>
                <option value={8}>8</option>
                <option value={10}>10</option>
                <option value={12}>12</option>
              </select>
            </div>
            
            <div>
              <label style={{ display: 'block', marginBottom: '0.5rem', color: '#b8bcc8' }}>
                Number of Estimators
              </label>
              <select
                value={trainingConfig.n_estimators}
                onChange={(e) => setTrainingConfig(prev => ({
                  ...prev,
                  n_estimators: parseInt(e.target.value)
                }))}
                style={{
                  width: '100%',
                  padding: '0.75rem',
                  backgroundColor: 'rgba(25, 25, 45, 0.5)',
                  border: '1px solid rgba(0, 255, 255, 0.3)',
                  borderRadius: '4px',
                  color: '#ffffff'
                }}
                disabled={loading}
              >
                <option value={50}>50</option>
                <option value={100}>100</option>
                <option value={150}>150</option>
                <option value={200}>200</option>
                <option value={250}>250</option>
                <option value={300}>300</option>
                <option value={350}>350</option>
                <option value={400}>400</option>
                <option value={450}>450</option>
                <option value={500}>500</option>
              </select>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
            <div style={{
              backgroundColor: 'rgba(0, 255, 255, 0.1)',
              border: '1px solid rgba(0, 255, 255, 0.3)',
              borderRadius: '4px',
              padding: '0.75rem',
              color: '#00ffff'
            }}>
              Training: {Math.round(trainingConfig.train_split * 100)}% | Testing: {Math.round((1 - trainingConfig.train_split) * 100)}%
            </div>
            <div style={{
              backgroundColor: 'rgba(0, 255, 255, 0.1)',
              border: '1px solid rgba(0, 255, 255, 0.3)',
              borderRadius: '4px',
              padding: '0.75rem',
              color: '#00ffff'
            }}>
              Max Depth: {trainingConfig.max_depth} | Estimators: {trainingConfig.n_estimators}
            </div>
          </div>
        </div>

        {/* Model Selection Tabs */}
        <div style={{ marginBottom: '2rem' }}>
          <h2 style={{ fontSize: '1.8rem', marginBottom: '1rem' }}>🎯 Model Selection</h2>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1rem', marginBottom: '2rem' }}>
            {modelTabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                style={{
                  padding: '1rem',
                  backgroundColor: activeTab === tab.id ? 'rgba(0, 255, 255, 0.2)' : 'rgba(25, 25, 45, 0.5)',
                  border: `2px solid ${activeTab === tab.id ? '#00ffff' : 'rgba(0, 255, 255, 0.3)'}`,
                  borderRadius: '8px',
                  color: activeTab === tab.id ? '#00ffff' : '#ffffff',
                  cursor: 'pointer',
                  textAlign: 'center',
                  fontSize: '1rem',
                  fontWeight: 'bold',
                  transition: 'all 0.3s ease'
                }}
                disabled={loading}
              >
                <div style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>{tab.icon}</div>
                <div>{tab.name}</div>
              </button>
            ))}
          </div>

          {/* Current Model Details */}
          {currentModel && (
            <div style={{
              backgroundColor: 'rgba(25, 25, 45, 0.5)',
              border: '1px solid rgba(0, 255, 255, 0.3)',
              borderRadius: '8px',
              padding: '1.5rem',
              marginBottom: '2rem'
            }}>
              <h3 style={{ fontSize: '1.5rem', marginBottom: '1rem', color: '#00ffff' }}>
                {currentModel.icon} {currentModel.name}
              </h3>
              <p style={{ color: '#b8bcc8', marginBottom: '1rem' }}>
                {currentModel.description}
              </p>

              {/* Feature Calculation Section */}
              <div style={{ marginBottom: '1.5rem' }}>
                <h4 style={{ fontSize: '1.2rem', marginBottom: '0.5rem', color: '#ffffff' }}>
                  {currentModel.name} Features
                </h4>
                
                {!featuresCalculated[activeTab] ? (
                  <div>
                    <div style={{
                      backgroundColor: 'rgba(255, 165, 0, 0.1)',
                      border: '1px solid rgba(255, 165, 0, 0.3)',
                      borderRadius: '4px',
                      padding: '0.75rem',
                      marginBottom: '1rem',
                      color: '#ffa500'
                    }}>
                      ⚠️ {currentModel.name} features not calculated yet.
                    </div>
                    
                    <button
                      onClick={() => calculateFeatures(activeTab)}
                      style={{
                        backgroundColor: selectedDataset ? '#00ffff' : '#666666',
                        color: '#0a0a0f',
                        border: 'none',
                        padding: '0.75rem 1.5rem',
                        borderRadius: '4px',
                        fontSize: '1rem',
                        fontWeight: 'bold',
                        cursor: selectedDataset ? 'pointer' : 'not-allowed'
                      }}
                      disabled={loading || !selectedDataset}
                    >
                      🔧 Calculate Technical Indicators
                    </button>
                  </div>
                ) : (
                  <div style={{
                    backgroundColor: 'rgba(0, 255, 65, 0.1)',
                    border: '1px solid rgba(0, 255, 65, 0.3)',
                    borderRadius: '4px',
                    padding: '0.75rem',
                    color: '#00ff41'
                  }}>
                    ✅ {currentModel.name} features ready
                  </div>
                )}
              </div>

              {/* Training Section */}
              <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                <button
                  onClick={trainModel}
                  style={{
                    backgroundColor: featuresCalculated[activeTab] ? '#00ff41' : '#666666',
                    color: '#0a0a0f',
                    border: 'none',
                    padding: '1rem 2rem',
                    borderRadius: '8px',
                    fontSize: '1.1rem',
                    fontWeight: 'bold',
                    cursor: featuresCalculated[activeTab] ? 'pointer' : 'not-allowed',
                    flex: 1
                  }}
                  disabled={loading || !selectedDataset || !featuresCalculated[activeTab]}
                >
                  {loading ? '🚀 Training...' : `🚀 Train ${currentModel.name}`}
                </button>
                
                {trainingResults[activeTab] && (
                  <div style={{
                    backgroundColor: 'rgba(0, 255, 65, 0.1)',
                    border: '1px solid rgba(0, 255, 65, 0.3)',
                    borderRadius: '4px',
                    padding: '0.75rem',
                    color: '#00ff41'
                  }}>
                    ✅ Trained
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Training Results */}
        {trainingResults[activeTab] && (
          <div style={{
            backgroundColor: 'rgba(25, 25, 45, 0.5)',
            border: '1px solid rgba(0, 255, 255, 0.3)',
            borderRadius: '8px',
            padding: '1.5rem',
            marginBottom: '2rem'
          }}>
            <h3 style={{ fontSize: '1.5rem', marginBottom: '1rem', color: '#00ffff' }}>
              📊 Training Results - {currentModel?.name}
            </h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '1rem' }}>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '2rem', color: '#00ff41', fontWeight: 'bold' }}>
                  {(trainingResults[activeTab]?.r2_score * 100)?.toFixed(2) || 'N/A'}%
                </div>
                <div style={{ color: '#b8bcc8' }}>R² Score</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '2rem', color: '#00ffff', fontWeight: 'bold' }}>
                  {trainingResults[activeTab]?.feature_count || 'N/A'}
                </div>
                <div style={{ color: '#b8bcc8' }}>Features</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '2rem', color: '#ffa500', fontWeight: 'bold' }}>
                  {trainingResults[activeTab]?.training_samples || 'N/A'}
                </div>
                <div style={{ color: '#b8bcc8' }}>Samples</div>
              </div>
              {trainingResults[activeTab]?.mse && (
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: '2rem', color: '#ff0080', fontWeight: 'bold' }}>
                    {trainingResults[activeTab]?.mse?.toFixed(4) || 'N/A'}
                  </div>
                  <div style={{ color: '#b8bcc8' }}>MSE</div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Model Status Summary */}
        <div style={{ marginBottom: '2rem' }}>
          <h2 style={{ fontSize: '1.8rem', marginBottom: '1rem' }}>📊 Model Status</h2>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1rem' }}>
            {modelTabs.map((tab) => (
              <div
                key={tab.id}
                style={{
                  backgroundColor: 'rgba(25, 25, 45, 0.5)',
                  border: '1px solid rgba(0, 255, 255, 0.3)',
                  borderRadius: '8px',
                  padding: '1rem',
                  textAlign: 'center'
                }}
              >
                <div style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>{tab.icon}</div>
                <div style={{ fontWeight: 'bold', marginBottom: '0.5rem' }}>{tab.name}</div>
                
                {trainingResults[tab.id] ? (
                  <div style={{ color: '#00ff41' }}>
                    ✅ Trained - R²: {(trainingResults[tab.id]?.r2_score * 100)?.toFixed(2) || 'N/A'}%
                  </div>
                ) : featuresCalculated[tab.id] ? (
                  <div style={{ color: '#ffa500' }}>
                    🔧 Features Ready
                  </div>
                ) : (
                  <div style={{ color: '#b8bcc8' }}>
                    ⚠️ Not prepared
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Next Steps */}
        {datasets.length === 0 && (
          <div style={{
            backgroundColor: 'rgba(0, 255, 255, 0.1)',
            border: '1px solid rgba(0, 255, 255, 0.3)',
            borderRadius: '8px',
            padding: '1.5rem',
            color: '#00ffff'
          }}>
            <h3 style={{ fontSize: '1.2rem', marginBottom: '0.5rem' }}>📋 No Datasets Found</h3>
            <p style={{ margin: 0 }}>
              Please upload data using the Data Upload page first, then return here to train models.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default ModelTraining;
