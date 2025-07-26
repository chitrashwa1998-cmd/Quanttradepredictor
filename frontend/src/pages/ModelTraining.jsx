/**
 * Model Training page - Exact Streamlit UI replication
 */

import { useState, useEffect } from 'react';
import { dataAPI, modelsAPI } from '../services/api';

const ModelTraining = () => {
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState('');
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('volatility');
  const [trainingConfig, setTrainingConfig] = useState({
    train_split: 0.8,
    max_depth: 6,
    n_estimators: 100
  });
  const [trainingStatus, setTrainingStatus] = useState('');
  const [trainingResults, setTrainingResults] = useState({});

  // Load datasets
  const loadDatasets = async () => {
    try {
      setLoading(true);
      const response = await dataAPI.getDatasets();
      const datasetList = Array.isArray(response?.data) ? response.data : [];
      setDatasets(datasetList);

      // Auto-select training_dataset if available
      const trainingDataset = datasetList.find(d => d.name === 'training_dataset');
      if (trainingDataset) {
        setSelectedDataset('training_dataset');
      } else if (datasetList.length > 0) {
        setSelectedDataset(datasetList[0].name);
      }
    } catch (error) {
      console.error('Error loading datasets:', error);
      setTrainingStatus(`❌ Error loading datasets: ${error?.message || 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDatasets();
  }, []);

  // Train model
  const trainModel = async () => {
    if (!selectedDataset) {
      setTrainingStatus('❌ Please select a dataset first');
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
        setTrainingStatus(`❌ Failed to train ${activeTab} model`);
      }
    } catch (error) {
      console.error('Training error:', error);
      setTrainingStatus(`❌ Training error: ${error?.response?.data?.detail || error?.message || 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  };

  const modelTabs = [
    { id: 'volatility', name: 'Volatility Model', icon: '📈' },
    { id: 'direction', name: 'Direction Model', icon: '🎯' },
    { id: 'profit_probability', name: 'Profit Probability Model', icon: '💰' },
    { id: 'reversal', name: 'Reversal Model', icon: '🔄' }
  ];

  return (
    <div style={{ backgroundColor: '#0a0a0f', minHeight: '100vh', color: '#ffffff', fontFamily: 'Space Grotesk, sans-serif' }}>
      <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '2rem' }}>
        
        {/* Header */}
        <h1 style={{ fontSize: '2.5rem', marginBottom: '0.5rem', fontFamily: 'Orbitron, monospace' }}>
          🧠 Model Training
        </h1>
        <p style={{ color: '#b8bcc8', fontSize: '1.1rem', marginBottom: '2rem' }}>
          Train prediction models using your processed data.
        </p>

        {/* Status */}
        {trainingStatus && (
          <div style={{
            padding: '1rem',
            borderRadius: '8px',
            marginBottom: '2rem',
            backgroundColor: trainingStatus.includes('✅') ? 'rgba(0, 255, 65, 0.1)' : 
                           trainingStatus.includes('❌') ? 'rgba(255, 0, 128, 0.1)' : 'rgba(0, 255, 255, 0.1)',
            border: `1px solid ${trainingStatus.includes('✅') ? 'rgba(0, 255, 65, 0.3)' : 
                              trainingStatus.includes('❌') ? 'rgba(255, 0, 128, 0.3)' : 'rgba(0, 255, 255, 0.3)'}`,
            color: trainingStatus.includes('✅') ? '#00ff41' : 
                   trainingStatus.includes('❌') ? '#ff0080' : '#00ffff'
          }}>
            {trainingStatus}
          </div>
        )}

        {/* Dataset Selection */}
        <div style={{ marginBottom: '2rem' }}>
          <h2 style={{ fontSize: '1.8rem', marginBottom: '1rem' }}>📊 Dataset Selection</h2>
          
          <div style={{ marginBottom: '1rem' }}>
            <label style={{ display: 'block', marginBottom: '0.5rem', color: '#b8bcc8' }}>
              Select Dataset for Training:
            </label>
            <select
              value={selectedDataset}
              onChange={(e) => setSelectedDataset(e.target.value)}
              style={{
                width: '100%',
                maxWidth: '400px',
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

          <button
            onClick={loadDatasets}
            style={{
              backgroundColor: '#00ffff',
              color: '#0a0a0f',
              border: 'none',
              padding: '0.75rem 1.5rem',
              borderRadius: '4px',
              fontSize: '1rem',
              fontWeight: 'bold',
              cursor: 'pointer',
              marginRight: '1rem'
            }}
            disabled={loading}
          >
            🔄 Load Selected Dataset
          </button>

          {selectedDataset && (
            <div style={{
              backgroundColor: 'rgba(0, 255, 255, 0.1)',
              border: '1px solid rgba(0, 255, 255, 0.3)',
              borderRadius: '4px',
              padding: '0.75rem',
              marginTop: '1rem',
              color: '#00ffff'
            }}>
              📈 Current dataset: {selectedDataset} ready for training
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

        {/* Model Selection */}
        <div style={{ marginBottom: '2rem' }}>
          <h2 style={{ fontSize: '1.8rem', marginBottom: '1rem' }}>🎯 Model Selection</h2>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginBottom: '1rem' }}>
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

          <button
            onClick={trainModel}
            style={{
              backgroundColor: '#00ff41',
              color: '#0a0a0f',
              border: 'none',
              padding: '1rem 2rem',
              borderRadius: '8px',
              fontSize: '1.1rem',
              fontWeight: 'bold',
              cursor: 'pointer',
              width: '100%',
              maxWidth: '300px'
            }}
            disabled={loading || !selectedDataset}
          >
            {loading ? '🚀 Training...' : `🚀 Train ${activeTab} Model`}
          </button>
        </div>

        {/* Training Results */}
        {trainingResults[activeTab] && (
          <div style={{
            backgroundColor: 'rgba(25, 25, 45, 0.5)',
            border: '1px solid rgba(0, 255, 255, 0.3)',
            borderRadius: '8px',
            padding: '1.5rem',
            marginTop: '2rem'
          }}>
            <h3 style={{ fontSize: '1.5rem', marginBottom: '1rem', color: '#00ffff' }}>
              📊 Training Results - {activeTab}
            </h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '1rem' }}>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '2rem', color: '#00ff41', fontWeight: 'bold' }}>
                  {(trainingResults[activeTab]?.accuracy * 100)?.toFixed(2) || 'N/A'}%
                </div>
                <div style={{ color: '#b8bcc8' }}>Accuracy</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '2rem', color: '#00ffff', fontWeight: 'bold' }}>
                  {trainingResults[activeTab]?.feature_count || 'N/A'}
                </div>
                <div style={{ color: '#b8bcc8' }}>Features</div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ModelTraining;