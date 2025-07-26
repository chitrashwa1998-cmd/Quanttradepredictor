/**
 * Predictions page - Exact Streamlit UI replication
 */

import { useState, useEffect } from 'react';
import { dataAPI, predictionsAPI } from '../services/api';

const Predictions = () => {
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState('');
  const [modelsStatus, setModelsStatus] = useState({});
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('volatility');
  const [predictions, setPredictions] = useState({});
  const [predictionStatus, setPredictionStatus] = useState('');

  // Load initial data
  const loadInitialData = async () => {
    try {
      setLoading(true);
      
      // Load datasets
      const datasetsResponse = await dataAPI.getDatasets();
      const datasetList = Array.isArray(datasetsResponse?.data) ? datasetsResponse.data : [];
      setDatasets(datasetList);

      // Auto-select preferred dataset
      const preferredDatasets = ['training_dataset', 'livenifty50'];
      let selectedName = '';
      
      for (const preferred of preferredDatasets) {
        const found = datasetList.find(d => d.name === preferred);
        if (found) {
          selectedName = preferred;
          break;
        }
      }
      
      if (!selectedName && datasetList.length > 0) {
        selectedName = datasetList[0].name;
      }
      
      setSelectedDataset(selectedName);

      // Load models status
      const modelsResponse = await predictionsAPI.getModelsStatus();
      const modelsData = modelsResponse?.data || {};
      setModelsStatus(modelsData);

      if (selectedName) {
        setPredictionStatus(`📊 Loaded data from dataset: ${selectedName} (${datasetList.find(d => d.name === selectedName)?.rows || 0} rows)`);
      } else {
        setPredictionStatus('⚠️ No data available in database. Please upload data first in the Data Upload page.');
      }
    } catch (error) {
      console.error('Error loading initial data:', error);
      setPredictionStatus(`❌ Error loading data: ${error?.message || 'Unknown error'}`);
      setDatasets([]);
      setModelsStatus({});
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadInitialData();
  }, []);

  // Generate predictions
  const generatePredictions = async () => {
    if (!selectedDataset) {
      setPredictionStatus('❌ Please select a dataset first');
      return;
    }

    try {
      setLoading(true);
      setPredictionStatus(`🔮 Generating ${activeTab} predictions...`);

      const response = await predictionsAPI.generatePredictions({
        model_type: activeTab,
        dataset_name: selectedDataset,
        config: {}
      });

      if (response?.success) {
        setPredictions(prev => ({
          ...prev,
          [activeTab]: response
        }));
        setPredictionStatus(`✅ Generated ${response.predictions_count || 0} ${activeTab} predictions`);
      } else {
        setPredictionStatus(`❌ Failed to generate ${activeTab} predictions`);
      }
    } catch (error) {
      console.error('Prediction generation error:', error);
      setPredictionStatus(`❌ Error: ${error?.response?.data?.detail || error?.message || 'Unknown error'}`);
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
          🔮 Real-Time Predictions
        </h1>
        <h3 style={{ color: '#b8bcc8', fontSize: '1.2rem', marginBottom: '2rem', fontWeight: 'normal' }}>
          Advanced ML Model Predictions - Authentic Data Only
        </h3>

        {/* Clear Cache Button */}
        <button
          onClick={() => {
            setPredictions({});
            setPredictionStatus('');
            loadInitialData();
          }}
          style={{
            backgroundColor: 'rgba(255, 0, 128, 0.2)',
            border: '1px solid rgba(255, 0, 128, 0.3)',
            borderRadius: '4px',
            color: '#ff0080',
            padding: '0.75rem 1rem',
            cursor: 'pointer',
            fontSize: '0.9rem',
            marginBottom: '2rem'
          }}
          title="Click if you see synthetic datetime warnings"
        >
          🗑️ Clear All Cached Data
        </button>

        {/* Status */}
        {predictionStatus && (
          <div style={{
            padding: '1rem',
            borderRadius: '8px',
            marginBottom: '2rem',
            backgroundColor: predictionStatus.includes('✅') ? 'rgba(0, 255, 65, 0.1)' : 
                           predictionStatus.includes('❌') ? 'rgba(255, 0, 128, 0.1)' : 
                           predictionStatus.includes('⚠️') ? 'rgba(255, 215, 0, 0.1)' : 'rgba(0, 255, 255, 0.1)',
            border: `1px solid ${predictionStatus.includes('✅') ? 'rgba(0, 255, 65, 0.3)' : 
                              predictionStatus.includes('❌') ? 'rgba(255, 0, 128, 0.3)' : 
                              predictionStatus.includes('⚠️') ? 'rgba(255, 215, 0, 0.3)' : 'rgba(0, 255, 255, 0.3)'}`,
            color: predictionStatus.includes('✅') ? '#00ff41' : 
                   predictionStatus.includes('❌') ? '#ff0080' : 
                   predictionStatus.includes('⚠️') ? '#ffd700' : '#00ffff'
          }}>
            {predictionStatus}
          </div>
        )}

        {/* Dataset Selection */}
        <div style={{ marginBottom: '2rem' }}>
          <h2 style={{ fontSize: '1.8rem', marginBottom: '1rem' }}>📊 Dataset Selection</h2>
          
          <div style={{ marginBottom: '1rem' }}>
            <label style={{ display: 'block', marginBottom: '0.5rem', color: '#b8bcc8' }}>
              Select Dataset:
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
              <option value="">Choose a dataset...</option>
              {datasets.map((dataset, index) => (
                <option key={index} value={dataset?.name || ''}>
                  {dataset?.name || 'Unknown'} ({dataset?.rows || 0} rows)
                </option>
              ))}
            </select>
          </div>

          <button
            onClick={loadInitialData}
            style={{
              backgroundColor: '#00ffff',
              color: '#0a0a0f',
              border: 'none',
              padding: '0.75rem 1.5rem',
              borderRadius: '4px',
              fontSize: '1rem',
              fontWeight: 'bold',
              cursor: 'pointer'
            }}
            disabled={loading}
          >
            🔄 Refresh Data
          </button>
        </div>

        {/* Model Tabs */}
        <div style={{ marginBottom: '2rem' }}>
          <h2 style={{ fontSize: '1.8rem', marginBottom: '1rem' }}>🎯 Model Selection</h2>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginBottom: '2rem' }}>
            {modelTabs.map((tab) => {
              const isActive = activeTab === tab.id;
              const modelStatus = modelsStatus[tab.id];
              const isTrained = modelStatus?.trained || false;
              
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  style={{
                    padding: '1rem',
                    backgroundColor: isActive ? 'rgba(0, 255, 255, 0.2)' : 'rgba(25, 25, 45, 0.5)',
                    border: `2px solid ${isActive ? '#00ffff' : 'rgba(0, 255, 255, 0.3)'}`,
                    borderRadius: '8px',
                    color: isActive ? '#00ffff' : '#ffffff',
                    cursor: 'pointer',
                    textAlign: 'center',
                    fontSize: '1rem',
                    fontWeight: 'bold',
                    position: 'relative'
                  }}
                  disabled={loading}
                >
                  <div style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>{tab.icon}</div>
                  <div style={{ marginBottom: '0.5rem' }}>{tab.name}</div>
                  <div style={{ 
                    fontSize: '0.75rem', 
                    color: isTrained ? '#00ff41' : '#ff0080' 
                  }}>
                    {isTrained ? '🟢 Trained' : '🔴 Not Trained'}
                  </div>
                </button>
              );
            })}
          </div>

          {/* Generate Predictions Button */}
          <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
            <button
              onClick={generatePredictions}
              style={{
                backgroundColor: '#8b5cf6',
                color: '#ffffff',
                border: 'none',
                padding: '1rem 2rem',
                borderRadius: '8px',
                fontSize: '1.1rem',
                fontWeight: 'bold',
                cursor: 'pointer',
                minWidth: '250px',
                background: 'linear-gradient(45deg, #8b5cf6, #ff0080)'
              }}
              disabled={loading || !selectedDataset || !modelsStatus[activeTab]?.trained}
            >
              {loading ? '🔮 Generating...' : `🔮 Generate ${activeTab} Predictions`}
            </button>
            
            <div style={{ marginTop: '0.5rem', color: '#b8bcc8', fontSize: '0.9rem' }}>
              {!selectedDataset && 'Please select a dataset first'}
              {selectedDataset && !modelsStatus[activeTab]?.trained && `${activeTab} model needs to be trained first`}
              {selectedDataset && modelsStatus[activeTab]?.trained && 'Ready to generate predictions'}
            </div>
          </div>
        </div>

        {/* Prediction Results */}
        {predictions[activeTab] && (
          <div style={{
            backgroundColor: 'rgba(25, 25, 45, 0.5)',
            border: '1px solid rgba(0, 255, 255, 0.3)',
            borderRadius: '8px',
            padding: '2rem',
            marginTop: '2rem'
          }}>
            <h2 style={{ fontSize: '1.8rem', marginBottom: '1.5rem', color: '#00ffff' }}>
              📊 Prediction Results - {activeTab}
            </h2>
            
            {/* Results Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginBottom: '2rem' }}>
              <div style={{
                backgroundColor: 'rgba(0, 255, 65, 0.1)',
                border: '1px solid rgba(0, 255, 65, 0.3)',
                borderRadius: '8px',
                padding: '1rem',
                textAlign: 'center'
              }}>
                <div style={{ fontSize: '2rem', color: '#00ff41', fontWeight: 'bold', marginBottom: '0.5rem' }}>
                  {predictions[activeTab]?.predictions_count || 0}
                </div>
                <div style={{ color: '#b8bcc8' }}>Predictions Generated</div>
              </div>
              
              <div style={{
                backgroundColor: 'rgba(0, 255, 255, 0.1)',
                border: '1px solid rgba(0, 255, 255, 0.3)',
                borderRadius: '8px',
                padding: '1rem',
                textAlign: 'center'
              }}>
                <div style={{ fontSize: '2rem', color: '#00ffff', fontWeight: 'bold', marginBottom: '0.5rem' }}>
                  {(predictions[activeTab]?.confidence * 100)?.toFixed(2) || 'N/A'}%
                </div>
                <div style={{ color: '#b8bcc8' }}>Confidence</div>
              </div>
              
              <div style={{
                backgroundColor: 'rgba(139, 92, 246, 0.1)',
                border: '1px solid rgba(139, 92, 246, 0.3)',
                borderRadius: '8px',
                padding: '1rem',
                textAlign: 'center'
              }}>
                <div style={{ fontSize: '2rem', color: '#8b5cf6', fontWeight: 'bold', marginBottom: '0.5rem' }}>
                  {activeTab.charAt(0).toUpperCase() + activeTab.slice(1)}
                </div>
                <div style={{ color: '#b8bcc8' }}>Model Type</div>
              </div>
            </div>
            
            {/* Prediction Details */}
            <div style={{
              backgroundColor: 'rgba(0, 0, 0, 0.3)',
              borderRadius: '8px',
              padding: '1.5rem'
            }}>
              <h3 style={{ fontSize: '1.2rem', color: '#00ffff', marginBottom: '1rem' }}>Prediction Details</h3>
              <div style={{ color: '#b8bcc8', lineHeight: '1.6' }}>
                <p><strong>Dataset:</strong> {selectedDataset}</p>
                <p><strong>Model:</strong> {activeTab}</p>
                <p><strong>Generated:</strong> {predictions[activeTab]?.timestamp || new Date().toISOString()}</p>
                <p><strong>Status:</strong> ✅ Predictions ready for analysis</p>
              </div>
            </div>
          </div>
        )}

        {/* No Data Warning */}
        {!selectedDataset && datasets.length === 0 && (
          <div style={{
            backgroundColor: 'rgba(255, 215, 0, 0.1)',
            border: '1px solid rgba(255, 215, 0, 0.3)',
            borderRadius: '8px',
            padding: '2rem',
            textAlign: 'center',
            color: '#ffd700'
          }}>
            <h3 style={{ marginBottom: '1rem' }}>⚠️ No Data Available</h3>
            <p style={{ marginBottom: '1.5rem', color: '#b8bcc8' }}>
              No data available in database. Please upload data first in the Data Upload page.
            </p>
            <button
              style={{
                backgroundColor: '#00ffff',
                color: '#0a0a0f',
                border: 'none',
                padding: '0.75rem 1.5rem',
                borderRadius: '4px',
                fontSize: '1rem',
                fontWeight: 'bold',
                cursor: 'pointer'
              }}
              onClick={() => window.location.href = '/data-upload'}
            >
              📊 Go to Data Upload
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default Predictions;