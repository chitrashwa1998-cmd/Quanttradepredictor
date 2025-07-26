/**
 * Predictions page - Simplified working version
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
  const [selectedDataset, setSelectedDataset] = useState('');
  const [generatingPredictions, setGeneratingPredictions] = useState(false);
  const [analysisTab, setAnalysisTab] = useState('chart');

  // Model configuration
  const modelTabs = [
    { id: 'volatility', name: 'Volatility', icon: '📈', color: 'blue' },
    { id: 'direction', name: 'Direction', icon: '🎯', color: 'green' },
    { id: 'profit_probability', name: 'Profit Probability', icon: '💰', color: 'yellow' },
    { id: 'reversal', name: 'Reversal', icon: '🔄', color: 'purple' }
  ];

  const analysisTabs = [
    { id: 'chart', name: 'Interactive Chart', icon: '📊' },
    { id: 'data', name: 'Detailed Data', icon: '📋' },
    { id: 'distribution', name: 'Distribution Analysis', icon: '📊' },
    { id: 'statistics', name: 'Statistical Analysis', icon: '🔢' },
    { id: 'performance', name: 'Performance Metrics', icon: '⚡' }
  ];

  // Load initial data
  const loadInitialData = useCallback(async () => {
    try {
      setLoading(true);
      
      // Load datasets and model status in parallel
      const [datasetsResponse, modelsResponse] = await Promise.all([
        dataAPI.getDatasets().catch(() => ({ data: [] })),
        predictionsAPI.getModelsStatus().catch(() => ({ data: {} }))
      ]);

      setDatasets(datasetsResponse.data || []);
      setModelStatus(modelsResponse.data || {});

      // Auto-select first dataset
      const datasetList = datasetsResponse.data || [];
      if (datasetList.length > 0 && !selectedDataset) {
        setSelectedDataset(datasetList[0].name);
      }

      setPredictionStatus(`✅ Loaded ${datasetList.length} datasets`);
    } catch (error) {
      console.error('Error loading initial data:', error);
      setPredictionStatus(`❌ Error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  }, [selectedDataset]);

  useEffect(() => {
    loadInitialData();
  }, [loadInitialData]);

  // Generate predictions
  const generatePredictions = async () => {
    if (!selectedDataset) {
      setPredictionStatus('❌ Please select a dataset first');
      return;
    }

    const modelInfo = modelStatus[activeTab];
    if (!modelInfo?.trained) {
      setPredictionStatus(`❌ ${activeTab} model not trained. Please train the model first.`);
      return;
    }

    try {
      setGeneratingPredictions(true);
      setPredictionStatus(`🔮 Generating ${activeTab} predictions...`);

      const response = await predictionsAPI.generatePredictions({
        model_type: activeTab,
        dataset_name: selectedDataset,
        config: {}
      });

      if (response.success) {
        setPredictions(prev => ({
          ...prev,
          [activeTab]: response
        }));
        setPredictionStatus(`✅ Generated ${response.total_predictions || 0} ${activeTab} predictions`);
      } else {
        setPredictionStatus(`❌ Failed to generate ${activeTab} predictions`);
      }
    } catch (error) {
      console.error('Prediction error:', error);
      setPredictionStatus(`❌ Error: ${error.response?.data?.detail || error.message}`);
    } finally {
      setGeneratingPredictions(false);
    }
  };

  const activeTabColor = modelTabs.find(tab => tab.id === activeTab)?.color || 'blue';
  const currentPredictions = predictions[activeTab];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-violet-800">
      <div className="container mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-600 mb-4">
            ML Predictions
          </h1>
          <p className="text-gray-300 text-lg">
            Advanced machine learning predictions with comprehensive analysis
          </p>
        </div>

        {/* Status */}
        {predictionStatus && (
          <div className="mb-6">
            <div className={`p-4 rounded-lg ${
              predictionStatus.includes('✅') ? 'bg-green-900/30 border border-green-500/30 text-green-400' :
              predictionStatus.includes('❌') ? 'bg-red-900/30 border border-red-500/30 text-red-400' :
              'bg-blue-900/30 border border-blue-500/30 text-blue-400'
            }`}>
              {predictionStatus}
            </div>
          </div>
        )}

        {/* Model Tabs */}
        <Card className="mb-6">
          <div className="flex flex-wrap gap-2 mb-4">
            {modelTabs.map((tab) => {
              const isActive = activeTab === tab.id;
              const isModelTrained = modelStatus[tab.id]?.trained;
              
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`px-4 py-2 rounded-lg border transition-colors flex items-center space-x-2 ${
                    isActive
                      ? `bg-${tab.color}-600 border-${tab.color}-500 text-white`
                      : `border-gray-600 text-gray-300 hover:border-${tab.color}-500 hover:text-${tab.color}-400`
                  }`}
                >
                  <span>{tab.icon}</span>
                  <span>{tab.name}</span>
                  {isModelTrained && <span className="text-green-400">✓</span>}
                  {!isModelTrained && <span className="text-red-400">✗</span>}
                </button>
              );
            })}
          </div>

          {/* Dataset Selection */}
          <div className="mb-4">
            <label className="block text-cyan-400 font-medium mb-2">Select Dataset:</label>
            <select
              value={selectedDataset}
              onChange={(e) => setSelectedDataset(e.target.value)}
              className="w-full p-3 bg-gray-800 border border-gray-600 rounded-lg text-white focus:border-cyan-500 focus:outline-none"
            >
              <option value="">Select a dataset...</option>
              {datasets.map((dataset) => (
                <option key={dataset.name} value={dataset.name}>
                  {dataset.name} ({dataset.rows?.toLocaleString() || 0} rows)
                </option>
              ))}
            </select>
          </div>

          {/* Generate Predictions Button */}
          <button
            onClick={generatePredictions}
            disabled={loading || generatingPredictions || !selectedDataset}
            className={`w-full px-6 py-3 bg-${activeTabColor}-600 hover:bg-${activeTabColor}-700 text-white font-medium rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2`}
          >
            {generatingPredictions && <LoadingSpinner size="sm" />}
            <span>
              {generatingPredictions 
                ? `Generating ${activeTab} predictions...` 
                : `Generate ${activeTab} Predictions`
              }
            </span>
          </button>
        </Card>

        {/* Predictions Results */}
        {currentPredictions && (
          <Card>
            <h2 className={`text-2xl font-bold text-${activeTabColor}-400 mb-4`}>
              {modelTabs.find(tab => tab.id === activeTab)?.icon} {activeTab.charAt(0).toUpperCase() + activeTab.slice(1)} Predictions
            </h2>

            {/* Analysis Tabs */}
            <div className="flex flex-wrap gap-2 mb-6">
              {analysisTabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setAnalysisTab(tab.id)}
                  className={`px-4 py-2 rounded-lg border transition-colors flex items-center space-x-2 ${
                    analysisTab === tab.id
                      ? `bg-${activeTabColor}-600 border-${activeTabColor}-500 text-white`
                      : 'border-gray-600 text-gray-300 hover:border-cyan-500 hover:text-cyan-400'
                  }`}
                >
                  <span>{tab.icon}</span>
                  <span>{tab.name}</span>
                </button>
              ))}
            </div>

            {/* Analysis Content */}
            <div className="space-y-6">
              {analysisTab === 'chart' && (
                <div className="bg-gray-800/50 p-6 rounded-lg">
                  <h3 className="text-xl font-bold text-cyan-400 mb-4">📊 Interactive Chart</h3>
                  <div className="text-center py-8 text-gray-400">
                    <p>Chart visualization would be displayed here</p>
                    <p className="text-sm mt-2">Total predictions: {currentPredictions.total_predictions || 0}</p>
                  </div>
                </div>
              )}

              {analysisTab === 'data' && (
                <div className="bg-gray-800/50 p-6 rounded-lg">
                  <h3 className="text-xl font-bold text-cyan-400 mb-4">📋 Detailed Data</h3>
                  <div className="text-center py-8 text-gray-400">
                    <p>Detailed prediction data table would be displayed here</p>
                    <p className="text-sm mt-2">Showing first 100 of {currentPredictions.total_predictions || 0} predictions</p>
                  </div>
                </div>
              )}

              {analysisTab === 'statistics' && currentPredictions.statistics && (
                <div className="bg-gray-800/50 p-6 rounded-lg">
                  <h3 className="text-xl font-bold text-cyan-400 mb-4">🔢 Statistical Analysis</h3>
                  <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                    {Object.entries(currentPredictions.statistics).map(([key, value]) => (
                      <div key={key} className="bg-gray-700/50 p-3 rounded-lg">
                        <div className="text-sm text-gray-400 capitalize">{key.replace('_', ' ')}</div>
                        <div className="text-lg font-bold text-white">
                          {typeof value === 'number' ? value.toFixed(4) : value}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {analysisTab === 'distribution' && (
                <div className="bg-gray-800/50 p-6 rounded-lg">
                  <h3 className="text-xl font-bold text-cyan-400 mb-4">📊 Distribution Analysis</h3>
                  <div className="text-center py-8 text-gray-400">
                    <p>Distribution analysis charts would be displayed here</p>
                  </div>
                </div>
              )}

              {analysisTab === 'performance' && (
                <div className="bg-gray-800/50 p-6 rounded-lg">
                  <h3 className="text-xl font-bold text-cyan-400 mb-4">⚡ Performance Metrics</h3>
                  <div className="text-center py-8 text-gray-400">
                    <p>Performance metrics and model evaluation would be displayed here</p>
                  </div>
                </div>
              )}
            </div>
          </Card>
        )}

        {loading && (
          <div className="flex justify-center">
            <LoadingSpinner />
          </div>
        )}
      </div>
    </div>
  );
};

export default Predictions;