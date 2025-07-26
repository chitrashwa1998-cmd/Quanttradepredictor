/**
 * Predictions page - Robust error-free version
 */

import { useState, useEffect } from 'react';
import Card from '../components/common/Card';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { dataAPI, predictionsAPI } from '../services/api';

const Predictions = () => {
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState('');
  const [modelsStatus, setModelsStatus] = useState({});
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('volatility');
  const [predictions, setPredictions] = useState({});
  const [predictionStatus, setPredictionStatus] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);

  // Model configuration
  const modelTabs = [
    { id: 'volatility', name: 'Volatility', icon: '📈', color: 'blue' },
    { id: 'direction', name: 'Direction', icon: '🎯', color: 'green' },
    { id: 'profit_probability', name: 'Profit Probability', icon: '💰', color: 'yellow' },
    { id: 'reversal', name: 'Reversal', icon: '🔄', color: 'purple' }
  ];

  // Load initial data
  const loadInitialData = async () => {
    try {
      setLoading(true);
      
      // Load datasets
      const datasetsResponse = await dataAPI.getDatasets();
      const datasetList = Array.isArray(datasetsResponse?.data) ? datasetsResponse.data : [];
      setDatasets(datasetList);

      // Auto-select first dataset
      if (datasetList.length > 0 && !selectedDataset) {
        setSelectedDataset(datasetList[0]?.name || '');
      }

      // Load models status
      const modelsResponse = await predictionsAPI.getModelsStatus();
      const modelsData = modelsResponse?.data || {};
      setModelsStatus(modelsData);

      setPredictionStatus(`✅ Loaded ${datasetList.length} datasets and models status`);
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

    const modelStatus = modelsStatus[activeTab];
    if (!modelStatus?.trained) {
      setPredictionStatus(`❌ ${activeTab} model is not trained yet`);
      return;
    }

    try {
      setIsGenerating(true);
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
      setIsGenerating(false);
      setLoading(false);
    }
  };

  // Get model status indicator
  const getModelStatusIndicator = (modelType) => {
    const status = modelsStatus[modelType];
    if (!status) return '⚪ Unknown';
    return status.trained ? '🟢 Trained' : '🔴 Not Trained';
  };

  // Get prediction summary
  const getPredictionSummary = (modelType) => {
    const predictionData = predictions[modelType];
    if (!predictionData) return null;

    return {
      count: predictionData.predictions_count || 0,
      accuracy: predictionData.confidence || 0,
      timestamp: predictionData.timestamp || new Date().toISOString()
    };
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-violet-800">
      <div className="container mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-600 mb-4">
            Market Predictions
          </h1>
          <p className="text-gray-300 text-lg">
            Generate AI-powered market predictions using trained models
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

        {loading && (
          <div className="flex justify-center mb-6">
            <LoadingSpinner />
          </div>
        )}

        {/* Dataset Selection */}
        <Card className="mb-8">
          <h2 className="text-2xl font-bold text-cyan-400 mb-4">📊 Dataset Selection</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-gray-300 mb-2">Select Dataset:</label>
              <select
                value={selectedDataset}
                onChange={(e) => setSelectedDataset(e.target.value)}
                className="w-full p-3 bg-gray-800 border border-gray-600 rounded-lg text-white focus:ring-2 focus:ring-cyan-500 focus:border-transparent"
                disabled={loading}
              >
                <option value="">Choose a dataset...</option>
                {datasets.map((dataset, index) => (
                  <option key={dataset?.name || index} value={dataset?.name || ''}>
                    {dataset?.name || 'Unknown'} ({dataset?.rows || 0} rows)
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-gray-300 mb-2">Refresh Data:</label>
              <button
                onClick={loadInitialData}
                className="w-full p-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
                disabled={loading}
              >
                🔄 Refresh
              </button>
            </div>
          </div>
        </Card>

        {/* Model Status Overview */}
        <Card className="mb-8">
          <h2 className="text-2xl font-bold text-cyan-400 mb-4">🤖 Model Status</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {modelTabs.map((tab) => {
              const status = modelsStatus[tab.id];
              const isReady = status?.trained || false;
              
              return (
                <div
                  key={tab.id}
                  className={`p-4 rounded-lg border-2 ${
                    isReady
                      ? `border-green-500 bg-green-900/30`
                      : 'border-red-500 bg-red-900/30'
                  }`}
                >
                  <div className="text-2xl mb-2">{tab.icon}</div>
                  <div className="font-bold text-white">{tab.name}</div>
                  <div className={`text-xs mt-1 ${isReady ? 'text-green-400' : 'text-red-400'}`}>
                    {getModelStatusIndicator(tab.id)}
                  </div>
                  {status?.last_updated && (
                    <div className="text-xs text-gray-400 mt-1">
                      Updated: {status.last_updated}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </Card>

        {/* Model Selection */}
        <Card className="mb-8">
          <h2 className="text-2xl font-bold text-cyan-400 mb-4">🎯 Prediction Model</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {modelTabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`p-4 rounded-lg border-2 transition-all ${
                  activeTab === tab.id
                    ? `border-${tab.color}-500 bg-${tab.color}-900/30 text-${tab.color}-400`
                    : 'border-gray-600 bg-gray-800/30 text-gray-300 hover:border-gray-500'
                }`}
                disabled={loading}
              >
                <div className="text-2xl mb-2">{tab.icon}</div>
                <div className="font-bold">{tab.name}</div>
                <div className="text-xs mt-1">
                  {getModelStatusIndicator(tab.id)}
                </div>
              </button>
            ))}
          </div>
        </Card>

        {/* Prediction Actions */}
        <Card className="mb-8">
          <h2 className="text-2xl font-bold text-cyan-400 mb-4">🔮 Generate Predictions</h2>
          <div className="space-y-4">
            <div className="text-center">
              <button
                onClick={generatePredictions}
                className="px-8 py-4 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white rounded-lg text-lg font-bold transition-all transform hover:scale-105"
                disabled={loading || !selectedDataset || !modelsStatus[activeTab]?.trained || isGenerating}
              >
                {isGenerating ? '🔮 Generating...' : `🔮 Generate ${activeTab} Predictions`}
              </button>
            </div>
            
            <div className="text-center text-gray-400 text-sm">
              {!selectedDataset && 'Please select a dataset first'}
              {selectedDataset && !modelsStatus[activeTab]?.trained && `${activeTab} model needs to be trained first`}
              {selectedDataset && modelsStatus[activeTab]?.trained && 'Ready to generate predictions'}
            </div>
          </div>
        </Card>

        {/* Prediction Results */}
        {predictions[activeTab] && (
          <Card>
            <h2 className="text-2xl font-bold text-cyan-400 mb-4">📊 Prediction Results - {activeTab}</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <div className="bg-gradient-to-br from-green-900/50 to-emerald-900/50 p-4 rounded-lg border border-green-500/30">
                <div className="text-2xl font-bold text-green-400">
                  {getPredictionSummary(activeTab)?.count || 0}
                </div>
                <div className="text-gray-300">Predictions Generated</div>
              </div>
              <div className="bg-gradient-to-br from-blue-900/50 to-cyan-900/50 p-4 rounded-lg border border-blue-500/30">
                <div className="text-2xl font-bold text-blue-400">
                  {(getPredictionSummary(activeTab)?.accuracy * 100)?.toFixed(2) || 'N/A'}%
                </div>
                <div className="text-gray-300">Confidence</div>
              </div>
              <div className="bg-gradient-to-br from-purple-900/50 to-pink-900/50 p-4 rounded-lg border border-purple-500/30">
                <div className="text-2xl font-bold text-purple-400">
                  {activeTab.charAt(0).toUpperCase() + activeTab.slice(1)}
                </div>
                <div className="text-gray-300">Model Type</div>
              </div>
            </div>
            
            <div className="mt-6 p-4 bg-gray-800/50 rounded-lg">
              <h3 className="text-lg font-bold text-cyan-400 mb-2">Prediction Details</h3>
              <div className="text-gray-300 text-sm">
                <p><strong>Dataset:</strong> {selectedDataset}</p>
                <p><strong>Model:</strong> {activeTab}</p>
                <p><strong>Generated:</strong> {getPredictionSummary(activeTab)?.timestamp || 'Unknown'}</p>
                <p><strong>Status:</strong> ✅ Predictions ready for analysis</p>
              </div>
            </div>
          </Card>
        )}
      </div>
    </div>
  );
};

export default Predictions;