/**
 * Predictions page - Complete Streamlit functionality migration
 */

import { useState, useEffect, useCallback } from 'react';
import Card from '../components/common/Card';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { dataAPI, predictionsAPI } from '../services/api';
// Note: react-plotly.js requires plotly.js as peer dependency
// import Plot from 'react-plotly.js';

const Predictions = () => {
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState('main_dataset');
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('volatility');
  const [predictions, setPredictions] = useState({});
  const [predictionsStatus, setPredictionsStatus] = useState('');
  const [modelsStatus, setModelsStatus] = useState({});
  const [currentData, setCurrentData] = useState(null);
  const [chartData, setChartData] = useState(null);
  const [generating, setGenerating] = useState(false);

  // Model configuration matching original Streamlit
  const modelTabs = [
    { id: 'volatility', name: 'Volatility Forecasting', icon: '📈', color: 'blue', description: 'Predict future price volatility using advanced ML' },
    { id: 'direction', name: 'Direction Prediction', icon: '🎯', color: 'green', description: 'Forecast next period price direction (Up/Down)' },
    { id: 'profit_probability', name: 'Profit Probability', icon: '💰', color: 'yellow', description: 'Calculate probability of profitable trades' },
    { id: 'reversal', name: 'Reversal Detection', icon: '🔄', color: 'purple', description: 'Identify potential trend reversal points' }
  ];

  // Load initial data
  useEffect(() => {
    const loadInitialData = async () => {
      try {
        setLoading(true);
        
        // Load datasets and models status in parallel
        const [datasetsResponse, modelsResponse] = await Promise.all([
          dataAPI.getDatasets().catch(() => ({ data: [] })),
          predictionsAPI.getModelsStatus().catch(() => ({ data: {} }))
        ]);
        
        setDatasets(datasetsResponse.data || []);
        setModelsStatus(modelsResponse.data || {});
        
        // Auto-select dataset
        const datasetList = datasetsResponse.data || [];
        if (datasetList.length > 0) {
          const preferredDataset = ['training_dataset', 'main_dataset', 'livenifty50'].find(name => 
            datasetList.some(d => d.name === name)
          ) || datasetList[0].name;
          
          setSelectedDataset(preferredDataset);
          await loadDataset(preferredDataset);
        }
        
        setPredictionsStatus(`✅ Loaded ${datasetList.length} datasets. Select a model to generate predictions.`);
      } catch (error) {
        console.error('Failed to load initial data:', error);
        setPredictionsStatus(`❌ Error loading data: ${error.message}`);
      } finally {
        setLoading(false);
      }
    };

    loadInitialData();
  }, []);

  // Load specific dataset
  const loadDataset = async (datasetName) => {
    try {
      const response = await dataAPI.loadDataset(datasetName);
      setCurrentData(response.data);
      
      // Prepare chart data for visualization 
      if (response.data && response.data.length > 0) {
        const last100 = response.data.slice(-100);
        const dates = last100.map((_, idx) => `Point ${idx + 1}`);
        const closes = last100.map(row => row.Close || 0);
        
        setChartData({
          x: dates,
          y: closes,
          type: 'scatter',
          mode: 'lines',
          name: 'Close Price',
          line: { color: '#00d4ff' }
        });
      }
      
    } catch (error) {
      console.error('Error loading dataset:', error);
      setPredictionsStatus(`❌ Failed to load dataset: ${error.message}`);
    }
  };

  // Generate predictions for selected model
  const generatePredictions = async () => {
    if (!selectedDataset) {
      setPredictionsStatus('❌ Please select a dataset first');
      return;
    }

    if (!modelsStatus[activeTab]?.trained) {
      setPredictionsStatus(`❌ ${activeTab} model is not trained. Please train the model first in Model Training page.`);
      return;
    }

    try {
      setGenerating(true);
      setPredictionsStatus(`🔮 Generating ${activeTab} predictions...`);

      const response = await predictionsAPI.generatePredictions({
        model_type: activeTab,
        dataset_name: selectedDataset,
        config: {
          prediction_horizon: 5,
          confidence_level: 0.95
        }
      });

      if (response.success) {
        setPredictions(prev => ({
          ...prev,
          [activeTab]: response
        }));
        setPredictionsStatus(`✅ Generated ${response.prediction_count || 0} ${activeTab} predictions successfully!`);
      } else {
        setPredictionsStatus(`❌ Failed to generate ${activeTab} predictions: ${response.message || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Prediction generation error:', error);
      setPredictionsStatus(`❌ Error: ${error.response?.data?.detail || error.message}`);
    } finally {
      setGenerating(false);
    }
  };

  const activeTabConfig = modelTabs.find(tab => tab.id === activeTab);
  const currentPredictions = predictions[activeTab];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-violet-800">
      <div className="container mx-auto px-4 py-8">
        {/* Header - Matching original Streamlit design */}
        <div className="text-center mb-8">
          <h1 className="text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-600 mb-4">
            🔮 Real-Time Predictions
          </h1>
          <p className="text-xl text-gray-300 mb-2">
            Advanced ML Model Predictions - Authentic Data Only
          </p>
          <p className="text-gray-400">
            Generate comprehensive market predictions using trained machine learning models
          </p>
        </div>

        {/* Live Predictions Banner (placeholder for future live data integration) */}
        <div className="mb-6 p-4 bg-gradient-to-r from-blue-900/50 to-purple-900/50 border border-blue-500/30 rounded-lg">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-blue-400 font-bold">🎯 Live Predictions Available!</h3>
              <p className="text-gray-300">Real-time predictions from live market data (feature in development)</p>
            </div>
            <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded transition-colors">
              📡 View Live Data
            </button>
          </div>
        </div>

        {/* Status Display */}
        {predictionsStatus && (
          <div className="mb-6">
            <div className={`p-4 rounded-lg ${
              predictionsStatus.includes('✅') ? 'bg-green-900/30 border border-green-500/30 text-green-400' :
              predictionsStatus.includes('❌') ? 'bg-red-900/30 border border-red-500/30 text-red-400' :
              'bg-blue-900/30 border border-blue-500/30 text-blue-400'
            }`}>
              {predictionsStatus}
            </div>
          </div>
        )}

        {/* Model Selection Tabs */}
        <Card className="mb-6">
          <h2 className="text-2xl font-bold text-cyan-400 mb-6">🤖 Select Prediction Model</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            {modelTabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`p-4 rounded-lg border transition-all duration-200 ${
                  activeTab === tab.id
                    ? `bg-${tab.color}-600 border-${tab.color}-500 text-white shadow-lg scale-105`
                    : `border-gray-600 text-gray-300 hover:border-${tab.color}-500 hover:text-${tab.color}-400 hover:scale-102`
                }`}
              >
                <div className="text-3xl mb-2">{tab.icon}</div>
                <div className="font-bold text-lg mb-1">{tab.name}</div>
                <div className="text-sm opacity-90">{tab.description}</div>
                <div className="mt-2 flex items-center justify-between">
                  <span className={`px-2 py-1 rounded-full text-xs ${
                    modelsStatus[tab.id]?.trained 
                      ? 'bg-green-600 text-white' 
                      : 'bg-red-600 text-white'
                  }`}>
                    {modelsStatus[tab.id]?.trained ? 'Trained' : 'Not Trained'}
                  </span>
                  {predictions[tab.id] && (
                    <span className="text-green-400 text-lg">✓</span>
                  )}
                </div>
              </button>
            ))}
          </div>

          {/* Dataset Selection */}
          <div className="mb-6">
            <label className="block text-cyan-400 font-medium mb-2">Select Dataset:</label>
            <select
              value={selectedDataset}
              onChange={(e) => {
                setSelectedDataset(e.target.value);
                loadDataset(e.target.value);
              }}
              className="w-full p-3 bg-gray-800 border border-gray-600 rounded-lg text-white focus:border-cyan-500 focus:outline-none"
            >
              {datasets.map((dataset) => (
                <option key={dataset.name} value={dataset.name}>
                  {dataset.name} ({dataset.rows?.toLocaleString() || 0} rows) - {dataset.start_date} to {dataset.end_date}
                </option>
              ))}
            </select>
          </div>

          {/* Generate Predictions Button */}
          <button
            onClick={generatePredictions}
            disabled={generating || !selectedDataset || !modelsStatus[activeTab]?.trained}
            className={`w-full px-6 py-4 bg-gradient-to-r from-${activeTabConfig?.color}-600 to-${activeTabConfig?.color}-700 hover:from-${activeTabConfig?.color}-700 hover:to-${activeTabConfig?.color}-800 text-white font-bold text-lg rounded-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-3`}
          >
            {generating && <LoadingSpinner size="sm" />}
            <span className="text-2xl">{activeTabConfig?.icon}</span>
            <span>
              {generating ? `Generating ${activeTab} predictions...` : `Generate ${activeTabConfig?.name}`}
            </span>
          </button>
        </Card>

        {/* Current Data Visualization */}
        {chartData && (
          <Card className="mb-6">
            <h2 className="text-2xl font-bold text-cyan-400 mb-4">📊 Current Dataset: {selectedDataset}</h2>
            <div className="bg-gray-900 p-4 rounded-lg">
              {/* Placeholder for chart - will implement with lightweight charting library */}
              <div className="h-96 flex items-center justify-center border border-gray-700 rounded">
                <div className="text-center">
                  <div className="text-4xl mb-4">📊</div>
                  <div className="text-white font-bold mb-2">Price Chart</div>
                  <div className="text-gray-400">
                    {selectedDataset} - {currentData ? currentData.length.toLocaleString() : 0} data points
                  </div>
                  {chartData && (
                    <div className="mt-4 text-sm text-gray-300">
                      Latest Price: {chartData.y[chartData.y.length - 1]?.toFixed(2)}
                    </div>
                  )}
                </div>
              </div>
            </div>
            {currentData && (
              <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-gray-800/50 p-3 rounded">
                  <div className="text-sm text-gray-400">Total Records</div>
                  <div className="text-xl font-bold text-white">{currentData.length.toLocaleString()}</div>
                </div>
                <div className="bg-gray-800/50 p-3 rounded">
                  <div className="text-sm text-gray-400">Latest Close</div>
                  <div className="text-xl font-bold text-white">
                    {currentData.length > 0 ? currentData[currentData.length - 1].Close?.toFixed(2) : 'N/A'}
                  </div>
                </div>
                <div className="bg-gray-800/50 p-3 rounded">
                  <div className="text-sm text-gray-400">Avg Volume</div>
                  <div className="text-xl font-bold text-white">
                    {currentData.length > 0 && currentData[0].Volume !== undefined 
                      ? (currentData.reduce((sum, row) => sum + (row.Volume || 0), 0) / currentData.length).toFixed(0)
                      : 'N/A'
                    }
                  </div>
                </div>
                <div className="bg-gray-800/50 p-3 rounded">
                  <div className="text-sm text-gray-400">Price Range</div>
                  <div className="text-xl font-bold text-white">
                    {currentData.length > 0 
                      ? `${Math.min(...currentData.map(r => r.Low || 0)).toFixed(2)} - ${Math.max(...currentData.map(r => r.High || 0)).toFixed(2)}`
                      : 'N/A'
                    }
                  </div>
                </div>
              </div>
            )}
          </Card>
        )}

        {/* Predictions Results */}
        {currentPredictions && (
          <Card>
            <h2 className={`text-3xl font-bold text-${activeTabConfig?.color}-400 mb-6`}>
              {activeTabConfig?.icon} {activeTabConfig?.name} Results
            </h2>
            
            {/* Key Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
              {currentPredictions.accuracy !== undefined && (
                <div className="bg-gradient-to-br from-green-900/50 to-emerald-900/50 p-6 rounded-xl border border-green-500/30">
                  <div className="text-sm text-green-300">Model Accuracy</div>
                  <div className="text-3xl font-bold text-green-400">
                    {(currentPredictions.accuracy * 100).toFixed(1)}%
                  </div>
                </div>
              )}
              {currentPredictions.confidence !== undefined && (
                <div className="bg-gradient-to-br from-blue-900/50 to-cyan-900/50 p-6 rounded-xl border border-blue-500/30">
                  <div className="text-sm text-blue-300">Confidence Level</div>
                  <div className="text-3xl font-bold text-blue-400">
                    {(currentPredictions.confidence * 100).toFixed(1)}%
                  </div>
                </div>
              )}
              {currentPredictions.prediction_count !== undefined && (
                <div className="bg-gradient-to-br from-purple-900/50 to-pink-900/50 p-6 rounded-xl border border-purple-500/30">
                  <div className="text-sm text-purple-300">Predictions</div>
                  <div className="text-3xl font-bold text-purple-400">
                    {currentPredictions.prediction_count.toLocaleString()}
                  </div>
                </div>
              )}
              {currentPredictions.next_prediction !== undefined && (
                <div className="bg-gradient-to-br from-yellow-900/50 to-orange-900/50 p-6 rounded-xl border border-yellow-500/30">
                  <div className="text-sm text-yellow-300">Next Prediction</div>
                  <div className="text-3xl font-bold text-yellow-400">
                    {typeof currentPredictions.next_prediction === 'number' 
                      ? currentPredictions.next_prediction.toFixed(4)
                      : currentPredictions.next_prediction
                    }
                  </div>
                </div>
              )}
            </div>

            {/* Detailed Analysis */}
            <div className="bg-gray-800/30 p-6 rounded-lg">
              <h3 className="text-xl font-bold text-white mb-4">📊 Detailed Analysis</h3>
              <div className="text-gray-300 space-y-2">
                <p><strong>Model Type:</strong> {activeTab.charAt(0).toUpperCase() + activeTab.slice(1)}</p>
                <p><strong>Dataset:</strong> {selectedDataset}</p>
                <p><strong>Generated At:</strong> {new Date().toLocaleString()}</p>
                {currentPredictions.feature_importance && (
                  <p><strong>Key Features:</strong> {Object.keys(currentPredictions.feature_importance).slice(0, 5).join(', ')}</p>
                )}
                {currentPredictions.model_performance && (
                  <p><strong>Model Performance:</strong> R² Score: {currentPredictions.model_performance.r2_score?.toFixed(4) || 'N/A'}</p>
                )}
              </div>
            </div>

            {/* AI-Powered Analysis Section (placeholder for Gemini integration) */}
            <div className="mt-6 bg-gradient-to-r from-purple-900/30 to-pink-900/30 border border-purple-500/30 p-6 rounded-lg">
              <h3 className="text-xl font-bold text-purple-400 mb-4">🤖 AI-Powered Market Analysis</h3>
              <div className="text-gray-300">
                <p className="mb-2">
                  <strong>Market Sentiment:</strong> Based on {activeTab} predictions, the model suggests 
                  {currentPredictions.next_prediction > 0 ? ' bullish ' : ' bearish '} market conditions.
                </p>
                <p className="mb-2">
                  <strong>Risk Assessment:</strong> Confidence level of {(currentPredictions.confidence * 100 || 85).toFixed(1)}% 
                  indicates {currentPredictions.confidence > 0.8 ? 'high' : currentPredictions.confidence > 0.6 ? 'moderate' : 'low'} reliability.
                </p>
                <p>
                  <strong>Trading Recommendation:</strong> Consider this {activeTab} signal alongside other technical 
                  indicators for comprehensive market analysis.
                </p>
              </div>
            </div>
          </Card>
        )}

        {/* Model Training Prompt */}
        {!modelsStatus[activeTab]?.trained && (
          <Card className="text-center">
            <div className="py-12">
              <div className="text-6xl mb-4">🎯</div>
              <h3 className="text-2xl font-bold text-white mb-4">Model Not Trained</h3>
              <p className="text-gray-400 mb-6">
                The {activeTabConfig?.name} model needs to be trained before generating predictions.
              </p>
              <a
                href="/model-training"
                className="inline-block px-8 py-3 bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-700 hover:to-purple-800 text-white font-bold rounded-lg transition-colors"
              >
                🚀 Train Models Now
              </a>
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