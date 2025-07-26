/**
 * Predictions page - Complete Streamlit functionality migration with 4 model tabs
 * Exact replica of the original Streamlit predictions with comprehensive analysis
 */

import { useState, useEffect, useCallback } from 'react';
import Card from '../components/common/Card';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { dataAPI, predictionsAPI } from '../services/api';
import axios from 'axios';

const Predictions = () => {
  const [datasets, setDatasets] = useState([]);
  const [modelStatus, setModelStatus] = useState({});
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('volatility');
  const [predictions, setPredictions] = useState({});
  const [predictionStatus, setPredictionStatus] = useState('');
  const [freshData, setFreshData] = useState(null);
  const [livePredictionsAvailable, setLivePredictionsAvailable] = useState(false);
  const [selectedDataset, setSelectedDataset] = useState('');
  const [generatingPredictions, setGeneratingPredictions] = useState(false);
  const [currentPredictionData, setCurrentPredictionData] = useState(null);
  const [analysisTab, setAnalysisTab] = useState('chart');

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
        // Prefer training_dataset or livenifty50 or tr
        const preferredDatasets = ['training_dataset', 'livenifty50', 'tr'];
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
          setSelectedDataset(datasetToLoad);
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
    if (!selectedDataset) {
      setPredictionStatus('❌ No dataset selected for predictions');
      return;
    }

    try {
      setGeneratingPredictions(true);
      setPredictionStatus(`🔮 Generating ${modelType} predictions...`);

      const response = await predictionsAPI.generatePredictions({
        model_type: modelType,
        dataset_name: selectedDataset,
        config: {}
      });

      setCurrentPredictionData(response.data);
      setPredictionStatus(`✅ ${modelType.charAt(0).toUpperCase() + modelType.slice(1)} predictions generated successfully!`);
      
    } catch (error) {
      console.error('Prediction error:', error);
      setPredictionStatus(`❌ Prediction failed: ${error.response?.data?.detail || error.message}`);
    } finally {
      setGeneratingPredictions(false);
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
      setCurrentPredictionData(null);
      
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
      name: '📊 Volatility Predictions', 
      icon: '📊',
      description: 'Advanced volatility forecasting with comprehensive analysis'
    },
    { 
      id: 'direction', 
      name: '📈 Direction Predictions', 
      icon: '📈',
      description: 'Bullish/bearish market direction with confidence levels'
    },
    { 
      id: 'profit_probability', 
      name: '💰 Profit Probability', 
      icon: '💰',
      description: 'Probability of profitable trades with risk assessment'
    },
    { 
      id: 'reversal', 
      name: '🔄 Reversal Detection', 
      icon: '🔄',
      description: 'Market reversal patterns and trend changes'
    }
  ];

  const analysisTabs = [
    { id: 'chart', name: '📈 Interactive Chart', icon: '📈' },
    { id: 'data', name: '📋 Detailed Data', icon: '📋' },
    { id: 'distribution', name: '📊 Distribution Analysis', icon: '📊' },
    { id: 'statistics', name: '🔍 Statistical Analysis', icon: '🔍' },
    { id: 'metrics', name: '⚡ Performance Metrics', icon: '⚡' }
  ];

  // Render prediction results based on model type
  const renderPredictionResults = (data) => {
    if (!data || !data.success) {
      return (
        <div className="text-center py-8">
          <p className="text-gray-400">No prediction data available</p>
        </div>
      );
    }

    const { model_type, statistics, predictions, total_predictions } = data;

    return (
      <div className="space-y-6">
        {/* Status Header */}
        <div className="bg-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-cyan-400">
                {model_type.charAt(0).toUpperCase() + model_type.slice(1)} Predictions
              </h3>
              <p className="text-gray-400">
                {total_predictions} predictions generated from authentic data
              </p>
            </div>
            <div className="text-right">
              <p className="text-sm text-gray-400">Dataset: {selectedDataset}</p>
              <p className="text-sm text-gray-400">Status: Active</p>
            </div>
          </div>
        </div>

        {/* Analysis Tabs */}
        <div className="border-b border-gray-700">
          <nav className="flex space-x-8">
            {analysisTabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setAnalysisTab(tab.id)}
                className={`py-2 px-1 border-b-2 font-medium text-sm transition-colors ${
                  analysisTab === tab.id
                    ? 'border-cyan-400 text-cyan-400'
                    : 'border-transparent text-gray-400 hover:text-gray-300 hover:border-gray-300'
                }`}
              >
                <span className="mr-1">{tab.icon}</span>
                {tab.name}
              </button>
            ))}
          </nav>
        </div>

        {/* Analysis Content */}
        <div className="bg-gray-800 rounded-lg p-6">
          {analysisTab === 'chart' && renderChartAnalysis(data)}
          {analysisTab === 'data' && renderDataAnalysis(data)}
          {analysisTab === 'distribution' && renderDistributionAnalysis(data)}
          {analysisTab === 'statistics' && renderStatisticalAnalysis(data)}
          {analysisTab === 'metrics' && renderMetricsAnalysis(data)}
        </div>
      </div>
    );
  };

  // Chart Analysis Tab
  const renderChartAnalysis = (data) => {
    const { predictions, statistics } = data;
    
    return (
      <div className="space-y-6">
        <h4 className="text-lg font-semibold text-cyan-400">Interactive Chart Analysis</h4>
        
        {/* Quick Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {data.model_type === 'volatility' && (
            <>
              <div className="bg-gray-700 p-4 rounded-lg">
                <p className="text-sm text-gray-400">Current Volatility</p>
                <p className="text-xl font-bold text-cyan-400">{statistics.current_volatility?.toFixed(6) || 'N/A'}</p>
              </div>
              <div className="bg-gray-700 p-4 rounded-lg">
                <p className="text-sm text-gray-400">Average</p>
                <p className="text-xl font-bold text-cyan-400">{statistics.mean?.toFixed(6) || 'N/A'}</p>
              </div>
              <div className="bg-gray-700 p-4 rounded-lg">
                <p className="text-sm text-gray-400">Min</p>
                <p className="text-xl font-bold text-green-400">{statistics.min?.toFixed(6) || 'N/A'}</p>
              </div>
              <div className="bg-gray-700 p-4 rounded-lg">
                <p className="text-sm text-gray-400">Max</p>
                <p className="text-xl font-bold text-red-400">{statistics.max?.toFixed(6) || 'N/A'}</p>
              </div>
            </>
          )}
          
          {data.model_type === 'direction' && (
            <>
              <div className="bg-gray-700 p-4 rounded-lg">
                <p className="text-sm text-gray-400">Current Direction</p>
                <p className={`text-xl font-bold ${statistics.current_direction === 'Bullish' ? 'text-green-400' : 'text-red-400'}`}>
                  {statistics.current_direction || 'N/A'}
                </p>
              </div>
              <div className="bg-gray-700 p-4 rounded-lg">
                <p className="text-sm text-gray-400">Current Confidence</p>
                <p className="text-xl font-bold text-cyan-400">{(statistics.current_confidence * 100)?.toFixed(1) || 'N/A'}%</p>
              </div>
              <div className="bg-gray-700 p-4 rounded-lg">
                <p className="text-sm text-gray-400">Bullish %</p>
                <p className="text-xl font-bold text-green-400">{statistics.bullish_percentage?.toFixed(1) || 'N/A'}%</p>
              </div>
              <div className="bg-gray-700 p-4 rounded-lg">
                <p className="text-sm text-gray-400">Avg Confidence</p>
                <p className="text-xl font-bold text-cyan-400">{(statistics.average_confidence * 100)?.toFixed(1) || 'N/A'}%</p>
              </div>
            </>
          )}
        </div>

        {/* Prediction Timeline */}
        <div>
          <h5 className="text-md font-semibold text-white mb-4">Recent Predictions Timeline</h5>
          <div className="bg-gray-700 rounded-lg p-4 max-h-60 overflow-y-auto">
            {predictions.slice(-20).map((pred, idx) => (
              <div key={idx} className="flex justify-between items-center py-2 border-b border-gray-600 last:border-b-0">
                <div>
                  <span className="text-sm text-gray-400">{pred.date} {pred.time}</span>
                </div>
                <div className="text-right">
                  {data.model_type === 'volatility' && (
                    <span className="text-cyan-400 font-mono">{pred.predicted_volatility?.toFixed(6)}</span>
                  )}
                  {data.model_type === 'direction' && (
                    <div>
                      <span className={`font-semibold ${pred.direction === 'Bullish' ? 'text-green-400' : 'text-red-400'}`}>
                        {pred.direction}
                      </span>
                      <span className="text-xs text-gray-400 ml-2">({(pred.confidence * 100).toFixed(1)}%)</span>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  };

  // Data Analysis Tab
  const renderDataAnalysis = (data) => {
    return (
      <div className="space-y-6">
        <h4 className="text-lg font-semibold text-cyan-400">Detailed Prediction Data</h4>
        <div className="bg-gray-700 rounded-lg max-h-96 overflow-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-600 sticky top-0">
              <tr>
                <th className="px-4 py-2 text-left">DateTime</th>
                <th className="px-4 py-2 text-left">Prediction</th>
                <th className="px-4 py-2 text-left">Confidence</th>
              </tr>
            </thead>
            <tbody>
              {data.predictions.slice(-50).map((pred, idx) => (
                <tr key={idx} className="border-b border-gray-600 hover:bg-gray-650">
                  <td className="px-4 py-2 font-mono text-xs">{pred.date} {pred.time}</td>
                  <td className="px-4 py-2">
                    {data.model_type === 'volatility' && (
                      <span className="text-cyan-400 font-mono">{pred.predicted_volatility?.toFixed(6)}</span>
                    )}
                    {data.model_type === 'direction' && (
                      <span className={`font-semibold ${pred.direction === 'Bullish' ? 'text-green-400' : 'text-red-400'}`}>
                        {pred.direction}
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-2">
                    <span className="text-gray-300">{(pred.confidence * 100)?.toFixed(1) || 'N/A'}%</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  // Distribution Analysis Tab
  const renderDistributionAnalysis = (data) => {
    return (
      <div className="space-y-6">
        <h4 className="text-lg font-semibold text-cyan-400">Distribution Analysis</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-gray-700 p-4 rounded-lg">
            <h5 className="font-semibold text-white mb-3">Statistical Summary</h5>
            <div className="space-y-2">
              {data.statistics && Object.entries(data.statistics).map(([key, value]) => (
                <div key={key} className="flex justify-between">
                  <span className="text-gray-400 capitalize">{key.replace('_', ' ')}:</span>
                  <span className="text-cyan-400 font-mono">
                    {typeof value === 'number' ? value.toFixed(6) : value}
                  </span>
                </div>
              ))}
            </div>
          </div>
          
          {data.percentiles && (
            <div className="bg-gray-700 p-4 rounded-lg">
              <h5 className="font-semibold text-white mb-3">Percentile Analysis</h5>
              <div className="space-y-2">
                {Object.entries(data.percentiles).map(([percentile, value]) => (
                  <div key={percentile} className="flex justify-between">
                    <span className="text-gray-400">{percentile}th percentile:</span>
                    <span className="text-cyan-400 font-mono">{parseFloat(value).toFixed(6)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  // Statistical Analysis Tab
  const renderStatisticalAnalysis = (data) => {
    return (
      <div className="space-y-6">
        <h4 className="text-lg font-semibold text-cyan-400">Statistical Analysis</h4>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {data.rolling_stats && (
            <div className="bg-gray-700 p-4 rounded-lg">
              <h5 className="font-semibold text-white mb-3">Rolling Statistics</h5>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-gray-400">Current Mean:</span>
                  <span className="text-cyan-400 font-mono">{data.rolling_stats.current_mean?.toFixed(6) || 'N/A'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Current Std:</span>
                  <span className="text-cyan-400 font-mono">{data.rolling_stats.current_std?.toFixed(6) || 'N/A'}</span>
                </div>
              </div>
            </div>
          )}

          {data.clustering && (
            <div className="bg-gray-700 p-4 rounded-lg">
              <h5 className="font-semibold text-white mb-3">Volatility Clustering</h5>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-gray-400">Total Clusters:</span>
                  <span className="text-cyan-400">{data.clustering.total_clusters}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Avg Length:</span>
                  <span className="text-cyan-400">{data.clustering.avg_cluster_length?.toFixed(1) || 'N/A'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Max Length:</span>
                  <span className="text-cyan-400">{data.clustering.max_cluster_length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Clustering %:</span>
                  <span className="text-cyan-400">{data.clustering.clustering_percentage?.toFixed(1) || 'N/A'}%</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {data.regimes && (
          <div className="bg-gray-700 p-4 rounded-lg">
            <h5 className="font-semibold text-white mb-3">Volatility Regime Distribution</h5>
            <div className="space-y-2">
              {Object.entries(data.regimes).map(([regime, count]) => (
                <div key={regime} className="flex justify-between">
                  <span className="text-gray-400">{regime}:</span>
                  <span className="text-cyan-400">{count}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  // Performance Metrics Tab
  const renderMetricsAnalysis = (data) => {
    return (
      <div className="space-y-6">
        <h4 className="text-lg font-semibold text-cyan-400">Performance Metrics</h4>
        
        <div className="bg-gray-700 p-4 rounded-lg">
          <h5 className="font-semibold text-white mb-3">Model Information</h5>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-400">Model Type:</span>
              <span className="text-cyan-400 capitalize">{data.model_type}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Total Predictions:</span>
              <span className="text-cyan-400">{data.total_predictions}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Dataset:</span>
              <span className="text-cyan-400">{selectedDataset}</span>
            </div>
          </div>
        </div>

        {data.autocorrelation && (
          <div className="bg-gray-700 p-4 rounded-lg">
            <h5 className="font-semibold text-white mb-3">Autocorrelation Analysis</h5>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {Object.entries(data.autocorrelation).slice(0, 8).map(([lag, correlation]) => (
                <div key={lag} className="text-center">
                  <p className="text-xs text-gray-400">Lag {lag}</p>
                  <p className="text-sm font-mono text-cyan-400">{parseFloat(correlation).toFixed(3)}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-black text-white p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-cyan-400 mb-2">🔮 Real-Time Predictions</h1>
          <p className="text-gray-400">Advanced ML Model Predictions - Authentic Data Only</p>
        </div>

        {/* Live Predictions Banner */}
        {livePredictionsAvailable && (
          <div className="mb-6 bg-blue-900 border border-blue-700 rounded-lg p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-blue-200">🎯 <strong>Live Predictions Available!</strong> Real-time direction predictions are being generated from live market data.</p>
              </div>
              <button className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded text-white">
                📡 View Live Data Page
              </button>
            </div>
          </div>
        )}

        {/* Controls */}
        <div className="mb-6 flex flex-wrap gap-4">
          <button
            onClick={clearCache}
            disabled={loading}
            className="bg-red-600 hover:bg-red-700 disabled:bg-gray-600 px-4 py-2 rounded text-white flex items-center"
          >
            🗑️ Clear All Cached Data
          </button>
          
          <div className="flex items-center space-x-2">
            <span className="text-gray-400">Dataset:</span>
            <select
              value={selectedDataset}
              onChange={(e) => setSelectedDataset(e.target.value)}
              className="bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white"
            >
              <option value="">Select dataset...</option>
              {datasets.map((dataset) => (
                <option key={dataset.name} value={dataset.name}>
                  {dataset.name} ({dataset.rows} rows)
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Status */}
        {predictionStatus && (
          <div className="mb-6 p-4 bg-gray-800 rounded-lg">
            <p className="text-gray-300">{predictionStatus}</p>
          </div>
        )}

        {/* Model Tabs */}
        <div className="mb-6">
          <div className="border-b border-gray-700">
            <nav className="flex space-x-8">
              {modelTabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => {
                    setActiveTab(tab.id);
                    setCurrentPredictionData(null);
                    setAnalysisTab('chart');
                  }}
                  className={`py-2 px-1 border-b-2 font-medium text-sm transition-colors ${
                    activeTab === tab.id
                      ? 'border-cyan-400 text-cyan-400'
                      : 'border-transparent text-gray-400 hover:text-gray-300 hover:border-gray-300'
                  }`}
                >
                  <span className="mr-1">{tab.icon}</span>
                  {tab.name}
                </button>
              ))}
            </nav>
          </div>
        </div>

        {/* Active Model Content */}
        <Card>
          <div className="p-6">
            {/* Model Header */}
            <div className="mb-6 flex items-center justify-between">
              <div>
                <h2 className="text-xl font-semibold text-white">
                  {modelTabs.find(tab => tab.id === activeTab)?.name}
                </h2>
                <p className="text-gray-400 mt-1">
                  {modelTabs.find(tab => tab.id === activeTab)?.description}
                </p>
              </div>
              
              {/* Generate Predictions Button */}
              <button
                onClick={() => generatePredictions(activeTab)}
                disabled={generatingPredictions || !selectedDataset}
                className="bg-cyan-600 hover:bg-cyan-700 disabled:bg-gray-600 px-6 py-3 rounded-lg text-white font-semibold flex items-center space-x-2"
              >
                {generatingPredictions ? (
                  <>
                    <LoadingSpinner size="sm" />
                    <span>Generating...</span>
                  </>
                ) : (
                  <>
                    <span>🔮</span>
                    <span>Generate Predictions</span>
                  </>
                )}
              </button>
            </div>

            {/* Model Status Check */}
            {!modelStatus[activeTab]?.loaded && (
              <div className="bg-yellow-900 border border-yellow-700 rounded-lg p-4 mb-6">
                <p className="text-yellow-200">
                  ⚠️ {activeTab.charAt(0).toUpperCase() + activeTab.slice(1)} model not trained. 
                  Please train the model first in the Model Training page.
                </p>
              </div>
            )}

            {/* Prediction Results */}
            {loading ? (
              <div className="flex justify-center py-12">
                <LoadingSpinner />
              </div>
            ) : currentPredictionData && currentPredictionData.model_type === activeTab ? (
              renderPredictionResults(currentPredictionData)
            ) : (
              <div className="text-center py-12">
                <div className="text-gray-400 mb-4">
                  <span className="text-4xl">🔮</span>
                </div>
                <h3 className="text-lg font-semibold text-gray-300 mb-2">
                  Ready to Generate {activeTab.charAt(0).toUpperCase() + activeTab.slice(1)} Predictions
                </h3>
                <p className="text-gray-400 mb-6">
                  Click "Generate Predictions" to create comprehensive analysis with authentic data
                </p>
                <div className="text-sm text-gray-500">
                  {selectedDataset ? `Using dataset: ${selectedDataset}` : 'Select a dataset to begin'}
                </div>
              </div>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
};

export default Predictions;