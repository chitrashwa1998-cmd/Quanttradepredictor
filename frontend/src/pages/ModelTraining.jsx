/**
 * Model Training page - Simplified working version
 */

import { useState, useEffect, useCallback } from 'react';
import Card from '../components/common/Card';
import LoadingSpinner from '../components/common/LoadingSpinner';
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
  const [trainingResults, setTrainingResults] = useState({});
  const [trainingStatus, setTrainingStatus] = useState('');
  const [featuresCalculated, setFeaturesCalculated] = useState({});
  const [isTraining, setIsTraining] = useState(false);

  // Model configuration
  const modelTabs = [
    { id: 'volatility', name: 'Volatility', icon: '📈', color: 'blue' },
    { id: 'direction', name: 'Direction', icon: '🎯', color: 'green' },
    { id: 'profit_probability', name: 'Profit Probability', icon: '💰', color: 'yellow' },
    { id: 'reversal', name: 'Reversal', icon: '🔄', color: 'purple' }
  ];

  // Load datasets on component mount
  const loadDatasets = useCallback(async () => {
    try {
      setLoading(true);
      const response = await dataAPI.getDatasets().catch(() => ({ data: [] }));
      const datasetList = response.data || [];
      setDatasets(datasetList);

      // Auto-select first dataset
      if (datasetList.length > 0 && !selectedDataset) {
        setSelectedDataset(datasetList[0].name);
      }

      setTrainingStatus(`✅ Loaded ${datasetList.length} datasets`);
    } catch (error) {
      console.error('Error loading datasets:', error);
      setTrainingStatus(`❌ Error loading datasets: ${error.message}`);
      setDatasets([]);
    } finally {
      setLoading(false);
    }
  }, [selectedDataset]);

  useEffect(() => {
    loadDatasets();
  }, [loadDatasets]);

  // Calculate features
  const calculateFeatures = async () => {
    if (!selectedDataset) {
      setTrainingStatus('❌ Please select a dataset first');
      return;
    }

    try {
      setLoading(true);
      setTrainingStatus(`🔧 Calculating ${activeTab} features...`);

      const response = await modelsAPI.calculateFeatures({
        dataset_name: selectedDataset,
        model_type: activeTab
      });

      if (response.success) {
        setFeaturesCalculated(prev => ({
          ...prev,
          [activeTab]: true
        }));
        setTrainingStatus(`✅ Calculated ${response.feature_count || 0} ${activeTab} features`);
      } else {
        setTrainingStatus(`❌ Failed to calculate ${activeTab} features`);
      }
    } catch (error) {
      console.error('Feature calculation error:', error);
      setTrainingStatus(`❌ Error: ${error.response?.data?.detail || error.message}`);
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
      setTrainingStatus(`❌ Please calculate ${activeTab} features first`);
      return;
    }

    try {
      setIsTraining(true);
      setTrainingStatus(`🎯 Training ${activeTab} model...`);

      const response = await modelsAPI.trainModel({
        model_type: activeTab,
        dataset_name: selectedDataset,
        config: trainingConfig
      });

      if (response.success) {
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
      setTrainingStatus(`❌ Error: ${error.response?.data?.detail || error.message}`);
    } finally {
      setIsTraining(false);
    }
  };

  const activeTabColor = modelTabs.find(tab => tab.id === activeTab)?.color || 'blue';
  const currentResults = trainingResults[activeTab];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-violet-800">
      <div className="container mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-600 mb-4">
            Model Training
          </h1>
          <p className="text-gray-300 text-lg">
            Train advanced machine learning models for market prediction
          </p>
        </div>

        {/* Status */}
        {trainingStatus && (
          <div className="mb-6">
            <div className={`p-4 rounded-lg ${
              trainingStatus.includes('✅') ? 'bg-green-900/30 border border-green-500/30 text-green-400' :
              trainingStatus.includes('❌') ? 'bg-red-900/30 border border-red-500/30 text-red-400' :
              'bg-blue-900/30 border border-blue-500/30 text-blue-400'
            }`}>
              {trainingStatus}
            </div>
          </div>
        )}

        {/* Model Tabs */}
        <Card className="mb-6">
          <div className="flex flex-wrap gap-2 mb-6">
            {modelTabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-2 rounded-lg border transition-colors flex items-center space-x-2 ${
                  activeTab === tab.id
                    ? `bg-${tab.color}-600 border-${tab.color}-500 text-white`
                    : `border-gray-600 text-gray-300 hover:border-${tab.color}-500 hover:text-${tab.color}-400`
                }`}
              >
                <span>{tab.icon}</span>
                <span>{tab.name}</span>
                {featuresCalculated[tab.id] && <span className="text-green-400">✓</span>}
              </button>
            ))}
          </div>

          {/* Dataset Selection */}
          <div className="mb-6">
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

          {/* Training Configuration */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div>
              <label className="block text-cyan-400 font-medium mb-2">Train Split:</label>
              <input
                type="number"
                min="0.1"
                max="0.9"
                step="0.1"
                value={trainingConfig.train_split}
                onChange={(e) => setTrainingConfig(prev => ({ ...prev, train_split: parseFloat(e.target.value) }))}
                className="w-full p-3 bg-gray-800 border border-gray-600 rounded-lg text-white focus:border-cyan-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-cyan-400 font-medium mb-2">Max Depth:</label>
              <input
                type="number"
                min="1"
                max="20"
                value={trainingConfig.max_depth}
                onChange={(e) => setTrainingConfig(prev => ({ ...prev, max_depth: parseInt(e.target.value) }))}
                className="w-full p-3 bg-gray-800 border border-gray-600 rounded-lg text-white focus:border-cyan-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-cyan-400 font-medium mb-2">N Estimators:</label>
              <input
                type="number"
                min="10"
                max="1000"
                step="10"
                value={trainingConfig.n_estimators}
                onChange={(e) => setTrainingConfig(prev => ({ ...prev, n_estimators: parseInt(e.target.value) }))}
                className="w-full p-3 bg-gray-800 border border-gray-600 rounded-lg text-white focus:border-cyan-500 focus:outline-none"
              />
            </div>
          </div>

          {/* Action Buttons */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <button
              onClick={calculateFeatures}
              disabled={loading || !selectedDataset}
              className={`px-6 py-3 bg-cyan-600 hover:bg-cyan-700 text-white font-medium rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2`}
            >
              {loading && <LoadingSpinner size="sm" />}
              <span>Calculate Features</span>
            </button>
            <button
              onClick={trainModel}
              disabled={loading || isTraining || !selectedDataset || !featuresCalculated[activeTab]}
              className={`px-6 py-3 bg-${activeTabColor}-600 hover:bg-${activeTabColor}-700 text-white font-medium rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2`}
            >
              {isTraining && <LoadingSpinner size="sm" />}
              <span>
                {isTraining ? `Training ${activeTab}...` : `Train ${activeTab} Model`}
              </span>
            </button>
          </div>
        </Card>

        {/* Training Results */}
        {currentResults && (
          <Card>
            <h2 className={`text-2xl font-bold text-${activeTabColor}-400 mb-4`}>
              {modelTabs.find(tab => tab.id === activeTab)?.icon} {activeTab.charAt(0).toUpperCase() + activeTab.slice(1)} Training Results
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {currentResults.mse !== undefined && (
                <div className="bg-gray-800/50 p-4 rounded-lg">
                  <div className="text-sm text-gray-400">MSE</div>
                  <div className="text-2xl font-bold text-white">{currentResults.mse.toFixed(6)}</div>
                </div>
              )}
              {currentResults.r2_score !== undefined && (
                <div className="bg-gray-800/50 p-4 rounded-lg">
                  <div className="text-sm text-gray-400">R² Score</div>
                  <div className="text-2xl font-bold text-white">{currentResults.r2_score.toFixed(4)}</div>
                </div>
              )}
              {currentResults.feature_count !== undefined && (
                <div className="bg-gray-800/50 p-4 rounded-lg">
                  <div className="text-sm text-gray-400">Features</div>
                  <div className="text-2xl font-bold text-white">{currentResults.feature_count}</div>
                </div>
              )}
              {currentResults.training_samples !== undefined && (
                <div className="bg-gray-800/50 p-4 rounded-lg">
                  <div className="text-sm text-gray-400">Training Samples</div>
                  <div className="text-2xl font-bold text-white">{currentResults.training_samples.toLocaleString()}</div>
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

export default ModelTraining;