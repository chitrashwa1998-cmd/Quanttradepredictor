/**
 * Model Training page - Robust error-free version
 */

import { useState, useEffect } from 'react';
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
  const loadDatasets = async () => {
    try {
      setLoading(true);
      const response = await dataAPI.getDatasets();
      const datasetList = Array.isArray(response?.data) ? response.data : [];
      setDatasets(datasetList);

      // Auto-select first dataset
      if (datasetList.length > 0 && !selectedDataset) {
        setSelectedDataset(datasetList[0]?.name || '');
      }

      setTrainingStatus(`✅ Loaded ${datasetList.length} datasets`);
    } catch (error) {
      console.error('Error loading datasets:', error);
      setTrainingStatus(`❌ Error loading datasets: ${error?.message || 'Unknown error'}`);
      setDatasets([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDatasets();
  }, []);

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

      if (response?.success) {
        setFeaturesCalculated(prev => ({
          ...prev,
          [activeTab]: true
        }));
        setTrainingStatus(`✅ Calculated ${response.features_calculated || 0} ${activeTab} features`);
      } else {
        setTrainingStatus(`❌ Failed to calculate ${activeTab} features`);
      }
    } catch (error) {
      console.error('Feature calculation error:', error);
      setTrainingStatus(`❌ Error: ${error?.response?.data?.detail || error?.message || 'Unknown error'}`);
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
      setIsTraining(true);
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
        setTrainingStatus(`✅ ${activeTab} model trained successfully! Accuracy: ${(response.accuracy * 100)?.toFixed(2) || 'N/A'}%`);
      } else {
        setTrainingStatus(`❌ Failed to train ${activeTab} model`);
      }
    } catch (error) {
      console.error('Training error:', error);
      setTrainingStatus(`❌ Training error: ${error?.response?.data?.detail || error?.message || 'Unknown error'}`);
    } finally {
      setIsTraining(false);
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-violet-800">
      <div className="container mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-600 mb-4">
            Model Training
          </h1>
          <p className="text-gray-300 text-lg">
            Train machine learning models for market predictions
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
              <label className="block text-gray-300 mb-2">Refresh Datasets:</label>
              <button
                onClick={loadDatasets}
                className="w-full p-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
                disabled={loading}
              >
                🔄 Refresh
              </button>
            </div>
          </div>
        </Card>

        {/* Model Tabs */}
        <Card className="mb-8">
          <h2 className="text-2xl font-bold text-cyan-400 mb-4">🤖 Model Selection</h2>
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
                {featuresCalculated[tab.id] && (
                  <div className="text-xs mt-1 text-green-400">Features Ready</div>
                )}
              </button>
            ))}
          </div>
        </Card>

        {/* Training Configuration */}
        <Card className="mb-8">
          <h2 className="text-2xl font-bold text-cyan-400 mb-4">⚙️ Training Configuration</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-gray-300 mb-2">Train Split Ratio:</label>
              <input
                type="number"
                min="0.1"
                max="0.9"
                step="0.1"
                value={trainingConfig.train_split}
                onChange={(e) => setTrainingConfig(prev => ({
                  ...prev,
                  train_split: parseFloat(e.target.value) || 0.8
                }))}
                className="w-full p-3 bg-gray-800 border border-gray-600 rounded-lg text-white focus:ring-2 focus:ring-cyan-500"
                disabled={loading}
              />
            </div>
            <div>
              <label className="block text-gray-300 mb-2">Max Depth:</label>
              <input
                type="number"
                min="1"
                max="20"
                value={trainingConfig.max_depth}
                onChange={(e) => setTrainingConfig(prev => ({
                  ...prev,
                  max_depth: parseInt(e.target.value) || 6
                }))}
                className="w-full p-3 bg-gray-800 border border-gray-600 rounded-lg text-white focus:ring-2 focus:ring-cyan-500"
                disabled={loading}
              />
            </div>
            <div>
              <label className="block text-gray-300 mb-2">N Estimators:</label>
              <input
                type="number"
                min="10"
                max="1000"
                value={trainingConfig.n_estimators}
                onChange={(e) => setTrainingConfig(prev => ({
                  ...prev,
                  n_estimators: parseInt(e.target.value) || 100
                }))}
                className="w-full p-3 bg-gray-800 border border-gray-600 rounded-lg text-white focus:ring-2 focus:ring-cyan-500"
                disabled={loading}
              />
            </div>
          </div>
        </Card>

        {/* Training Actions */}
        <Card className="mb-8">
          <h2 className="text-2xl font-bold text-cyan-400 mb-4">🚀 Training Actions</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <button
              onClick={calculateFeatures}
              className="p-4 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
              disabled={loading || !selectedDataset}
            >
              🔧 Calculate {activeTab} Features
            </button>
            <button
              onClick={trainModel}
              className="p-4 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors"
              disabled={loading || !selectedDataset || !featuresCalculated[activeTab] || isTraining}
            >
              🚀 Train {activeTab} Model
            </button>
          </div>
        </Card>

        {/* Training Results */}
        {trainingResults[activeTab] && (
          <Card>
            <h2 className="text-2xl font-bold text-cyan-400 mb-4">📊 Training Results - {activeTab}</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="bg-gradient-to-br from-green-900/50 to-emerald-900/50 p-4 rounded-lg border border-green-500/30">
                <div className="text-2xl font-bold text-green-400">
                  {(trainingResults[activeTab]?.accuracy * 100)?.toFixed(2) || 'N/A'}%
                </div>
                <div className="text-gray-300">Accuracy</div>
              </div>
              <div className="bg-gradient-to-br from-blue-900/50 to-cyan-900/50 p-4 rounded-lg border border-blue-500/30">
                <div className="text-2xl font-bold text-blue-400">
                  {trainingResults[activeTab]?.feature_count || 'N/A'}
                </div>
                <div className="text-gray-300">Features</div>
              </div>
              <div className="bg-gradient-to-br from-purple-900/50 to-pink-900/50 p-4 rounded-lg border border-purple-500/30">
                <div className="text-2xl font-bold text-purple-400">
                  {trainingResults[activeTab]?.train_samples || 'N/A'}
                </div>
                <div className="text-gray-300">Train Samples</div>
              </div>
              <div className="bg-gradient-to-br from-yellow-900/50 to-orange-900/50 p-4 rounded-lg border border-yellow-500/30">
                <div className="text-2xl font-bold text-yellow-400">
                  {trainingResults[activeTab]?.test_samples || 'N/A'}
                </div>
                <div className="text-gray-300">Test Samples</div>
              </div>
            </div>
          </Card>
        )}
      </div>
    </div>
  );
};

export default ModelTraining;