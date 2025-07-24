/**
 * Predictions page - Generate and view model predictions
 */

import { useState, useEffect } from 'react';
import { predictionsAPI, dataAPI } from '../services/api';
import LoadingSpinner from '../components/common/LoadingSpinner';

export default function Predictions() {
  const [models, setModels] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [selectedDataset, setSelectedDataset] = useState('');
  const [predicting, setPredicting] = useState(false);
  const [predictions, setPredictions] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [modelsResponse, datasetsResponse] = await Promise.all([
          predictionsAPI.getModelsStatus(),
          dataAPI.listDatasets()
        ]);
        
        if (modelsResponse.status?.models) {
          const availableModels = Object.entries(modelsResponse.status.models)
            .filter(([_, info]) => info.loaded)
            .map(([name, info]) => ({ name, ...info }));
          setModels(availableModels);
          if (availableModels.length > 0) {
            setSelectedModel(availableModels[0].name);
          }
        }
        
        setDatasets(datasetsResponse);
        if (datasetsResponse.length > 0) {
          setSelectedDataset(datasetsResponse[0].name);
        }
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const handlePrediction = async () => {
    if (!selectedModel || !selectedDataset) {
      setError('Please select both a model and dataset');
      return;
    }

    try {
      setPredicting(true);
      setError(null);
      setPredictions(null);

      // Get latest data from dataset for prediction
      const datasetResponse = await dataAPI.getDataset(selectedDataset, 10);
      const latestData = datasetResponse.data[datasetResponse.data.length - 1];

      const result = await predictionsAPI.predict(selectedModel, latestData);
      setPredictions(result);
    } catch (err) {
      setError(err.response?.data?.error || err.message);
    } finally {
      setPredicting(false);
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <LoadingSpinner size="lg" text="Loading prediction options..." />
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold cyber-text mb-4">Model Predictions</h1>
        <p className="text-gray-300">
          Generate predictions using your trained models
        </p>
      </div>

      {/* Prediction Configuration */}
      <div className="cyber-bg cyber-border rounded-lg p-6">
        <h2 className="text-xl font-semibold cyber-blue mb-6">Prediction Setup</h2>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Model Selection */}
          <div>
            <label className="block text-sm font-semibold text-cyber-yellow mb-3">
              Select Model
            </label>
            {models.length > 0 ? (
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full bg-gray-800 border border-gray-600 rounded-md px-3 py-2 text-white focus:border-cyber-blue focus:outline-none"
              >
                {models.map((model) => (
                  <option key={model.name} value={model.name}>
                    {model.name.charAt(0).toUpperCase() + model.name.slice(1)} 
                    ({model.features?.length || 0} features)
                  </option>
                ))}
              </select>
            ) : (
              <div className="text-center py-4 text-gray-400">
                <p>No trained models available. Please train models first.</p>
              </div>
            )}
          </div>

          {/* Dataset Selection */}
          <div>
            <label className="block text-sm font-semibold text-cyber-yellow mb-3">
              Select Data Source
            </label>
            {datasets.length > 0 ? (
              <select
                value={selectedDataset}
                onChange={(e) => setSelectedDataset(e.target.value)}
                className="w-full bg-gray-800 border border-gray-600 rounded-md px-3 py-2 text-white focus:border-cyber-blue focus:outline-none"
              >
                {datasets.map((dataset) => (
                  <option key={dataset.name} value={dataset.name}>
                    {dataset.name} ({dataset.rows?.toLocaleString()} rows)
                  </option>
                ))}
              </select>
            ) : (
              <div className="text-center py-4 text-gray-400">
                <p>No datasets available. Please upload data first.</p>
              </div>
            )}
          </div>
        </div>

        {/* Prediction Button */}
        <div className="mt-8 text-center">
          <button
            onClick={handlePrediction}
            disabled={predicting || !selectedModel || !selectedDataset || models.length === 0}
            className="cyber-border rounded-md py-3 px-8 text-cyber-green hover:cyber-glow transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {predicting ? 'Generating Prediction...' : 'Generate Prediction'}
          </button>
        </div>
      </div>

      {/* Prediction Progress */}
      {predicting && (
        <div className="cyber-bg cyber-border rounded-lg p-6 slide-in-up">
          <div className="text-center">
            <LoadingSpinner size="lg" text="Generating prediction..." />
          </div>
        </div>
      )}

      {/* Prediction Results */}
      {predictions && (
        <div className="cyber-bg cyber-border rounded-lg p-6 slide-in-up">
          <h2 className="text-xl font-semibold text-cyber-green mb-4">
            🔮 Prediction Results
          </h2>
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h3 className="font-semibold text-cyber-blue mb-2">Model Info</h3>
                <div className="space-y-1 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Model:</span>
                    <span className="text-white capitalize">{predictions.model_name}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Timestamp:</span>
                    <span className="text-white font-mono">
                      {new Date(predictions.timestamp).toLocaleString()}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Status:</span>
                    <span className={predictions.success ? 'text-cyber-green' : 'text-cyber-red'}>
                      {predictions.success ? 'Success' : 'Failed'}
                    </span>
                  </div>
                </div>
              </div>
              
              {predictions.prediction && (
                <div>
                  <h3 className="font-semibold text-cyber-purple mb-2">Prediction Values</h3>
                  <div className="bg-gray-800 rounded-lg p-3">
                    <pre className="text-sm text-gray-300 overflow-auto">
                      {JSON.stringify(predictions.prediction, null, 2)}
                    </pre>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Recent Predictions */}
      <div className="cyber-bg cyber-border rounded-lg p-6">
        <h2 className="text-xl font-semibold cyber-text mb-6">Recent Predictions</h2>
        <div className="text-center text-gray-400 py-8">
          <p>Recent predictions history will be displayed here</p>
          <p className="text-sm mt-2">Generate your first prediction to see results</p>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="cyber-bg border border-cyber-red rounded-lg p-6 slide-in-up">
          <h2 className="text-xl font-semibold text-cyber-red mb-4">
            ❌ Prediction Failed
          </h2>
          <p className="text-gray-300">{error}</p>
        </div>
      )}
    </div>
  );
}