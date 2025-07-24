/**
 * Model Training page - Train and manage ML models
 */

import { useState, useEffect } from 'react';
import { modelsAPI, dataAPI } from '../services/api';
import LoadingSpinner from '../components/common/LoadingSpinner';

const MODEL_TYPES = [
  { id: 'volatility', name: 'Volatility Model', description: 'Predicts market volatility patterns' },
  { id: 'direction', name: 'Direction Model', description: 'Predicts price movement direction' },
  { id: 'profit_probability', name: 'Profit Probability', description: 'Calculates profit probability for trades' },
  { id: 'reversal', name: 'Reversal Model', description: 'Detects potential market reversals' }
];

export default function ModelTraining() {
  const [datasets, setDatasets] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [selectedDataset, setSelectedDataset] = useState('');
  const [training, setTraining] = useState(false);
  const [trainingResult, setTrainingResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchDatasets = async () => {
      try {
        const response = await dataAPI.listDatasets();
        setDatasets(response);
        if (response.length > 0) {
          setSelectedDataset(response[0].name);
        }
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchDatasets();
  }, []);

  const handleTraining = async () => {
    if (!selectedModel || !selectedDataset) {
      setError('Please select both a model and dataset');
      return;
    }

    try {
      setTraining(true);
      setError(null);
      setTrainingResult(null);

      const result = await modelsAPI.trainModel(selectedModel, selectedDataset);
      setTrainingResult(result);
    } catch (err) {
      setError(err.response?.data?.error || err.message);
    } finally {
      setTraining(false);
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <LoadingSpinner size="lg" text="Loading training options..." />
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold cyber-text mb-4">Model Training</h1>
        <p className="text-gray-300">
          Train machine learning models on your uploaded datasets
        </p>
      </div>

      {/* Training Configuration */}
      <div className="cyber-bg cyber-border rounded-lg p-6">
        <h2 className="text-xl font-semibold cyber-blue mb-6">Training Configuration</h2>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Model Selection */}
          <div>
            <label className="block text-sm font-semibold text-cyber-yellow mb-3">
              Select Model Type
            </label>
            <div className="space-y-3">
              {MODEL_TYPES.map((model) => (
                <div
                  key={model.id}
                  className={`cursor-pointer rounded-lg p-4 border transition-all ${
                    selectedModel === model.id
                      ? 'border-cyber-blue cyber-glow bg-cyber-blue bg-opacity-10'
                      : 'border-gray-600 hover:border-cyber-blue'
                  }`}
                  onClick={() => setSelectedModel(model.id)}
                >
                  <div className="flex items-start">
                    <input
                      type="radio"
                      name="model"
                      value={model.id}
                      checked={selectedModel === model.id}
                      onChange={() => setSelectedModel(model.id)}
                      className="mt-1 mr-3"
                    />
                    <div>
                      <h3 className="font-medium text-white">{model.name}</h3>
                      <p className="text-sm text-gray-400">{model.description}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Dataset Selection */}
          <div>
            <label className="block text-sm font-semibold text-cyber-yellow mb-3">
              Select Training Dataset
            </label>
            {datasets.length > 0 ? (
              <div className="space-y-3">
                {datasets.map((dataset) => (
                  <div
                    key={dataset.name}
                    className={`cursor-pointer rounded-lg p-4 border transition-all ${
                      selectedDataset === dataset.name
                        ? 'border-cyber-purple cyber-glow bg-cyber-purple bg-opacity-10'
                        : 'border-gray-600 hover:border-cyber-purple'
                    }`}
                    onClick={() => setSelectedDataset(dataset.name)}
                  >
                    <div className="flex items-start">
                      <input
                        type="radio"
                        name="dataset"
                        value={dataset.name}
                        checked={selectedDataset === dataset.name}
                        onChange={() => setSelectedDataset(dataset.name)}
                        className="mt-1 mr-3"
                      />
                      <div className="flex-1">
                        <h3 className="font-medium text-white">{dataset.name}</h3>
                        <div className="text-sm text-gray-400 mt-1">
                          <span>{dataset.rows?.toLocaleString()} rows</span>
                          {dataset.start_date && dataset.end_date && (
                            <span className="ml-4">
                              {dataset.start_date} to {dataset.end_date}
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-400">
                <p>No datasets available. Please upload data first.</p>
              </div>
            )}
          </div>
        </div>

        {/* Training Button */}
        <div className="mt-8 text-center">
          <button
            onClick={handleTraining}
            disabled={training || !selectedModel || !selectedDataset || datasets.length === 0}
            className="cyber-border rounded-md py-3 px-8 text-cyber-green hover:cyber-glow transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {training ? 'Training Model...' : 'Start Training'}
          </button>
        </div>
      </div>

      {/* Training Progress */}
      {training && (
        <div className="cyber-bg cyber-border rounded-lg p-6 slide-in-up">
          <div className="text-center">
            <LoadingSpinner size="lg" text="Training model, please wait..." />
            <p className="text-gray-400 mt-4">
              This may take several minutes depending on dataset size
            </p>
          </div>
        </div>
      )}

      {/* Training Results */}
      {trainingResult && (
        <div className="cyber-bg cyber-border rounded-lg p-6 slide-in-up">
          <h2 className="text-xl font-semibold text-cyber-green mb-4">
            ✅ Training Completed
          </h2>
          <div className="space-y-4">
            <div>
              <h3 className="font-semibold text-cyber-blue mb-2">Model: {selectedModel}</h3>
              <div className="bg-gray-800 rounded-lg p-4">
                <pre className="text-sm text-gray-300 overflow-auto">
                  {JSON.stringify(trainingResult.results, null, 2)}
                </pre>
              </div>
            </div>
            <div className="pt-4 border-t border-gray-700">
              <p className="text-sm text-gray-400">
                Model trained successfully and saved to database.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="cyber-bg border border-cyber-red rounded-lg p-6 slide-in-up">
          <h2 className="text-xl font-semibold text-cyber-red mb-4">
            ❌ Training Failed
          </h2>
          <p className="text-gray-300">{error}</p>
        </div>
      )}
    </div>
  );
}