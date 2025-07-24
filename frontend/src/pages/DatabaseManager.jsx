/**
 * Database Manager page - Manage datasets and database operations
 */

import { useState, useEffect } from 'react';
import { dataAPI } from '../services/api';
import LoadingSpinner from '../components/common/LoadingSpinner';

export default function DatabaseManager() {
  const [datasets, setDatasets] = useState([]);
  const [dbInfo, setDbInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [datasetsResponse, dbResponse] = await Promise.all([
          dataAPI.listDatasets(),
          dataAPI.getDatabaseInfo()
        ]);
        
        setDatasets(datasetsResponse);
        setDbInfo(dbResponse.info);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const handleDeleteDataset = async (datasetName) => {
    if (!confirm(`Are you sure you want to delete dataset "${datasetName}"?`)) {
      return;
    }

    try {
      await dataAPI.deleteDataset(datasetName);
      setDatasets(datasets.filter(d => d.name !== datasetName));
    } catch (err) {
      setError(err.message);
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <LoadingSpinner size="lg" text="Loading database information..." />
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold cyber-text mb-4">Database Manager</h1>
        <p className="text-gray-300">
          Manage your datasets and database operations
        </p>
      </div>

      {/* Database Info */}
      {dbInfo && (
        <div className="cyber-bg cyber-border rounded-lg p-6">
          <h2 className="text-xl font-semibold cyber-blue mb-6">Database Information</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-gray-800 rounded-lg p-4">
              <h3 className="text-cyber-yellow font-semibold mb-2">Storage</h3>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Type:</span>
                  <span className="text-white">{dbInfo.backend}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Storage:</span>
                  <span className="text-white">{dbInfo.storage_type}</span>
                </div>
              </div>
            </div>
            
            <div className="bg-gray-800 rounded-lg p-4">
              <h3 className="text-cyber-green font-semibold mb-2">Data</h3>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Datasets:</span>
                  <span className="text-white">{dbInfo.total_datasets}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Records:</span>
                  <span className="text-white">{dbInfo.total_records?.toLocaleString()}</span>
                </div>
              </div>
            </div>
            
            <div className="bg-gray-800 rounded-lg p-4">
              <h3 className="text-cyber-purple font-semibold mb-2">Models</h3>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Trained:</span>
                  <span className="text-white">{dbInfo.total_trained_models}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Predictions:</span>
                  <span className="text-white">{dbInfo.total_predictions}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Datasets Table */}
      <div className="cyber-bg cyber-border rounded-lg p-6">
        <h2 className="text-xl font-semibold cyber-blue mb-6">Datasets</h2>
        
        {datasets.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left py-3 px-4 font-semibold text-cyber-yellow">Name</th>
                  <th className="text-left py-3 px-4 font-semibold text-cyber-yellow">Rows</th>
                  <th className="text-left py-3 px-4 font-semibold text-cyber-yellow">Date Range</th>
                  <th className="text-left py-3 px-4 font-semibold text-cyber-yellow">Created</th>
                  <th className="text-left py-3 px-4 font-semibold text-cyber-yellow">Actions</th>
                </tr>
              </thead>
              <tbody>
                {datasets.map((dataset) => (
                  <tr key={dataset.name} className="border-b border-gray-800 hover:bg-gray-800">
                    <td className="py-3 px-4">
                      <span className="text-white font-mono">{dataset.name}</span>
                    </td>
                    <td className="py-3 px-4">
                      <span className="text-gray-300">{dataset.rows?.toLocaleString()}</span>
                    </td>
                    <td className="py-3 px-4">
                      <span className="text-gray-300">
                        {dataset.start_date && dataset.end_date
                          ? `${dataset.start_date} to ${dataset.end_date}`
                          : 'Unknown'}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <span className="text-gray-300">
                        {dataset.created_at 
                          ? new Date(dataset.created_at).toLocaleDateString()
                          : 'Unknown'}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <button
                        onClick={() => handleDeleteDataset(dataset.name)}
                        className="text-cyber-red hover:text-red-400 transition-colors"
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center text-gray-400 py-8">
            <p>No datasets found</p>
            <p className="text-sm mt-2">Upload data to get started</p>
          </div>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <div className="cyber-bg border border-cyber-red rounded-lg p-6">
          <h2 className="text-xl font-semibold text-cyber-red mb-4">Error</h2>
          <p className="text-gray-300">{error}</p>
        </div>
      )}
    </div>
  );
}