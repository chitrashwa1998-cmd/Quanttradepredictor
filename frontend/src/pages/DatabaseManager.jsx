/**
 * Database Manager page - Complete Streamlit functionality migration
 */

import { useState, useEffect } from 'react';
import Card from '../components/common/Card';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { dataAPI } from '../services/api';

export default function DatabaseManager() {
  const [datasets, setDatasets] = useState([]);
  const [dbInfo, setDbInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState('');
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [datasetData, setDatasetData] = useState(null);
  const [loadingDataset, setLoadingDataset] = useState(false);

  useEffect(() => {
    loadDatabaseInfo();
  }, []);

  const loadDatabaseInfo = async () => {
    try {
      setLoading(true);

      // Load database info
      const dbResponse = await dataAPI.getDatabaseInfo();
      setDbInfo(dbResponse);

      // Load datasets
      const datasetsResponse = await dataAPI.listDatasets();
      setDatasets(datasetsResponse.datasets || []);

    } catch (error) {
      console.error('Error loading database info:', error);
      setMessage(`Error loading database info: ${error.message}`);
      setMessageType('error');
    } finally {
      setLoading(false);
    }
  };

  const handleDatasetSelect = async (datasetName) => {
    if (selectedDataset === datasetName) {
      setSelectedDataset(null);
      setDatasetData(null);
      return;
    }

    setSelectedDataset(datasetName);
    setLoadingDataset(true);

    try {
      const response = await dataAPI.getDataset(datasetName, 100); // Limit to 100 rows
      setDatasetData(response);
    } catch (error) {
      console.error('Error loading dataset:', error);
      setMessage(`Error loading dataset: ${error.message}`);
      setMessageType('error');
    } finally {
      setLoadingDataset(false);
    }
  };

  const handleDeleteDataset = async (datasetName) => {
    if (!window.confirm(`Are you sure you want to delete dataset "${datasetName}"?`)) {
      return;
    }

    try {
      await dataAPI.deleteDataset(datasetName);
      setMessage(`Dataset "${datasetName}" deleted successfully`);
      setMessageType('success');

      // Refresh data
      await loadDatabaseInfo();

      // Clear selected dataset if it was deleted
      if (selectedDataset === datasetName) {
        setSelectedDataset(null);
        setDatasetData(null);
      }
    } catch (error) {
      console.error('Error deleting dataset:', error);
      setMessage(`Error deleting dataset: ${error.message}`);
      setMessageType('error');
    }
  };

  const handleExportDataset = async (datasetName) => {
    try {
      const response = await dataAPI.exportDataset(datasetName);

      // Create download link
      const blob = new Blob([response.data], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${datasetName}.csv`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);

      setMessage(`Dataset "${datasetName}" exported successfully`);
      setMessageType('success');
    } catch (error) {
      console.error('Error exporting dataset:', error);
      setMessage(`Error exporting dataset: ${error.message}`);
      setMessageType('error');
    }
  };

  const handleClearAllData = async () => {
    if (!window.confirm('Are you sure you want to clear ALL data? This action cannot be undone.')) {
      return;
    }

    try {
      await dataAPI.clearAllData();
      setMessage('All data cleared successfully');
      setMessageType('success');
      await loadDatabaseInfo();
      setSelectedDataset(null);
      setDatasetData(null);
    } catch (error) {
      console.error('Error clearing data:', error);
      setMessage(`Error clearing data: ${error.message}`);
      setMessageType('error');
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 p-6">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-white mb-4">🗄️ Database Manager</h1>
          <p className="text-gray-300">Manage your datasets and database operations</p>
        </div>

        {/* Database Overview */}
        <Card className="mb-6">
          <h2 className="text-xl font-semibold text-white mb-4">Database Overview</h2>
          {dbInfo && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-gray-800 p-4 rounded-lg">
                <h3 className="text-lg font-medium text-blue-400">Total Datasets</h3>
                <p className="text-2xl font-bold text-white">{dbInfo.total_datasets || 0}</p>
              </div>
              <div className="bg-gray-800 p-4 rounded-lg">
                <h3 className="text-lg font-medium text-green-400">Total Records</h3>
                <p className="text-2xl font-bold text-white">{dbInfo.total_records || 0}</p>
              </div>
              <div className="bg-gray-800 p-4 rounded-lg">
                <h3 className="text-lg font-medium text-purple-400">Models Trained</h3>
                <p className="text-2xl font-bold text-white">{dbInfo.total_models || 0}</p>
              </div>
            </div>
          )}
        </Card>

        {/* Dataset Management */}
        <Card className="mb-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold text-white">Dataset Management</h2>
            <button
              onClick={handleClearAllData}
              className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors"
            >
              Clear All Data
            </button>
          </div>

          {datasets.length === 0 ? (
            <p className="text-gray-400 text-center py-8">No datasets found. Upload some data to get started.</p>
          ) : (
            <div className="space-y-4">
              {datasets.map((dataset) => (
                <div key={dataset.name} className="bg-gray-800 rounded-lg p-4">
                  <div className="flex justify-between items-center">
                    <div>
                      <h3 className="text-lg font-medium text-white">{dataset.name}</h3>
                      <p className="text-sm text-gray-400">
                        {dataset.rows} rows • Created: {new Date(dataset.created_at).toLocaleString()}
                      </p>
                    </div>
                    <div className="flex space-x-2">
                      <button
                        onClick={() => handleDatasetSelect(dataset.name)}
                        className="px-3 py-1 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
                      >
                        {selectedDataset === dataset.name ? 'Hide' : 'View'}
                      </button>
                      <button
                        onClick={() => handleExportDataset(dataset.name)}
                        className="px-3 py-1 bg-green-600 text-white rounded hover:bg-green-700 transition-colors"
                      >
                        Export
                      </button>
                      <button
                        onClick={() => handleDeleteDataset(dataset.name)}
                        className="px-3 py-1 bg-red-600 text-white rounded hover:bg-red-700 transition-colors"
                      >
                        Delete
                      </button>
                    </div>
                  </div>

                  {selectedDataset === dataset.name && (
                    <div className="mt-4 border-t border-gray-700 pt-4">
                      {loadingDataset ? (
                        <LoadingSpinner size="sm" />
                      ) : datasetData ? (
                        <div className="overflow-x-auto">
                          <table className="min-w-full text-sm">
                            <thead>
                              <tr className="border-b border-gray-700">
                                {datasetData.columns.map((col) => (
                                  <th key={col} className="text-left py-2 px-3 text-gray-300 font-medium">
                                    {col}
                                  </th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {datasetData.data.slice(0, 10).map((row, idx) => (
                                <tr key={idx} className="border-b border-gray-800">
                                  {datasetData.columns.map((col) => (
                                    <td key={col} className="py-2 px-3 text-gray-300">
                                      {row[col]}
                                    </td>
                                  ))}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                          {datasetData.data.length > 10 && (
                            <p className="text-gray-400 text-center mt-2">
                              Showing first 10 rows of {datasetData.data.length}
                            </p>
                          )}
                        </div>
                      ) : (
                        <p className="text-gray-400">No data available</p>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </Card>

        {/* Message Display */}
        {message && (
          <div className={`p-4 rounded-md ${
            messageType === 'success' 
              ? 'bg-green-900 text-green-300 border border-green-700' 
              : 'bg-red-900 text-red-300 border border-red-700'
          }`}>
            {message}
          </div>
        )}
      </div>
    </div>
  );
}