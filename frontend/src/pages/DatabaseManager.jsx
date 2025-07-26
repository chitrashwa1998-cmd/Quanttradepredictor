/**
 * Database Manager page - Simplified working version
 */

import { useState, useEffect, useCallback } from 'react';
import Card from '../components/common/Card';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { dataAPI } from '../services/api';

const DatabaseManager = () => {
  const [databaseInfo, setDatabaseInfo] = useState({});
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');
  const [selectedDataset, setSelectedDataset] = useState('');
  const [showConfirmClearAll, setShowConfirmClearAll] = useState(false);

  // Load database information
  const loadDatabaseInfo = useCallback(async () => {
    try {
      setLoading(true);
      
      // Fetch database info with explicit error handling
      let dbInfo = {};
      try {
        const dbInfoResponse = await dataAPI.getDatabaseInfo();
        dbInfo = dbInfoResponse?.data || {};
      } catch (dbError) {
        console.error('Database info fetch error:', dbError);
        dbInfo = {};
      }
      
      // Fetch datasets with explicit error handling
      let datasetList = [];
      try {
        const datasetsResponse = await dataAPI.getDatasets();
        datasetList = Array.isArray(datasetsResponse?.data) ? datasetsResponse.data : [];
      } catch (datasetsError) {
        console.error('Datasets fetch error:', datasetsError);
        datasetList = [];
      }

      setDatabaseInfo(dbInfo);
      setDatasets(datasetList);
      
      setStatus('✅ Database information refreshed');
    } catch (error) {
      console.error('Error loading database info:', error);
      setStatus(`❌ Error: ${error?.message || 'Unknown error'}`);
      setDatabaseInfo({});
      setDatasets([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadDatabaseInfo();
  }, [loadDatabaseInfo]);

  // Clear all data
  const clearAllData = async () => {
    if (!showConfirmClearAll) {
      setShowConfirmClearAll(true);
      return;
    }

    try {
      setLoading(true);
      setStatus('🗑️ Clearing all database data...');
      
      await dataAPI.clearAllData();
      
      // Refresh database info
      await loadDatabaseInfo();
      
      setStatus('✅ All database data cleared successfully');
      setSelectedDataset('');
      setShowConfirmClearAll(false);
    } catch (error) {
      setStatus(`❌ Error clearing data: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Delete specific dataset
  const deleteDataset = async (datasetName) => {
    if (!confirm(`⚠️ Are you sure you want to delete dataset "${datasetName}"? This action cannot be undone!`)) {
      return;
    }

    try {
      setLoading(true);
      setStatus(`🗑️ Deleting dataset: ${datasetName}...`);
      
      await dataAPI.deleteDataset(datasetName);
      
      // Refresh database info
      await loadDatabaseInfo();
      
      setStatus(`✅ Dataset "${datasetName}" deleted successfully`);
      if (selectedDataset === datasetName) {
        setSelectedDataset('');
      }
    } catch (error) {
      setStatus(`❌ Error deleting dataset: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-violet-800">
      <div className="container mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-600 mb-4">
            Database Manager
          </h1>
          <p className="text-gray-300 text-lg">
            Manage your PostgreSQL database, datasets, and models
          </p>
        </div>

        {/* Status */}
        {status && (
          <div className="mb-6">
            <div className={`p-4 rounded-lg ${
              status.includes('✅') ? 'bg-green-900/30 border border-green-500/30 text-green-400' :
              status.includes('❌') ? 'bg-red-900/30 border border-red-500/30 text-red-400' :
              'bg-blue-900/30 border border-blue-500/30 text-blue-400'
            }`}>
              {status}
            </div>
          </div>
        )}

        {loading && (
          <div className="flex justify-center mb-6">
            <LoadingSpinner />
          </div>
        )}

        {/* Database Overview */}
        <Card className="mb-8">
          <h2 className="text-2xl font-bold text-cyan-400 mb-4">📊 Database Overview</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-gradient-to-br from-blue-900/50 to-cyan-900/50 p-4 rounded-lg border border-cyan-500/30">
              <div className="text-2xl font-bold text-cyan-400">{(databaseInfo && typeof databaseInfo.total_datasets === 'number') ? databaseInfo.total_datasets : 0}</div>
              <div className="text-gray-300">Datasets</div>
            </div>
            <div className="bg-gradient-to-br from-green-900/50 to-emerald-900/50 p-4 rounded-lg border border-green-500/30">
              <div className="text-2xl font-bold text-green-400">{(databaseInfo && typeof databaseInfo.total_records === 'number') ? databaseInfo.total_records : 0}</div>
              <div className="text-gray-300">Records</div>
            </div>
            <div className="bg-gradient-to-br from-purple-900/50 to-pink-900/50 p-4 rounded-lg border border-purple-500/30">
              <div className="text-2xl font-bold text-purple-400">{(databaseInfo && typeof databaseInfo.total_trained_models === 'number') ? databaseInfo.total_trained_models : 0}</div>
              <div className="text-gray-300">Trained Models</div>
            </div>
            <div className="bg-gradient-to-br from-yellow-900/50 to-orange-900/50 p-4 rounded-lg border border-yellow-500/30">
              <div className="text-2xl font-bold text-yellow-400">{(databaseInfo && typeof databaseInfo.total_predictions === 'number') ? databaseInfo.total_predictions : 0}</div>
              <div className="text-gray-300">Predictions</div>
            </div>
          </div>
          
          <div className="mt-4 text-sm text-gray-400">
            <p><strong>Backend:</strong> {(databaseInfo && databaseInfo.backend) ? databaseInfo.backend : 'PostgreSQL'}</p>
            <p><strong>Storage Type:</strong> {(databaseInfo && databaseInfo.storage_type) ? databaseInfo.storage_type : 'Row-Based'}</p>
          </div>
        </Card>

        {/* Datasets Management */}
        <Card className="mb-8">
          <h2 className="text-2xl font-bold text-cyan-400 mb-4">📁 Datasets ({datasets.length})</h2>
          
          {datasets.length === 0 ? (
            <div className="text-center py-8 text-gray-400">
              <p>No datasets found. Upload data to create your first dataset.</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-gray-600">
                    <th className="pb-3 text-cyan-400">Dataset Name</th>
                    <th className="pb-3 text-cyan-400">Rows</th>
                    <th className="pb-3 text-cyan-400">Date Range</th>
                    <th className="pb-3 text-cyan-400">Last Updated</th>
                    <th className="pb-3 text-cyan-400">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {datasets.map((dataset, index) => (
                    <tr key={dataset?.name || index} className="border-b border-gray-700 hover:bg-gray-800/30">
                      <td className="py-3 text-white font-medium">{dataset?.name || 'Unknown'}</td>
                      <td className="py-3 text-gray-300">{(dataset?.rows && typeof dataset.rows === 'number') ? dataset.rows.toLocaleString() : 0}</td>
                      <td className="py-3 text-gray-300">
                        {(dataset?.start_date && dataset?.end_date) ? 
                          `${dataset.start_date} to ${dataset.end_date}` : 
                          'N/A'}
                      </td>
                      <td className="py-3 text-gray-300">{dataset?.updated_at || 'N/A'}</td>
                      <td className="py-3">
                        <button
                          onClick={() => deleteDataset(dataset?.name)}
                          className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white rounded text-sm transition-colors"
                          disabled={loading || !dataset?.name}
                        >
                          Delete
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        {/* Database Actions */}
        <Card>
          <h2 className="text-2xl font-bold text-cyan-400 mb-4">⚙️ Database Actions</h2>
          <div className="space-y-4">
            
            <div className="flex items-center justify-between p-4 bg-blue-900/20 border border-blue-500/30 rounded-lg">
              <div>
                <h3 className="text-blue-400 font-bold">Refresh Database Info</h3>
                <p className="text-gray-400 text-sm">Reload database statistics and dataset information</p>
              </div>
              <button
                onClick={loadDatabaseInfo}
                className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded transition-colors"
                disabled={loading}
              >
                Refresh
              </button>
            </div>

            <div className="flex items-center justify-between p-4 bg-red-900/20 border border-red-500/30 rounded-lg">
              <div>
                <h3 className="text-red-400 font-bold">Clear All Data</h3>
                <p className="text-gray-400 text-sm">Remove all datasets, models, and predictions (irreversible)</p>
              </div>
              <div className="flex space-x-2">
                {showConfirmClearAll && (
                  <button
                    onClick={() => setShowConfirmClearAll(false)}
                    className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded transition-colors"
                  >
                    Cancel
                  </button>
                )}
                <button
                  onClick={clearAllData}
                  className={`px-6 py-2 ${showConfirmClearAll ? 'bg-red-700 hover:bg-red-800' : 'bg-red-600 hover:bg-red-700'} text-white rounded transition-colors`}
                  disabled={loading}
                >
                  {showConfirmClearAll ? 'Confirm Clear All' : 'Clear All Data'}
                </button>
              </div>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
};

export default DatabaseManager;