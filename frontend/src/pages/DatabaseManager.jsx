
/**
 * Database Manager page - Complete Streamlit functionality migration with cyberpunk theme
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
  const [confirmDelete, setConfirmDelete] = useState(null);
  const [renameDataset, setRenameDataset] = useState(null);
  const [newName, setNewName] = useState('');

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
      const response = await dataAPI.getDataset(datasetName, 100);
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
    try {
      await dataAPI.deleteDataset(datasetName);
      setMessage(`Dataset "${datasetName}" deleted successfully`);
      setMessageType('success');
      await loadDatabaseInfo();
      if (selectedDataset === datasetName) {
        setSelectedDataset(null);
        setDatasetData(null);
      }
      setConfirmDelete(null);
    } catch (error) {
      console.error('Error deleting dataset:', error);
      setMessage(`Error deleting dataset: ${error.message}`);
      setMessageType('error');
    }
  };

  const handleExportDataset = async (datasetName) => {
    try {
      const response = await dataAPI.exportDataset(datasetName);
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
      <div className="min-h-screen flex items-center justify-center" style={{
        background: 'var(--primary-bg)',
        backgroundImage: `
          radial-gradient(circle at 20% 50%, rgba(0, 255, 255, 0.1) 0%, transparent 50%),
          radial-gradient(circle at 80% 20%, rgba(255, 0, 128, 0.1) 0%, transparent 50%),
          radial-gradient(circle at 40% 80%, rgba(0, 255, 65, 0.05) 0%, transparent 50%)
        `
      }}>
        <div className="cyber-spinner" style={{ width: '60px', height: '60px' }}></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen p-6" style={{
      background: 'var(--primary-bg)',
      backgroundImage: `
        radial-gradient(circle at 20% 50%, rgba(0, 255, 255, 0.1) 0%, transparent 50%),
        radial-gradient(circle at 80% 20%, rgba(255, 0, 128, 0.1) 0%, transparent 50%),
        radial-gradient(circle at 40% 80%, rgba(0, 255, 65, 0.05) 0%, transparent 50%)
      `
    }}>
      <div className="max-w-6xl mx-auto">
        {/* Header - Original Streamlit Style */}
        <div className="trading-header">
          <h1 style={{
            margin: 0,
            fontFamily: 'var(--font-display)',
            fontSize: '2.5rem',
            fontWeight: '900',
            background: 'var(--gradient-text)',
            backgroundClip: 'text',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            textShadow: '0 0 20px var(--shadow-cyan)',
            letterSpacing: '0.1em',
            textTransform: 'uppercase'
          }}>
            💾 DATA CONTROL CENTER
          </h1>
          <p style={{
            fontSize: '1.2rem',
            margin: '1rem 0 0 0',
            color: 'rgba(255,255,255,0.8)',
            fontFamily: 'var(--font-primary)'
          }}>
            Database Management & Storage
          </p>
        </div>

        {/* Database Overview */}
        <div className="cyber-card mb-6">
          <div className="flex justify-between items-center mb-6">
            <h2 className="cyber-subtitle">📊 Database Overview</h2>
            <button
              onClick={loadDatabaseInfo}
              className="cyber-button-secondary px-4 py-2"
            >
              🔄 Refresh
            </button>
          </div>

          {dbInfo && (
            <div>
              {/* Connection Status */}
              <div className="mb-6 p-4 rounded-lg" style={{
                background: 'rgba(0, 255, 255, 0.1)',
                border: '1px solid var(--accent-cyan)'
              }}>
                <div className="flex items-center">
                  <span className="status-online text-2xl mr-3">🟢</span>
                  <div>
                    <h3 className="cyber-subtitle">PostgreSQL Row-Based Connected</h3>
                    <p className="cyber-mono">
                      Backend: {dbInfo.backend || 'PostgreSQL (Row-Based)'} | 
                      Storage: {dbInfo.storage_type || 'Row-Based'} | 
                      Append Support: {dbInfo.supports_append ? '✅' : '❌'}
                    </p>
                  </div>
                </div>
              </div>

              {/* Metrics Grid */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="metric-container">
                  <div className="metric-label">📊 Total Datasets</div>
                  <div className="metric-value">{dbInfo.total_datasets || 0}</div>
                </div>
                <div className="metric-container">
                  <div className="metric-label">📈 Total Records</div>
                  <div className="metric-value">{dbInfo.total_records || 0}</div>
                </div>
                <div className="metric-container">
                  <div className="metric-label">🤖 Models Trained</div>
                  <div className="metric-value">{dbInfo.total_models || 0}</div>
                </div>
                <div className="metric-container">
                  <div className="metric-label">🔮 Predictions</div>
                  <div className="metric-value">{dbInfo.total_predictions || 0}</div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Dataset Management */}
        <div className="cyber-card mb-6">
          <div className="flex justify-between items-center mb-6">
            <h2 className="cyber-subtitle">📈 Saved Datasets</h2>
            <button
              onClick={() => setConfirmDelete('ALL')}
              className="cyber-button px-4 py-2"
              style={{
                background: 'linear-gradient(135deg, #ff0080 0%, #ff4500 100%)',
                boxShadow: '0 4px 15px rgba(255, 0, 128, 0.3)'
              }}
            >
              🗑️ Clear All Data
            </button>
          </div>

          {datasets.length === 0 ? (
            <div className="text-center py-12">
              <div className="cyber-mono text-6xl mb-4 opacity-50">📂</div>
              <p className="cyber-text text-xl">No datasets found in database</p>
              <p className="cyber-mono mt-2">Upload some data to get started</p>
            </div>
          ) : (
            <div className="space-y-4">
              {datasets.map((dataset, index) => (
                <div key={dataset.name} className="cyber-card" style={{
                  background: 'rgba(42, 42, 42, 0.8)',
                  border: '1px solid var(--border)'
                }}>
                  <div className="flex justify-between items-center mb-4">
                    <div className="flex-1">
                      <div className="flex items-center mb-2">
                        <span className="cyber-mono text-2xl mr-3">📊</span>
                        <h3 className="cyber-subtitle text-xl">{dataset.name}</h3>
                        <span className="ml-3 px-2 py-1 rounded text-xs cyber-mono" style={{
                          background: dataset.purpose === 'training' ? 'rgba(0, 255, 255, 0.2)' : 
                                   dataset.purpose === 'pre_seed' ? 'rgba(0, 255, 65, 0.2)' :
                                   dataset.purpose === 'validation' ? 'rgba(255, 140, 0, 0.2)' :
                                   'rgba(255, 0, 128, 0.2)',
                          color: dataset.purpose === 'training' ? 'var(--accent-cyan)' : 
                                dataset.purpose === 'pre_seed' ? 'var(--accent-green)' :
                                dataset.purpose === 'validation' ? 'var(--accent-orange)' :
                                'var(--accent-pink)'
                        }}>
                          {dataset.purpose || 'unknown'}
                        </span>
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 cyber-mono text-sm">
                        <div>
                          <span className="text-gray-400">Rows:</span> <span className="status-online">{dataset.rows}</span>
                        </div>
                        <div>
                          <span className="text-gray-400">Created:</span> <span className="cyber-text">{new Date(dataset.created_at).toLocaleString()}</span>
                        </div>
                        <div>
                          <span className="text-gray-400">Date Range:</span> <span className="cyber-text">
                            {dataset.start_date ? `${dataset.start_date} to ${dataset.end_date}` : 'Not available'}
                          </span>
                        </div>
                      </div>
                    </div>

                    <div className="flex space-x-2 ml-6">
                      <button
                        onClick={() => handleDatasetSelect(dataset.name)}
                        className="cyber-button-secondary px-3 py-1 text-sm"
                      >
                        {selectedDataset === dataset.name ? '👁️ Hide' : '👁️ View'}
                      </button>
                      <button
                        onClick={() => handleExportDataset(dataset.name)}
                        className="cyber-button-secondary px-3 py-1 text-sm"
                        style={{
                          borderColor: 'var(--accent-green)',
                          color: 'var(--accent-green)'
                        }}
                      >
                        📥 Export
                      </button>
                      <button
                        onClick={() => setRenameDataset(dataset.name)}
                        className="cyber-button-secondary px-3 py-1 text-sm"
                        style={{
                          borderColor: 'var(--accent-orange)',
                          color: 'var(--accent-orange)'
                        }}
                      >
                        ✏️ Rename
                      </button>
                      <button
                        onClick={() => setConfirmDelete(dataset.name)}
                        className="cyber-button-secondary px-3 py-1 text-sm"
                        style={{
                          borderColor: 'var(--accent-pink)',
                          color: 'var(--accent-pink)'
                        }}
                      >
                        🗑️ Delete
                      </button>
                    </div>
                  </div>

                  {/* Rename Form */}
                  {renameDataset === dataset.name && (
                    <div className="mt-4 p-4 rounded-lg" style={{
                      background: 'rgba(255, 140, 0, 0.1)',
                      border: '1px solid var(--accent-orange)'
                    }}>
                      <h4 className="cyber-subtitle mb-3">✏️ Rename Dataset</h4>
                      <div className="flex gap-3">
                        <input
                          type="text"
                          value={newName}
                          onChange={(e) => setNewName(e.target.value)}
                          placeholder={dataset.name}
                          className="cyber-input flex-1"
                        />
                        <button
                          onClick={async () => {
                            if (newName && newName !== dataset.name) {
                              // Implement rename logic here
                              setMessage(`Rename functionality not yet implemented`);
                              setMessageType('warning');
                            }
                            setRenameDataset(null);
                            setNewName('');
                          }}
                          className="cyber-button-secondary px-4 py-2"
                        >
                          ✅ Confirm
                        </button>
                        <button
                          onClick={() => {
                            setRenameDataset(null);
                            setNewName('');
                          }}
                          className="cyber-button-secondary px-4 py-2"
                        >
                          ❌ Cancel
                        </button>
                      </div>
                    </div>
                  )}

                  {/* Dataset Preview */}
                  {selectedDataset === dataset.name && (
                    <div className="mt-4 pt-4 border-t border-gray-700">
                      {loadingDataset ? (
                        <div className="flex justify-center py-8">
                          <div className="cyber-spinner"></div>
                        </div>
                      ) : datasetData ? (
                        <div>
                          <h4 className="cyber-subtitle mb-3">📊 Dataset Preview</h4>
                          <div className="overflow-x-auto">
                            <table className="cyber-table">
                              <thead>
                                <tr>
                                  {datasetData.columns.map((col) => (
                                    <th key={col}>{col}</th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody>
                                {datasetData.data.slice(0, 10).map((row, idx) => (
                                  <tr key={idx}>
                                    {datasetData.columns.map((col) => (
                                      <td key={col}>{row[col]}</td>
                                    ))}
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                          {datasetData.data.length > 10 && (
                            <p className="cyber-mono text-center mt-3 text-sm">
                              Showing first 10 rows of {datasetData.data.length}
                            </p>
                          )}
                        </div>
                      ) : (
                        <p className="cyber-text text-center py-4">No data available</p>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Confirmation Modals */}
        {confirmDelete && (
          <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
            <div className="cyber-card max-w-md w-full mx-4">
              <h3 className="cyber-subtitle text-xl mb-4">
                ⚠️ Confirm {confirmDelete === 'ALL' ? 'Clear All Data' : 'Delete Dataset'}
              </h3>
              <p className="cyber-text mb-6">
                {confirmDelete === 'ALL' 
                  ? 'This will permanently delete ALL data from the database including all datasets, models, and predictions. This action cannot be undone.'
                  : `This will permanently delete the dataset "${confirmDelete}". This action cannot be undone.`
                }
              </p>
              <div className="flex gap-3">
                <button
                  onClick={() => {
                    if (confirmDelete === 'ALL') {
                      handleClearAllData();
                    } else {
                      handleDeleteDataset(confirmDelete);
                    }
                  }}
                  className="cyber-button flex-1"
                  style={{
                    background: 'linear-gradient(135deg, #ff0080 0%, #ff4500 100%)'
                  }}
                >
                  ✅ Yes, Delete
                </button>
                <button
                  onClick={() => setConfirmDelete(null)}
                  className="cyber-button-secondary flex-1"
                >
                  ❌ Cancel
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Message Display */}
        {message && (
          <div className={`cyber-alert ${
            messageType === 'success' ? 'cyber-alert-success' : 
            messageType === 'warning' ? 'cyber-alert-warning' : 'cyber-alert-error'
          }`}>
            <div className="flex items-center">
              <span className="mr-3 text-xl">
                {messageType === 'success' ? '✅' : messageType === 'warning' ? '⚠️' : '❌'}
              </span>
              {message}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
