/**
 * Database Manager page - Complete Streamlit functionality migration
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
  const [dataPreview, setDataPreview] = useState(null);
  const [showRawDebugView, setShowRawDebugView] = useState(false);
  const [rawDatabaseKeys, setRawDatabaseKeys] = useState([]);
  const [selectedKey, setSelectedKey] = useState('');
  const [keyContent, setKeyContent] = useState(null);
  const [showConfirmClearAll, setShowConfirmClearAll] = useState(false);
  const [modelResults, setModelResults] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [renameMode, setRenameMode] = useState({});
  const [newDatasetName, setNewDatasetName] = useState('');

  // Load database information
  const loadDatabaseInfo = useCallback(async () => {
    try {
      setLoading(true);

      const [dbInfoResponse, datasetsResponse] = await Promise.all([
        dataAPI.getDatabaseInfo(),
        dataAPI.getDatasets()
      ]);

      const dbInfo = dbInfoResponse.data || {};
      setDatabaseInfo(dbInfo);
      setDatasets(datasetsResponse.data || []);

      // Load model results and predictions
      const availableKeys = dbInfo.available_keys || [];
      const modelKeys = availableKeys.filter(key => key.startsWith('model_results_'));
      const predKeys = availableKeys.filter(key => key.startsWith('predictions_'));

      setModelResults(modelKeys.map(key => key.replace('model_results_', '')));
      setPredictions(predKeys.map(key => key.replace('predictions_', '')));
      setRawDatabaseKeys(availableKeys);

      setStatus('✅ Database information refreshed');
    } catch (error) {
      setStatus(`❌ Error loading database info: ${error.response?.data?.detail || error.message}`);
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
      setDataPreview(null);
      setSelectedDataset('');
      setShowConfirmClearAll(false);
    } catch (error) {
      setStatus(`❌ Error clearing data: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Clean data mode - preserve only user data
  const activateCleanDataMode = async () => {
    if (!confirm('🧹 Activate clean data mode? This will keep only your uploaded data and remove all metadata overhead.')) {
      return;
    }

    try {
      setLoading(true);
      setStatus('🧹 Activating clean data mode...');

      await dataAPI.cleanDataMode();

      // Refresh database info
      await loadDatabaseInfo();

      setStatus('✅ Clean data mode activated! Only your data remains.');
    } catch (error) {
      setStatus(`❌ Error activating clean mode: ${error.response?.data?.detail || error.message}`);
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
        setDataPreview(null);
      }
    } catch (error) {
      setStatus(`❌ Error deleting dataset: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Rename dataset
  const renameDataset = async (oldName, newName) => {
    if (!newName || newName === oldName) {
      setStatus('❌ Please enter a different name');
      return;
    }

    try {
      setLoading(true);
      setStatus(`✏️ Renaming dataset from "${oldName}" to "${newName}"...`);

      await dataAPI.renameDataset(oldName, newName);

      // Refresh database info
      await loadDatabaseInfo();

      setStatus(`✅ Dataset renamed from "${oldName}" to "${newName}" successfully`);
      setRenameMode({});
      setNewDatasetName('');
    } catch (error) {
      setStatus(`❌ Error renaming dataset: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Preview dataset
  const previewDataset = async (datasetName) => {
    try {
      setLoading(true);
      setStatus(`📊 Loading preview for ${datasetName}...`);

      const response = await dataAPI.loadDataset(datasetName, { limit: 100 });
      setDataPreview(response.data);
      setSelectedDataset(datasetName);
      setStatus(`✅ Loaded preview: ${response.data?.length || 0} records`);
    } catch (error) {
      setStatus(`❌ Error loading preview: ${error.response?.data?.detail || error.message}`);
      setDataPreview(null);
    } finally {
      setLoading(false);
    }
  };

  // Load dataset to session
  const loadDatasetToSession = async (datasetName) => {
    try {
      setLoading(true);
      setStatus(`📥 Loading ${datasetName} to session...`);

      const response = await dataAPI.loadDataset(datasetName);
      // Store in session storage for other components to use
      sessionStorage.setItem('currentDataset', JSON.stringify(response.data));
      sessionStorage.setItem('currentDatasetName', datasetName);

      setStatus(`✅ Dataset "${datasetName}" loaded to session`);
    } catch (error) {
      setStatus(`❌ Error loading dataset to session: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Export dataset
  const exportDataset = async (datasetName) => {
    try {
      setLoading(true);
      setStatus(`📤 Exporting ${datasetName}...`);

      const response = await dataAPI.exportDataset(datasetName);

      // Create download link
      const blob = new Blob([response.data], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${datasetName}_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.csv`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);

      setStatus(`✅ Dataset "${datasetName}" exported successfully`);
    } catch (error) {
      setStatus(`❌ Error exporting dataset: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Delete model results
  const deleteModelResults = async (modelName) => {
    if (!confirm(`⚠️ Delete model results for "${modelName}"?`)) {
      return;
    }

    try {
      setLoading(true);
      setStatus(`🗑️ Deleting model results for ${modelName}...`);

      await dataAPI.deleteModelResults(modelName);
      await loadDatabaseInfo();

      setStatus(`✅ Deleted ${modelName} model results`);
    } catch (error) {
      setStatus(`❌ Error deleting model results: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Delete predictions
  const deletePredictions = async (modelName) => {
    if (!confirm(`⚠️ Delete predictions for "${modelName}"?`)) {
      return;
    }

    try {
      setLoading(true);
      setStatus(`🗑️ Deleting predictions for ${modelName}...`);

      await dataAPI.deletePredictions(modelName);
      await loadDatabaseInfo();

      setStatus(`✅ Deleted ${modelName} predictions`);
    } catch (error) {
      setStatus(`❌ Error deleting predictions: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Clear session data
  const clearSessionData = () => {
    sessionStorage.clear();
    localStorage.clear();
    setStatus('✅ Session data cleared successfully!');
  };

  // Sync metadata
  const syncMetadata = async () => {
    try {
      setLoading(true);
      setStatus('🔄 Syncing metadata...');

      await dataAPI.syncMetadata();
      await loadDatabaseInfo();

      setStatus('✅ Metadata synced successfully!');
    } catch (error) {
      setStatus(`❌ Error syncing metadata: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Clean database (keep only main dataset)
  const cleanDatabase = async () => {
    if (!confirm('🧹 Clean database? This will remove inconsistent data and keep only main dataset.')) {
      return;
    }

    try {
      setLoading(true);
      setStatus('🧹 Cleaning database...');

      await dataAPI.cleanDatabase();
      await loadDatabaseInfo();

      setStatus('✅ Database cleaned! Only main_dataset data remains.');
    } catch (error) {
      setStatus(`❌ Error cleaning database: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // View key content (for raw debug view)
  const viewKeyContent = async (key) => {
    try {
      setLoading(true);
      setStatus(`🔍 Loading content for key: ${key}...`);

      const response = await dataAPI.getKeyContent(key);
      setKeyContent(response.data);
      setSelectedKey(key);
      setStatus(`✅ Loaded content for key: ${key}`);
    } catch (error) {
      setStatus(`❌ Error loading key content: ${error.response?.data?.detail || error.message}`);
      setKeyContent(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto px-6 py-8">
      {/* Header */}
      <div className="trading-header mb-8">
        <h1 style={{
          margin: '0',
          background: 'var(--gradient-primary)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          backgroundClip: 'text',
          fontFamily: 'var(--font-display)',
          fontSize: '2.5rem'
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
      <Card style={{ marginBottom: '2rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
          <h2 style={{
            color: 'var(--accent-cyan)',
            fontFamily: 'var(--font-display)',
            fontSize: '1.5rem',
            margin: '0'
          }}>
            📊 Database Overview
          </h2>
          <button
            onClick={loadDatabaseInfo}
            disabled={loading}
            style={{
              padding: '0.5rem 1rem',
              background: loading ? 'var(--bg-secondary)' : 'var(--gradient-primary)',
              border: '1px solid var(--border-hover)',
              borderRadius: '6px',
              color: 'white',
              fontFamily: 'var(--font-primary)',
              fontWeight: '600',
              cursor: loading ? 'not-allowed' : 'pointer',
              fontSize: '0.9rem'
            }}
          >
            {loading ? '⏳ Refreshing...' : '🔄 Refresh'}
          </button>
        </div>

        {/* Database Status */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <div>
            <h3 style={{ color: 'var(--accent-gold)', margin: '0 0 1rem 0' }}>
              Database Connection
            </h3>
            {databaseInfo.database_type === 'postgresql_row_based' ? (
              <div style={{
                background: 'rgba(0, 255, 0, 0.05)',
                border: '1px solid rgba(0, 255, 0, 0.2)',
                borderRadius: '8px',
                padding: '1rem'
              }}>
                <p style={{ color: '#51cf66', margin: '0 0 0.5rem 0', fontWeight: '600' }}>
                  🟢 PostgreSQL Row-Based Connected
                </p>
                <p style={{ color: 'var(--text-secondary)', margin: '0', fontSize: '0.9rem' }}>
                  Backend: {databaseInfo.backend || 'PostgreSQL (Row-Based)'}
                </p>
                <p style={{ color: 'var(--text-secondary)', margin: '0', fontSize: '0.9rem' }}>
                  Storage: {databaseInfo.storage_type || 'Row-Based'}
                </p>
              </div>
            ) : (
              <div style={{
                background: 'rgba(255, 165, 0, 0.05)',
                border: '1px solid rgba(255, 165, 0, 0.2)',
                borderRadius: '8px',
                padding: '1rem'
              }}>
                <p style={{ color: '#ffa500', margin: '0', fontWeight: '600' }}>
                  📊 Database Type: {databaseInfo.database_type || 'Unknown'}
                </p>
              </div>
            )}
          </div>

          <div>
            <h3 style={{ color: 'var(--accent-gold)', margin: '0 0 1rem 0' }}>
              Your Data Only
            </h3>
            <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
              <button
                onClick={activateCleanDataMode}
                disabled={loading || !datasets.length}
                style={{
                  padding: '0.75rem 1rem',
                  background: loading || !datasets.length ? 'var(--bg-secondary)' : '#17a2b8',
                  border: '1px solid #17a2b8',
                  borderRadius: '6px',
                  color: 'white',
                  fontFamily: 'var(--font-primary)',
                  fontWeight: '600',
                  cursor: loading || !datasets.length ? 'not-allowed' : 'pointer',
                  fontSize: '0.9rem'
                }}
                title="Show only your uploaded data, remove all metadata overhead"
              >
                🧹 Clean Data Mode
              </button>
            </div>
          </div>
        </div>

        {/* Database Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div style={{
            background: 'rgba(0, 255, 255, 0.05)',
            border: '1px solid rgba(0, 255, 255, 0.2)',
            borderRadius: '8px',
            padding: '1.5rem',
            textAlign: 'center'
          }}>
            <div style={{
              color: 'var(--accent-cyan)',
              fontSize: '2rem',
              fontWeight: '700',
              marginBottom: '0.5rem'
            }}>
              {databaseInfo.total_datasets || 0}
            </div>
            <div style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
              Total Datasets
            </div>
          </div>

          <div style={{
            background: 'rgba(0, 255, 255, 0.05)',
            border: '1px solid rgba(0, 255, 255, 0.2)',
            borderRadius: '8px',
            padding: '1.5rem',
            textAlign: 'center'
          }}>
            <div style={{
              color: 'var(--accent-cyan)',
              fontSize: '2rem',
              fontWeight: '700',
              marginBottom: '0.5rem'
            }}>
              {(databaseInfo.total_records || 0).toLocaleString()}
            </div>
            <div style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
              Total Records
            </div>
          </div>

          <div style={{
            background: 'rgba(0, 255, 255, 0.05)',
            border: '1px solid rgba(0, 255, 255, 0.2)',
            borderRadius: '8px',
            padding: '1.5rem',
            textAlign: 'center'
          }}>
            <div style={{
              color: 'var(--accent-cyan)',
              fontSize: '2rem',
              fontWeight: '700',
              marginBottom: '0.5rem'
            }}>
              {databaseInfo.supports_append ? '✅' : '❌'}
            </div>
            <div style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
              Supports Append
            </div>
          </div>
        </div>
      </Card>

      {/* Saved Datasets */}
      <Card style={{ marginBottom: '2rem' }}>
        <h2 style={{
          color: 'var(--accent-cyan)',
          fontFamily: 'var(--font-display)',
          fontSize: '1.5rem',
          marginBottom: '1.5rem'
        }}>
          📈 Saved Datasets
        </h2>

        {datasets.length === 0 ? (
          <div style={{
            background: 'rgba(255, 165, 0, 0.05)',
            border: '1px solid rgba(255, 165, 0, 0.2)',
            borderRadius: '8px',
            padding: '2rem',
            textAlign: 'center'
          }}>
            <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>📭</div>
            <h3 style={{ color: '#ffa500', margin: '0 0 0.5rem 0' }}>
              No Datasets Found
            </h3>
            <p style={{ color: 'var(--text-secondary)', margin: '0' }}>
              Upload data using the Data Upload page to see datasets here.
            </p>
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{
              width: '100%',
              borderCollapse: 'collapse',
              fontFamily: 'var(--font-primary)',
              fontSize: '0.9rem'
            }}>
              <thead>
                <tr style={{ background: 'rgba(0, 255, 255, 0.1)' }}>
                  <th style={{
                    padding: '1rem',
                    textAlign: 'left',
                    color: 'var(--accent-cyan)',
                    borderBottom: '2px solid var(--border)'
                  }}>
                    Dataset Name
                  </th>
                  <th style={{
                    padding: '1rem',
                    textAlign: 'left',
                    color: 'var(--accent-cyan)',
                    borderBottom: '2px solid var(--border)'
                  }}>
                    Records
                  </th>
                  <th style={{
                    padding: '1rem',
                    textAlign: 'left',
                    color: 'var(--accent-cyan)',
                    borderBottom: '2px solid var(--border)'
                  }}>
                    Date Range
                  </th>
                  <th style={{
                    padding: '1rem',
                    textAlign: 'left',
                    color: 'var(--accent-cyan)',
                    borderBottom: '2px solid var(--border)'
                  }}>
                    Purpose
                  </th>
                  <th style={{
                    padding: '1rem',
                    textAlign: 'center',
                    color: 'var(--accent-cyan)',
                    borderBottom: '2px solid var(--border)'
                  }}>
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody>
                {datasets.map((dataset, index) => (
                  <tr key={index} style={{
                    background: selectedDataset === dataset.name ? 'rgba(0, 255, 255, 0.05)' : 'transparent'
                  }}>
                    <td style={{
                      padding: '1rem',
                      color: 'var(--text-primary)',
                      borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
                      fontWeight: selectedDataset === dataset.name ? '600' : 'normal'
                    }}>
                      {renameMode[dataset.name] ? (
                        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                          <input
                            type="text"
                            value={newDatasetName}
                            onChange={(e) => setNewDatasetName(e.target.value)}
                            placeholder={dataset.name}
                            style={{
                              padding: '0.25rem 0.5rem',
                              background: 'var(--bg-secondary)',
                              border: '1px solid var(--border)',
                              borderRadius: '4px',
                              color: 'var(--text-primary)',
                              fontSize: '0.8rem',
                              width: '120px'
                            }}
                          />
                          <button
                            onClick={() => renameDataset(dataset.name, newDatasetName)}
                            disabled={loading}
                            style={{
                              padding: '0.25rem 0.5rem',
                              background: '#28a745',
                              border: 'none',
                              borderRadius: '4px',
                              color: 'white',
                              fontSize: '0.7rem',
                              cursor: 'pointer'
                            }}
                          >
                            ✅
                          </button>
                          <button
                            onClick={() => {
                              setRenameMode({ ...renameMode, [dataset.name]: false });
                              setNewDatasetName('');
                            }}
                            style={{
                              padding: '0.25rem 0.5rem',
                              background: '#dc3545',
                              border: 'none',
                              borderRadius: '4px',
                              color: 'white',
                              fontSize: '0.7rem',
                              cursor: 'pointer'
                            }}
                          >
                            ❌
                          </button>
                        </div>
                      ) : (
                        dataset.name
                      )}
                    </td>
                    <td style={{
                      padding: '1rem',
                      color: 'var(--text-primary)',
                      borderBottom: '1px solid rgba(255, 255, 255, 0.1)'
                    }}>
                      {(dataset.rows || 0).toLocaleString()}
                    </td>
                    <td style={{
                      padding: '1rem',
                      color: 'var(--text-primary)',
                      borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
                      fontSize: '0.8rem'
                    }}>
                      {dataset.date_range || (dataset.start_date && dataset.end_date 
                        ? `${dataset.start_date} to ${dataset.end_date}`
                        : 'Unknown'
                      )}
                    </td>
                    <td style={{
                      padding: '1rem',
                      color: 'var(--text-secondary)',
                      borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
                      fontSize: '0.8rem'
                    }}>
                      {dataset.purpose || 'training'}
                    </td>
                    <td style={{
                      padding: '1rem',
                      borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
                      textAlign: 'center'
                    }}>
                      <div style={{ display: 'flex', gap: '0.25rem', justifyContent: 'center', flexWrap: 'wrap' }}>
                        <button
                          onClick={() => loadDatasetToSession(dataset.name)}
                          disabled={loading}
                          style={{
                            padding: '0.25rem 0.5rem',
                            background: '#007bff',
                            border: 'none',
                            borderRadius: '4px',
                            color: 'white',
                            fontFamily: 'var(--font-primary)',
                            fontSize: '0.7rem',
                            cursor: loading ? 'not-allowed' : 'pointer'
                          }}
                        >
                          📥 Load
                        </button>
                        <button
                          onClick={() => previewDataset(dataset.name)}
                          disabled={loading}
                          style={{
                            padding: '0.25rem 0.5rem',
                            background: 'var(--gradient-primary)',
                            border: 'none',
                            borderRadius: '4px',
                            color: 'white',
                            fontFamily: 'var(--font-primary)',
                            fontSize: '0.7rem',
                            cursor: loading ? 'not-allowed' : 'pointer'
                          }}
                        >
                          👁️ View
                        </button>
                        <button
                          onClick={() => exportDataset(dataset.name)}
                          disabled={loading}
                          style={{
                            padding: '0.25rem 0.5rem',
                            background: '#28a745',
                            border: 'none',
                            borderRadius: '4px',
                            color: 'white',
                            fontFamily: 'var(--font-primary)',
                            fontSize: '0.7rem',
                            cursor: loading ? 'not-allowed' : 'pointer'
                          }}
                        >
                          📤 Export
                        </button>
                        <button
                          onClick={() => {
                            setRenameMode({ ...renameMode, [dataset.name]: true });
                            setNewDatasetName(dataset.name);
                          }}
                          disabled={loading || renameMode[dataset.name]}
                          style={{
                            padding: '0.25rem 0.5rem',
                            background: '#ffc107',
                            border: 'none',
                            borderRadius: '4px',
                            color: 'black',
                            fontFamily: 'var(--font-primary)',
                            fontSize: '0.7rem',
                            cursor: loading ? 'not-allowed' : 'pointer'
                          }}
                        >
                          ✏️ Rename
                        </button>
                        <button
                          onClick={() => deleteDataset(dataset.name)}
                          disabled={loading}
                          style={{
                            padding: '0.25rem 0.5rem',
                            background: '#dc3545',
                            border: 'none',
                            borderRadius: '4px',
                            color: 'white',
                            fontFamily: 'var(--font-primary)',
                            fontSize: '0.7rem',
                            cursor: loading ? 'not-allowed' : 'pointer'
                          }}
                        >
                          🗑️ Delete
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {/* Model Results Management */}
      {modelResults.length > 0 && (
        <Card style={{ marginBottom: '2rem' }}>
          <h2 style={{
            color: 'var(--accent-cyan)',
            fontFamily: 'var(--font-display)',
            fontSize: '1.5rem',
            marginBottom: '1.5rem'
          }}>
            🤖 Model Results
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {modelResults.map((modelName, index) => (
              <div key={index} style={{
                background: 'rgba(0, 255, 255, 0.05)',
                border: '1px solid rgba(0, 255, 255, 0.2)',
                borderRadius: '8px',
                padding: '1rem'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                  <h3 style={{ color: 'var(--accent-gold)', margin: '0' }}>
                    🎯 {modelName} Model
                  </h3>
                  <button
                    onClick={() => deleteModelResults(modelName)}
                    disabled={loading}
                    style={{
                      padding: '0.25rem 0.5rem',
                      background: '#dc3545',
                      border: 'none',
                      borderRadius: '4px',
                      color: 'white',
                      fontSize: '0.8rem',
                      cursor: loading ? 'not-allowed' : 'pointer'
                    }}
                  >
                    🗑️ Delete
                  </button>
                </div>
                <p style={{ color: 'var(--text-secondary)', margin: '0', fontSize: '0.9rem' }}>
                  Model training results stored in database
                </p>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Predictions Management */}
      {predictions.length > 0 && (
        <Card style={{ marginBottom: '2rem' }}>
          <h2 style={{
            color: 'var(--accent-cyan)',
            fontFamily: 'var(--font-display)',
            fontSize: '1.5rem',
            marginBottom: '1.5rem'
          }}>
            🔮 Saved Predictions
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {predictions.map((modelName, index) => (
              <div key={index} style={{
                background: 'rgba(255, 215, 0, 0.05)',
                border: '1px solid rgba(255, 215, 0, 0.2)',
                borderRadius: '8px',
                padding: '1rem'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                  <h3 style={{ color: 'var(--accent-gold)', margin: '0' }}>
                    📈 {modelName} Predictions
                  </h3>
                  <div style={{ display: 'flex', gap: '0.5rem' }}>
                    <button
                      onClick={() => previewDataset(`predictions_${modelName}`)}
                      disabled={loading}
                      style={{
                        padding: '0.25rem 0.5rem',
                        background: 'var(--gradient-primary)',
                        border: 'none',
                        borderRadius: '4px',
                        color: 'white',
                        fontSize: '0.8rem',
                        cursor: loading ? 'not-allowed' : 'pointer'
                      }}
                    >
                      👁️ View
                    </button>
                    <button
                      onClick={() => deletePredictions(modelName)}
                      disabled={loading}
                      style={{
                        padding: '0.25rem 0.5rem',
                        background: '#dc3545',
                        border: 'none',
                        borderRadius: '4px',
                        color: 'white',
                        fontSize: '0.8rem',
                        cursor: loading ? 'not-allowed' : 'pointer'
                      }}
                    >
                      🗑️ Delete
                    </button>
                  </div>
                </div>
                <p style={{ color: 'var(--text-secondary)', margin: '0', fontSize: '0.9rem' }}>
                  Prediction results for {modelName} model
                </p>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Database Maintenance */}
      <Card style={{ marginBottom: '2rem' }}>
        <h2 style={{
          color: 'var(--accent-cyan)',
          fontFamily: 'var(--font-display)',
          fontSize: '1.5rem',
          marginBottom: '1.5rem'
        }}>
          🔧 Database Maintenance
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Export Data */}
          <div>
            <h3 style={{ color: 'var(--accent-gold)', margin: '0 0 1rem 0' }}>
              Export Data
            </h3>

            <div style={{ marginBottom: '1rem' }}>
              <button
                onClick={() => {
                  const sessionData = sessionStorage.getItem('currentDataset');
                  if (sessionData) {
                    const blob = new Blob([sessionData], { type: 'application/json' });
                    const url = window.URL.createObjectURL(blob);
                    const link = document.createElement('a');
                    link.href = url;
                    link.download = `session_data_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`;
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                    window.URL.revokeObjectURL(url);
                    setStatus('✅ Session data exported');
                  } else {
                    setStatus('❌ No session data to export');
                  }
                }}
                disabled={loading}
                style={{
                  padding: '0.75rem 1rem',
                  background: '#28a745',
                  border: 'none',
                  borderRadius: '6px',
                  color: 'white',
                  fontFamily: 'var(--font-primary)',
                  fontWeight: '600',
                  cursor: loading ? 'not-allowed' : 'pointer',
                  fontSize: '0.9rem',
                  width: '100%',
                  marginBottom: '0.5rem'
                }}
              >
                📥 Download Current Session Data
              </button>
            </div>

            {datasets.length > 0 && (
              <div>
                <p style={{ color: 'var(--text-secondary)', margin: '0 0 0.5rem 0', fontSize: '0.9rem' }}>
                  Export All Datasets:
                </p>
                <button
                  onClick={async () => {
                    for (const dataset of datasets) {
                      await exportDataset(dataset.name);
                    }
                  }}
                  disabled={loading}
                  style={{
                    padding: '0.75rem 1rem',
                    background: '#007bff',
                    border: 'none',
                    borderRadius: '6px',
                    color: 'white',
                    fontFamily: 'var(--font-primary)',
                    fontWeight: '600',
                    cursor: loading ? 'not-allowed' : 'pointer',
                    fontSize: '0.9rem',
                    width: '100%'
                  }}
                >
                  📤 Export All Datasets
                </button>
              </div>
            )}
          </div>

          {/* Danger Zone */}
          <div>
            <h3 style={{ color: '#ff6b6b', margin: '0 0 1rem 0' }}>
              ⚠️ Danger Zone
            </h3>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <button
                onClick={clearSessionData}
                disabled={loading}
                style={{
                  padding: '0.75rem 1rem',
                  background: '#ffc107',
                  border: 'none',
                  borderRadius: '6px',
                  color: 'black',
                  fontFamily: 'var(--font-primary)',
                  fontWeight: '600',
                  cursor: loading ? 'not-allowed' : 'pointer',
                  fontSize: '0.9rem'
                }}
              >
                🧹 Clear Session Data
              </button>

              <button
                onClick={syncMetadata}
                disabled={loading}
                style={{
                  padding: '0.75rem 1rem',
                  background: '#17a2b8',
                  border: 'none',
                  borderRadius: '6px',
                  color: 'white',
                  fontFamily: 'var(--font-primary)',
                  fontWeight: '600',
                  cursor: loading ? 'not-allowed' : 'pointer',
                  fontSize: '0.9rem'
                }}
                title="Fix metadata inconsistencies"
              >
                🔄 Sync Metadata
              </button>

              <button
                onClick={cleanDatabase}
                disabled={loading}
                style={{
                  padding: '0.75rem 1rem',
                  background: '#6f42c1',
                  border: 'none',
                  borderRadius: '6px',
                  color: 'white',
                  fontFamily: 'var(--font-primary)',
                  fontWeight: '600',
                  cursor: loading ? 'not-allowed' : 'pointer',
                  fontSize: '0.9rem'
                }}
                title="Remove inconsistent data and keep only main dataset"
              >
                🧹 Clean Database
              </button>

              {!showConfirmClearAll ? (
                <button
                  onClick={() => setShowConfirmClearAll(true)}
                  disabled={loading}
                  style={{
                    padding: '0.75rem 1rem',
                    background: '#dc3545',
                    border: 'none',
                    borderRadius: '6px',
                    color: 'white',
                    fontFamily: 'var(--font-primary)',
                    fontWeight: '600',
                    cursor: loading ? 'not-allowed' : 'pointer',
                    fontSize: '0.9rem'
                  }}
                >
                  🗑️ Clear All Data
                </button>
              ) : (
                <div style={{
                  background: 'rgba(255, 0, 0, 0.1)',
                  border: '1px solid rgba(255, 0, 0, 0.3)',
                  borderRadius: '6px',
                  padding: '1rem'
                }}>
                  <p style={{ color: '#ff6b6b', margin: '0 0 1rem 0', fontSize: '0.9rem', fontWeight: '600' }}>
                    ⚠️ This will permanently delete ALL data from the database!
                  </p>
                  <div style={{ display: 'flex', gap: '0.5rem' }}>
                    <button
                      onClick={clearAllData}
                      disabled={loading}
                      style={{
                        padding: '0.5rem 1rem',
                        background: '#dc3545',
                        border: 'none',
                        borderRadius: '4px',
                        color: 'white',
                        fontFamily: 'var(--font-primary)',
                        fontWeight: '600',
                        cursor: loading ? 'not-allowed' : 'pointer',
                        fontSize: '0.8rem',
                        flex: 1
                      }}
                    >
                      ✅ Yes, Delete Everything
                    </button>
                    <button
                      onClick={() => setShowConfirmClearAll(false)}
                      style={{
                        padding: '0.5rem 1rem',
                        background: '#6c757d',
                        border: 'none',
                        borderRadius: '4px',
                        color: 'white',
                        fontFamily: 'var(--font-primary)',
                        fontSize: '0.8rem',
                        flex: 1,
                        cursor: 'pointer'
                      }}
                    >
                      ❌ Cancel
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </Card>

      {/* Data Recovery Section */}
      <Card style={{ marginBottom: '2rem' }}>
        <h2 style={{
          color: 'var(--accent-cyan)',
          fontFamily: 'var(--font-display)',
          fontSize: '1.5rem',
          marginBottom: '1.5rem'
        }}>
          🔄 Data Recovery
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h3 style={{ color: 'var(--accent-gold)', margin: '0 0 1rem 0' }}>
              Check Session Data
            </h3>
            {sessionStorage.getItem('currentDataset') ? (
              <div style={{
                background: 'rgba(0, 255, 0, 0.05)',
                border: '1px solid rgba(0, 255, 0, 0.2)',
                borderRadius: '8px',
                padding: '1rem'
              }}>
                <p style={{ color: '#51cf66', margin: '0 0 1rem 0', fontWeight: '600' }}>
                  ✅ Session data exists
                </p>
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  <button
                    onClick={() => {
                      const sessionData = JSON.parse(sessionStorage.getItem('currentDataset') || '[]');
                      // Here you would typically save to database via API
                      setStatus('💾 Session data would be saved as "recovered_data"');
                    }}
                    style={{
                      padding: '0.5rem 1rem',
                      background: '#28a745',
                      border: 'none',
                      borderRadius: '4px',
                      color: 'white',
                      fontSize: '0.8rem',
                      cursor: 'pointer'
                    }}
                  >
                    💾 Save to Database
                  </button>
                  <button
                    onClick={clearSessionData}
                    style={{
                      padding: '0.5rem 1rem',
                      background: '#dc3545',
                      border: 'none',
                      borderRadius: '4px',
                      color: 'white',
                      fontSize: '0.8rem',
                      cursor: 'pointer'
                    }}
                  >
                    🗑️ Clear Session
                  </button>
                </div>
              </div>
            ) : (
              <div style={{
                background: 'rgba(255, 165, 0, 0.05)',
                border: '1px solid rgba(255, 165, 0, 0.2)',
                borderRadius: '8px',
                padding: '1rem'
              }}>
                <p style={{ color: '#ffa500', margin: '0' }}>
                  ⚠️ No data in current session
                </p>
              </div>
            )}
          </div>

          <div>
            <h3 style={{ color: 'var(--accent-gold)', margin: '0 0 1rem 0' }}>
              Auto-save Settings
            </h3>
            <div style={{
              background: 'rgba(0, 255, 255, 0.05)',
              border: '1px solid rgba(0, 255, 255, 0.2)',
              borderRadius: '8px',
              padding: '1rem'
            }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  defaultChecked={true}
                  style={{ marginRight: '0.5rem' }}
                />
                <span style={{ color: 'var(--text-primary)', fontSize: '0.9rem' }}>
                  Auto-save uploaded data
                </span>
              </label>
              <p style={{ color: 'var(--text-secondary)', margin: '0.5rem 0 0 0', fontSize: '0.8rem' }}>
                New uploads will be automatically saved to database
              </p>
            </div>
          </div>
        </div>
      </Card>

      {/* Raw Database View (Debug) */}
      <Card>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
          <h2 style={{
            color: 'var(--accent-cyan)',
            fontFamily: 'var(--font-display)',
            fontSize: '1.5rem',
            margin: '0'
          }}>
            🔍 Raw Database View (Debug)
          </h2>
          <button
            onClick={() => setShowRawDebugView(!showRawDebugView)}
            style={{
              padding: '0.5rem 1rem',
              background: showRawDebugView ? '#dc3545' : 'var(--gradient-primary)',
              border: 'none',
              borderRadius: '6px',
              color: 'white',
              fontFamily: 'var(--font-primary)',
              fontWeight: '600',
              cursor: 'pointer',
              fontSize: '0.9rem'
            }}
          >
            {showRawDebugView ? '❌ Hide Debug' : '🔍 Show Debug'}
          </button>
        </div>

        {showRawDebugView && (
          <div>
            <div style={{ marginBottom: '1.5rem' }}>
              <p style={{ color: 'var(--text-primary)', margin: '0 0 0.5rem 0', fontWeight: '600' }}>
                All Keys: {rawDatabaseKeys.length}
              </p>
              <p style={{ color: 'var(--text-secondary)', margin: '0 0 1rem 0', fontSize: '0.9rem' }}>
                Database Type: {databaseInfo.database_type || 'Unknown'}
              </p>

              {rawDatabaseKeys.length > 0 && (
                <div style={{ marginBottom: '1rem' }}>
                  <select
                    value={selectedKey}
                    onChange={(e) => setSelectedKey(e.target.value)}
                    style={{
                      padding: '0.5rem',
                      background: 'var(--bg-secondary)',
                      border: '1px solid var(--border)',
                      borderRadius: '4px',
                      color: 'var(--text-primary)',
                      fontSize: '0.9rem',
                      marginRight: '0.5rem',
                      minWidth: '200px'
                    }}
                  >
                    <option value="">Select key to inspect...</option>
                    {rawDatabaseKeys.map((key, index) => (
                      <option key={index} value={key}>{key}</option>
                    ))}
                  </select>

                  <button
                    onClick={() => viewKeyContent(selectedKey)}
                    disabled={!selectedKey || loading}
                    style={{
                      padding: '0.5rem 1rem',
                      background: !selectedKey || loading ? 'var(--bg-secondary)' : 'var(--gradient-primary)',
                      border: 'none',
                      borderRadius: '4px',
                      color: 'white',
                      fontFamily: 'var(--font-primary)',
                      cursor: !selectedKey || loading ? 'not-allowed' : 'pointer',
                      fontSize: '0.9rem'
                    }}
                  >
                    View Key Content
                  </button>
                </div>
              )}
            </div>

            {keyContent && (
              <div style={{
                background: 'var(--bg-secondary)',
                border: '1px solid var(--border)',
                borderRadius: '8px',
                padding: '1rem',
                maxHeight: '400px',
                overflowY: 'auto'
              }}>
                <h4 style={{ color: 'var(--accent-gold)', margin: '0 0 1rem 0' }}>
                  Content for key: {selectedKey}
                </h4>
                <pre style={{
                  color: 'var(--text-primary)',
                  fontSize: '0.8rem',
                  fontFamily: 'var(--font-mono)',
                  margin: '0',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word'
                }}>
                  {typeof keyContent === 'object' 
                    ? JSON.stringify(keyContent, null, 2)
                    : keyContent
                  }
                </pre>
              </div>
            )}

            {rawDatabaseKeys.length === 0 && (
              <div style={{
                background: 'rgba(255, 165, 0, 0.05)',
                border: '1px solid rgba(255, 165, 0, 0.2)',
                borderRadius: '8px',
                padding: '1rem',
                textAlign: 'center'
              }}>
                <p style={{ color: '#ffa500', margin: '0' }}>
                  No database keys found or database connection unavailable
                </p>
              </div>
            )}
          </div>
        )}
      </Card>

      {/* Status Message */}
      {status && (
        <div style={{
          position: 'fixed',
          bottom: '2rem',
          right: '2rem',
          padding: '1rem 1.5rem',
          background: status.includes('❌') 
            ? 'rgba(255, 0, 0, 0.9)' 
            : status.includes('✅')
            ? 'rgba(0, 255, 0, 0.9)'
            : 'rgba(0, 255, 255, 0.9)',
          border: `1px solid ${
            status.includes('❌') ? '#ff0000' : 
            status.includes('✅') ? '#00ff00' : '#00ffff'
          }`,
          borderRadius: '8px',
          color: 'white',
          fontFamily: 'var(--font-primary)',
          fontWeight: '600',
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
          zIndex: 1000,
          maxWidth: '400px'
        }}>
          {status}
        </div>
      )}

      {/* Data Preview */}
      {dataPreview && (
        <Card style={{ marginTop: '2rem' }}>
          <h2 style={{
            color: 'var(--accent-gold)',
            fontFamily: 'var(--font-display)',
            fontSize: '1.5rem',
            marginBottom: '1.5rem'
          }}>
            🔍 Data Preview: {selectedDataset}
          </h2>

          <div style={{
            background: 'rgba(0, 255, 255, 0.05)',
            border: '1px solid rgba(0, 255, 255, 0.2)',
            borderRadius: '8px',
            padding: '1rem',
            marginBottom: '1.5rem'
          }}>
            <p style={{ color: 'var(--accent-cyan)', margin: '0' }}>
              📊 Showing first {Math.min(dataPreview.length, 100)} records out of {dataPreview.length} total
            </p>
          </div>

          <div style={{ overflowX: 'auto', maxHeight: '400px', overflowY: 'auto' }}>
            <table style={{
              width: '100%',
              borderCollapse: 'collapse',
              fontFamily: 'var(--font-mono)',
              fontSize: '0.8rem'
            }}>
              <thead style={{ position: 'sticky', top: 0, background: 'var(--bg-primary)' }}>
                <tr style={{ background: 'rgba(255, 215, 0, 0.1)' }}>
                  <th style={{
                    padding: '0.75rem',
                    textAlign: 'left',
                    color: 'var(--accent-gold)',
                    borderBottom: '2px solid var(--border)'
                  }}>
                    Timestamp
                  </th>
                  <th style={{
                    padding: '0.75rem',
                    textAlign: 'right',
                    color: 'var(--accent-gold)',
                    borderBottom: '2px solid var(--border)'
                  }}>
                    Open
                  </th>
                  <th style={{
                    padding: '0.75rem',
                    textAlign: 'right',
                    color: 'var(--accent-gold)',
                    borderBottom: '2px solid var(--border)'
                  }}>
                    High
                  </th>
                  <th style={{
                    padding: '0.75rem',
                    textAlign: 'right',
                    color: 'var(--accent-gold)',
                    borderBottom: '2px solid var(--border)'
                  }}>
                    Low
                  </th>
                  <th style={{
                    padding: '0.75rem',
                    textAlign: 'right',
                    color: 'var(--accent-gold)',
                    borderBottom: '2px solid var(--border)'
                  }}>
                    Close
                  </th>
                  <th style={{
                    padding: '0.75rem',
                    textAlign: 'right',
                    color: 'var(--accent-gold)',
                    borderBottom: '2px solid var(--border)'
                  }}>
                    Volume
                  </th>
                </tr>
              </thead>
              <tbody>
                {dataPreview.slice(0, 100).map((row, index) => (
                  <tr key={index}>
                    <td style={{
                      padding: '0.5rem 0.75rem',
                      color: 'var(--text-primary)',
                      borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
                      whiteSpace: 'nowrap'
                    }}>
                      {row.timestamp || row.DateTime || index}
                    </td>
                    <td style={{
                      padding: '0.5rem 0.75rem',
                      color: 'var(--text-primary)',
                      borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
                      textAlign: 'right'
                    }}>
                      {row.Open?.toFixed(2) || 'N/A'}
                    </td>
                    <td style={{
                      padding: '0.5rem 0.75rem',
                      color: '#51cf66',
                      borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
                      textAlign: 'right'
                    }}>
                      {row.High?.toFixed(2) || 'N/A'}
                    </td>
                    <td style={{
                      padding: '0.5rem 0.75rem',
                      color: '#ff6b6b',
                      borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
                      textAlign: 'right'
                    }}>
                      {row.Low?.toFixed(2) || 'N/A'}
                    </td>
                    <td style={{
                      padding: '0.5rem 0.75rem',
                      color: 'var(--accent-cyan)',
                      borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
                      textAlign: 'right',
                      fontWeight: '600'
                    }}>
                      {row.Close?.toFixed(2) || 'N/A'}
                    </td>
                    <td style={{
                      padding: '0.5rem 0.75rem',
                      color: 'var(--text-secondary)',
                      borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
                      textAlign: 'right'
                    }}>
                      {row.Volume?.toLocaleString() || 'N/A'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
};

export default DatabaseManager;