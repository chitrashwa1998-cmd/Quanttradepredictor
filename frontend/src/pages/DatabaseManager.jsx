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

  // Load database information
  const loadDatabaseInfo = useCallback(async () => {
    try {
      setLoading(true);
      
      const [dbInfoResponse, datasetsResponse] = await Promise.all([
        dataAPI.getDatabaseInfo(),
        dataAPI.getDatasets()
      ]);

      setDatabaseInfo(dbInfoResponse.data || {});
      setDatasets(datasetsResponse.data || []);
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
    if (!confirm('⚠️ Are you sure you want to clear ALL data? This action cannot be undone!')) {
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
      link.download = `${datasetName}.csv`;
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
              Data Management
            </h3>
            <div style={{ display: 'flex', gap: '1rem' }}>
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
              <button
                onClick={clearAllData}
                disabled={loading}
                style={{
                  padding: '0.75rem 1rem',
                  background: loading ? 'var(--bg-secondary)' : '#dc3545',
                  border: '1px solid #dc3545',
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
                    Last Updated
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
                      {dataset.name}
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
                      {dataset.start_date && dataset.end_date 
                        ? `${dataset.start_date} to ${dataset.end_date}`
                        : 'Unknown'
                      }
                    </td>
                    <td style={{
                      padding: '1rem',
                      color: 'var(--text-secondary)',
                      borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
                      fontSize: '0.8rem'
                    }}>
                      {dataset.updated_at || 'Unknown'}
                    </td>
                    <td style={{
                      padding: '1rem',
                      borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
                      textAlign: 'center'
                    }}>
                      <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'center' }}>
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
                            fontSize: '0.8rem',
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
                            fontSize: '0.8rem',
                            cursor: loading ? 'not-allowed' : 'pointer'
                          }}
                        >
                          📤 Export
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
                            fontSize: '0.8rem',
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

      {/* Data Preview */}
      {dataPreview && (
        <Card>
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
    </div>
  );
};

export default DatabaseManager;