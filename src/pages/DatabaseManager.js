import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { API_BASE_URL } from '../config/api';

const DatabaseManager = () => {
  const [dbInfo, setDbInfo] = useState(null);
  const [datasets, setDatasets] = useState([]);
  // Removed unused state variables to fix ESLint warnings
  const [loading, setLoading] = useState(true);
  const [confirmDelete, setConfirmDelete] = useState({});
  const [selectedDatasets, setSelectedDatasets] = useState([]);

  useEffect(() => {
    fetchDatabaseInfo();
  }, []);

  const fetchDatabaseInfo = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_BASE_URL}/database-info`);
      const dbData = response.data.success ? response.data.data : null;
      setDbInfo(dbData);

      // Set datasets from database info
      if (dbData && dbData.datasets) {
        setDatasets(dbData.datasets);
      }
    } catch (error) {
      console.error('Failed to fetch database info:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleLoadDataset = async (datasetName) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/data/load`, { dataset_name: datasetName });
      if (response.data.success) {
        alert(`✅ Loaded dataset: ${datasetName}`);
      } else {
        alert('Failed to load dataset');
      }
    } catch (error) {
      console.error('Failed to load dataset:', error);
      alert('Failed to load dataset');
    }
  };

  const handleDeleteDataset = async (datasetName) => {
    try {
      const response = await axios.delete(`${API_BASE_URL}/database/dataset/${datasetName}`);
      if (response.data.success) {
        alert(`✅ Deleted dataset: ${datasetName}`);
        fetchDatabaseInfo(); // Refresh the list
        setConfirmDelete({});
      } else {
        alert('Failed to delete dataset');
      }
    } catch (error) {
      console.error('Failed to delete dataset:', error);
      alert('Failed to delete dataset');
    }
  };

  const handleExportDataset = async (datasetName) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/database/export/${datasetName}`, {
        responseType: 'blob'
      });

      const blob = new Blob([response.data], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${datasetName}_${new Date().toISOString().slice(0, 10)}.csv`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Failed to export dataset:', error);
      alert('Failed to export dataset');
    }
  };

  const handleClearAllData = async () => {
    try {
      const response = await axios.delete(`${API_BASE_URL}/database/clear-all`);
      if (response.data.success) {
        alert('✅ All database data cleared');
        fetchDatabaseInfo();
      } else {
        alert('Failed to clear database');
      }
    } catch (error) {
      console.error('Failed to clear database:', error);
      alert('Failed to clear database');
    }
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleString();
  };

  if (loading) {
    return (
      <div className="container">
        <div className="loading">
          <div className="spinner"></div>
          <p>Loading database information...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="container">
      <div className="header">
        <h1>💾 DATA CONTROL CENTER</h1>
        <p>Database Management & Storage</p>
      </div>

      {/* Database Overview */}
      <div className="card">
        <h3>📊 Database Overview</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginTop: '1rem' }}>
          <div className="metric-card">
            <div className="metric-value">{dbInfo?.total_datasets || 0}</div>
            <div className="metric-label">Total Datasets</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">{dbInfo?.total_models || 0}</div>
            <div className="metric-label">Model Results</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">{dbInfo?.total_trained_models || 0}</div>
            <div className="metric-label">Trained Models</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">{dbInfo?.database_type || 'Unknown'}</div>
            <div className="metric-label">Database Type</div>
          </div>
        </div>
        <button 
          onClick={fetchDatabaseInfo}
          className="btn btn-secondary"
          style={{ marginTop: '1rem' }}
        >
          🔄 Refresh
        </button>
      </div>

      {/* Datasets Management */}
      <div className="card">
        <h3>📈 Saved Datasets</h3>
        {datasets.length > 0 ? (
          <div>
            <p style={{ color: '#00ff41', marginBottom: '1rem' }}>
              Found {datasets.length} dataset(s) in database
            </p>
            {datasets.map((dataset, index) => (
              <div key={index} className="dataset-item" style={{ 
                background: 'rgba(0, 255, 255, 0.05)', 
                border: '1px solid rgba(0, 255, 255, 0.2)', 
                borderRadius: '8px', 
                padding: '1rem', 
                marginBottom: '1rem' 
              }}>
                <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr 1fr', gap: '1rem', alignItems: 'start' }}>
                  <div>
                    <h4 style={{ color: '#00ffff', margin: '0 0 0.5rem 0' }}>
                      📊 {dataset.name} ({dataset.rows} rows)
                    </h4>
                    <p style={{ margin: '0.25rem 0', color: '#b8bcc8' }}>
                      <strong>Columns:</strong> {dataset.columns?.join(', ') || 'N/A'}
                    </p>
                    <p style={{ margin: '0.25rem 0', color: '#b8bcc8' }}>
                      <strong>Date Range:</strong> {dataset.start_date && dataset.end_date ? 
                        `${dataset.start_date} to ${dataset.end_date}` : 'Not available'}
                    </p>
                    <p style={{ margin: '0.25rem 0', color: '#b8bcc8' }}>
                      <strong>Updated:</strong> {formatDate(dataset.updated_at)}
                    </p>
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                    <button 
                      onClick={() => handleLoadDataset(dataset.name)}
                      className="btn btn-primary"
                      style={{ fontSize: '0.9rem' }}
                    >
                      Load Dataset
                    </button>
                    <button 
                      onClick={() => handleExportDataset(dataset.name)}
                      className="btn btn-secondary"
                      style={{ fontSize: '0.9rem' }}
                    >
                      📥 Export CSV
                    </button>
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                    {!confirmDelete[dataset.name] ? (
                      <button 
                        onClick={() => setConfirmDelete({...confirmDelete, [dataset.name]: true})}
                        className="btn btn-danger"
                        style={{ fontSize: '0.9rem' }}
                      >
                        🗑️ Delete
                      </button>
                    ) : (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                        <p style={{ color: '#ff6b6b', fontSize: '0.8rem', margin: 0 }}>
                          ⚠️ Delete '{dataset.name}'?
                        </p>
                        <div style={{ display: 'flex', gap: '0.25rem' }}>
                          <button 
                            onClick={() => handleDeleteDataset(dataset.name)}
                            className="btn btn-danger"
                            style={{ fontSize: '0.8rem', padding: '0.25rem 0.5rem' }}
                          >
                            ✅ Yes
                          </button>
                          <button 
                            onClick={() => setConfirmDelete({...confirmDelete, [dataset.name]: false})}
                            className="btn btn-secondary"
                            style={{ fontSize: '0.8rem', padding: '0.25rem 0.5rem' }}
                          >
                            ❌ No
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p style={{ color: '#b8bcc8' }}>No datasets found in database. Upload data first!</p>
        )}
      </div>

      {/* Model Results */}
      <div className="card">
        <h3>🤖 Model Results</h3>
        
          <p style={{ color: '#b8bcc8' }}>No model results found. Train models first!</p>
        
      </div>

      {/* Predictions */}
      <div className="card">
        <h3>🔮 Saved Predictions</h3>
       
          <p style={{ color: '#b8bcc8' }}>No predictions found. Generate predictions first!</p>
        
      </div>

      {/* Database Maintenance */}
      <div className="card">
        <h3>🔧 Database Maintenance</h3>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
          <div>
            <h4>Export Data</h4>
            <p style={{ color: '#b8bcc8', lineHeight: 1.6 }}>
              Export datasets from the database as CSV files for backup or analysis.
            </p>
            {datasets.length > 0 && (
              <div style={{ marginTop: '1rem' }}>
                <label style={{ display: 'block', marginBottom: '0.5rem', color: '#00ffff' }}>
                  Select datasets to export:
                </label>
                <div style={{ marginBottom: '1rem' }}>
                  {datasets.map((dataset) => (
                    <label key={dataset.name} style={{ display: 'block', marginBottom: '0.25rem', color: '#b8bcc8' }}>
                      <input
                        type="checkbox"
                        checked={selectedDatasets.includes(dataset.name)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setSelectedDatasets([...selectedDatasets, dataset.name]);
                          } else {
                            setSelectedDatasets(selectedDatasets.filter(name => name !== dataset.name));
                          }
                        }}
                        style={{ marginRight: '0.5rem' }}
                      />
                      {dataset.name}
                    </label>
                  ))}
                </div>
                {selectedDatasets.length > 0 && (
                  <button 
                    onClick={() => {
                      selectedDatasets.forEach(name => handleExportDataset(name));
                    }}
                    className="btn btn-primary"
                  >
                    📥 Export Selected Datasets
                  </button>
                )}
              </div>
            )}
          </div>

          <div>
            <h4 style={{ color: '#ff6b6b' }}>⚠️ Danger Zone</h4>
            <p style={{ color: '#b8bcc8', lineHeight: 1.6 }}>
              This will permanently delete all data from the database including datasets, models, and predictions.
            </p>
            <div style={{ marginTop: '1rem' }}>
              <label style={{ display: 'block', marginBottom: '1rem', color: '#ff6b6b' }}>
                <input
                  type="checkbox"
                  onChange={(e) => setConfirmDelete({...confirmDelete, clearAll: e.target.checked})}
                  style={{ marginRight: '0.5rem' }}
                />
                ⚠️ I confirm I want to delete ALL data from the database
              </label>
              <button 
                onClick={handleClearAllData}
                disabled={!confirmDelete.clearAll}
                className="btn btn-danger"
                style={{ opacity: confirmDelete.clearAll ? 1 : 0.5 }}
              >
                🗑️ Clear All Database
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DatabaseManager;