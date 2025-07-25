/**
 * Data Upload page - Complete Streamlit functionality migration
 */

import { useState, useEffect, useCallback } from 'react';
import Card from '../components/common/Card';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { dataAPI } from '../services/api';

const DataUpload = () => {
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [selectedDataset, setSelectedDataset] = useState('');
  const [datasetPurpose, setDatasetPurpose] = useState('training');
  const [uploadStatus, setUploadStatus] = useState('');
  const [previewData, setPreviewData] = useState(null);
  const [processingResults, setProcessingResults] = useState(null);

  // Load datasets on component mount
  const loadDatasets = useCallback(async () => {
    try {
      setLoading(true);
      const response = await dataAPI.getDatasets();
      setDatasets(response?.data || response || []);
    } catch (error) {
      console.error('Error loading datasets:', error);
      setDatasets([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadDatasets();
  }, [loadDatasets]);

  // Handle file selection
  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file && file.type === 'text/csv') {
      setUploadedFile(file);
      setUploadStatus('');
      setPreviewData(null);
      setProcessingResults(null);
    } else {
      setUploadStatus('Please select a valid CSV file');
      setUploadedFile(null);
    }
  };

  // Handle file upload and processing
  const handleUpload = async () => {
    if (!uploadedFile) {
      setUploadStatus('Please select a file first');
      return;
    }

    if (!selectedDataset.trim()) {
      setUploadStatus('Please enter a dataset name');
      return;
    }

    try {
      setLoading(true);
      setUploadProgress(0);
      setUploadStatus('Uploading and processing data...');

      const formData = new FormData();
      formData.append('file', uploadedFile);
      formData.append('dataset_name', selectedDataset);
      formData.append('purpose', datasetPurpose);

      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => Math.min(prev + 10, 90));
      }, 200);

      const response = await dataAPI.uploadData(formData);

      clearInterval(progressInterval);
      setUploadProgress(100);

      if (response.data) {
        setUploadStatus('✅ Data uploaded and processed successfully!');
        setProcessingResults(response.data);
        
        // Load preview data
        if (response.data.preview) {
          setPreviewData(response.data.preview);
        }

        // Refresh datasets list
        await loadDatasets();
        
        // Clear form
        setUploadedFile(null);
        setSelectedDataset('');
        document.getElementById('file-input').value = '';
      }
    } catch (error) {
      setUploadStatus(`❌ Upload failed: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
      setUploadProgress(0);
    }
  };

  // Delete dataset
  const handleDeleteDataset = async (datasetName) => {
    if (!window.confirm(`Are you sure you want to delete "${datasetName}"?`)) {
      return;
    }

    try {
      setLoading(true);
      await dataAPI.deleteDataset(datasetName);
      setUploadStatus(`✅ Dataset "${datasetName}" deleted successfully`);
      await loadDatasets();
    } catch (error) {
      setUploadStatus(`❌ Failed to delete dataset: ${error.response?.data?.detail || error.message}`);
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
          📊 DATA UPLOAD CENTER
        </h1>
        <p style={{
          fontSize: '1.2rem',
          margin: '1rem 0 0 0',
          color: 'rgba(255,255,255,0.8)',
          fontFamily: 'var(--font-primary)'
        }}>
          Load and Process Market Data
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Upload Section */}
        <div className="lg:col-span-2">
          <Card>
            <h2 style={{
              color: 'var(--accent-cyan)',
              fontFamily: 'var(--font-display)',
              fontSize: '1.5rem',
              marginBottom: '1.5rem'
            }}>
              Upload OHLC Data
            </h2>
            
            <div style={{ marginBottom: '1.5rem' }}>
              <p style={{ color: 'var(--text-primary)', marginBottom: '1rem' }}>
                Upload your historical price data in CSV format. The file should contain columns for Date, Open, High, Low, Close, and optionally Volume.
              </p>
              
              <div style={{
                background: 'rgba(0, 255, 255, 0.05)',
                border: '1px solid rgba(0, 255, 255, 0.2)',
                borderRadius: '8px',
                padding: '1rem',
                marginBottom: '1.5rem'
              }}>
                <h4 style={{ color: 'var(--accent-cyan)', marginBottom: '0.5rem' }}>
                  📋 Supported formats:
                </h4>
                <ul style={{ color: 'var(--text-secondary)', paddingLeft: '1.5rem', lineHeight: '1.6' }}>
                  <li>Date formats: YYYY-MM-DD, MM/DD/YYYY, DD/MM/YYYY</li>
                  <li>Column names: Date/Datetime, Open, High, Low, Close, Volume (case-insensitive)</li>
                  <li>File format: CSV only</li>
                </ul>
              </div>

              {/* File Input */}
              <div style={{ marginBottom: '1.5rem' }}>
                <label style={{
                  display: 'block',
                  color: 'var(--text-primary)',
                  marginBottom: '0.5rem',
                  fontWeight: '500'
                }}>
                  Choose CSV File:
                </label>
                <input
                  id="file-input"
                  type="file"
                  accept=".csv"
                  onChange={handleFileSelect}
                  style={{
                    width: '100%',
                    padding: '0.75rem',
                    background: 'var(--bg-secondary)',
                    border: '2px solid var(--border)',
                    borderRadius: '8px',
                    color: 'var(--text-primary)',
                    fontFamily: 'var(--font-primary)'
                  }}
                />
              </div>

              {/* Dataset Configuration */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div>
                  <label style={{
                    display: 'block',
                    color: 'var(--text-primary)',
                    marginBottom: '0.5rem',
                    fontWeight: '500'
                  }}>
                    Dataset Name:
                  </label>
                  <input
                    type="text"
                    value={selectedDataset}
                    onChange={(e) => setSelectedDataset(e.target.value)}
                    placeholder="Enter dataset name"
                    style={{
                      width: '100%',
                      padding: '0.75rem',
                      background: 'var(--bg-secondary)',
                      border: '2px solid var(--border)',
                      borderRadius: '8px',
                      color: 'var(--text-primary)',
                      fontFamily: 'var(--font-primary)'
                    }}
                  />
                </div>
                <div>
                  <label style={{
                    display: 'block',
                    color: 'var(--text-primary)',
                    marginBottom: '0.5rem',
                    fontWeight: '500'
                  }}>
                    Purpose:
                  </label>
                  <select
                    value={datasetPurpose}
                    onChange={(e) => setDatasetPurpose(e.target.value)}
                    style={{
                      width: '100%',
                      padding: '0.75rem',
                      background: 'var(--bg-secondary)',
                      border: '2px solid var(--border)',
                      borderRadius: '8px',
                      color: 'var(--text-primary)',
                      fontFamily: 'var(--font-primary)'
                    }}
                  >
                    <option value="training">Training</option>
                    <option value="testing">Testing</option>
                    <option value="validation">Validation</option>
                    <option value="production">Production</option>
                  </select>
                </div>
              </div>

              {/* Upload Button */}
              <button
                onClick={handleUpload}
                disabled={loading || !uploadedFile}
                style={{
                  width: '100%',
                  padding: '1rem',
                  background: loading || !uploadedFile 
                    ? 'var(--bg-secondary)' 
                    : 'var(--gradient-primary)',
                  border: '2px solid var(--border-hover)',
                  borderRadius: '8px',
                  color: 'white',
                  fontFamily: 'var(--font-primary)',
                  fontWeight: '600',
                  fontSize: '1rem',
                  cursor: loading || !uploadedFile ? 'not-allowed' : 'pointer',
                  transition: 'all 0.3s ease'
                }}
              >
                {loading ? '⏳ Processing...' : '📤 Upload & Process Data'}
              </button>

              {/* Upload Progress */}
              {uploadProgress > 0 && (
                <div style={{ marginTop: '1rem' }}>
                  <div style={{
                    width: '100%',
                    height: '8px',
                    background: 'var(--bg-secondary)',
                    borderRadius: '4px',
                    overflow: 'hidden'
                  }}>
                    <div style={{
                      width: `${uploadProgress}%`,
                      height: '100%',
                      background: 'var(--gradient-primary)',
                      transition: 'width 0.3s ease'
                    }} />
                  </div>
                  <p style={{
                    color: 'var(--text-secondary)',
                    fontSize: '0.9rem',
                    marginTop: '0.5rem',
                    textAlign: 'center'
                  }}>
                    {uploadProgress}% Complete
                  </p>
                </div>
              )}

              {/* Status Message */}
              {uploadStatus && (
                <div style={{
                  marginTop: '1rem',
                  padding: '1rem',
                  background: uploadStatus.includes('❌') 
                    ? 'rgba(255, 0, 0, 0.1)' 
                    : 'rgba(0, 255, 0, 0.1)',
                  border: `1px solid ${uploadStatus.includes('❌') ? '#ff0000' : '#00ff00'}`,
                  borderRadius: '8px',
                  color: uploadStatus.includes('❌') ? '#ff6b6b' : '#51cf66',
                  fontFamily: 'var(--font-primary)'
                }}>
                  {uploadStatus}
                </div>
              )}

              {/* Processing Results */}
              {processingResults && (
                <div style={{
                  marginTop: '1.5rem',
                  padding: '1.5rem',
                  background: 'rgba(0, 255, 255, 0.05)',
                  border: '1px solid rgba(0, 255, 255, 0.2)',
                  borderRadius: '12px'
                }}>
                  <h4 style={{ color: 'var(--accent-cyan)', marginBottom: '1rem' }}>
                    📊 Processing Results
                  </h4>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <span style={{ color: 'var(--text-secondary)' }}>Records Processed:</span>
                      <span style={{ color: 'var(--accent-gold)', marginLeft: '0.5rem', fontWeight: '600' }}>
                        {processingResults.records_processed || 0}
                      </span>
                    </div>
                    <div>
                      <span style={{ color: 'var(--text-secondary)' }}>Date Range:</span>
                      <span style={{ color: 'var(--accent-gold)', marginLeft: '0.5rem', fontWeight: '600' }}>
                        {processingResults.date_range || 'N/A'}
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </Card>

          {/* Data Preview */}
          {previewData && (
            <Card style={{ marginTop: '2rem' }}>
              <h3 style={{
                color: 'var(--accent-cyan)',
                fontFamily: 'var(--font-display)',
                fontSize: '1.3rem',
                marginBottom: '1rem'
              }}>
                📈 Data Preview
              </h3>
              <div style={{ overflowX: 'auto' }}>
                <table style={{
                  width: '100%',
                  borderCollapse: 'collapse',
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.9rem'
                }}>
                  <thead>
                    <tr style={{ background: 'rgba(0, 255, 255, 0.1)' }}>
                      {previewData.columns?.map((col, index) => (
                        <th key={index} style={{
                          padding: '0.75rem',
                          textAlign: 'left',
                          color: 'var(--accent-cyan)',
                          borderBottom: '1px solid var(--border)'
                        }}>
                          {col}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {previewData.rows?.slice(0, 5).map((row, rowIndex) => (
                      <tr key={rowIndex}>
                        {row.map((cell, cellIndex) => (
                          <td key={cellIndex} style={{
                            padding: '0.75rem',
                            color: 'var(--text-primary)',
                            borderBottom: '1px solid rgba(255, 255, 255, 0.1)'
                          }}>
                            {cell}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}
        </div>

        {/* Current Datasets Sidebar */}
        <div>
          <Card>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h3 style={{
                color: 'var(--accent-cyan)',
                fontFamily: 'var(--font-display)',
                fontSize: '1.3rem',
                margin: '0'
              }}>
                📋 Current Datasets
              </h3>
              <button
                onClick={loadDatasets}
                disabled={loading}
                style={{
                  background: 'transparent',
                  border: '1px solid var(--border)',
                  borderRadius: '6px',
                  color: 'var(--accent-cyan)',
                  padding: '0.5rem',
                  cursor: 'pointer',
                  fontSize: '1rem'
                }}
                title="Refresh dataset list"
              >
                🔄
              </button>
            </div>

            {loading ? (
              <LoadingSpinner />
            ) : datasets.length === 0 ? (
              <div style={{
                textAlign: 'center',
                padding: '2rem',
                color: 'var(--text-secondary)'
              }}>
                <p>No datasets found</p>
                <p style={{ fontSize: '0.9rem' }}>Upload your first dataset to get started</p>
              </div>
            ) : (
              <div>
                {datasets.map((dataset, index) => (
                  <div key={index} style={{
                    background: 'rgba(0, 255, 255, 0.05)',
                    border: '1px solid rgba(0, 255, 255, 0.2)',
                    borderRadius: '8px',
                    padding: '1rem',
                    marginBottom: '0.75rem'
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                      <div style={{ flex: '1' }}>
                        <h4 style={{
                          color: 'var(--accent-cyan)',
                          margin: '0 0 0.5rem 0',
                          fontSize: '1rem',
                          fontFamily: 'var(--font-primary)'
                        }}>
                          {dataset.name}
                        </h4>
                        <div style={{
                          color: 'var(--text-secondary)',
                          fontSize: '0.85rem',
                          fontFamily: 'var(--font-mono)'
                        }}>
                          <div>📊 {dataset.rows || 0} rows</div>
                          <div>📅 {dataset.date_range || 'Unknown range'}</div>
                          <div>🎯 {dataset.purpose || 'General'}</div>
                        </div>
                      </div>
                      <button
                        onClick={() => handleDeleteDataset(dataset.name)}
                        style={{
                          background: 'transparent',
                          border: '1px solid #ff4757',
                          borderRadius: '4px',
                          color: '#ff4757',
                          padding: '0.25rem 0.5rem',
                          cursor: 'pointer',
                          fontSize: '0.8rem',
                          marginLeft: '0.5rem'
                        }}
                        title="Delete dataset"
                      >
                        🗑️
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
};

export default DataUpload;