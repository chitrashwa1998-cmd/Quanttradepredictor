/**
 * Data Upload page - Exact Streamlit UI replication
 */

import { useState, useEffect } from 'react';
import { dataAPI } from '../services/api';

const DataUpload = () => {
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const [databaseInfo, setDatabaseInfo] = useState(null);

  // Load datasets and database info
  const loadData = async () => {
    try {
      const datasetsResponse = await dataAPI.getDatasets();
      const dbResponse = await dataAPI.getDatabaseInfo();
      
      setDatasets(datasetsResponse?.data || []);
      setDatabaseInfo(dbResponse?.data || {});
    } catch (error) {
      console.error('Error loading data:', error);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  // Handle file upload
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    try {
      setLoading(true);
      setUploadStatus('Uploading file...');
      
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await dataAPI.uploadData(formData);
      
      if (response?.data?.success) {
        setUploadStatus(`✅ Successfully uploaded ${file.name}`);
        await loadData(); // Refresh data
      } else {
        setUploadStatus(`❌ Upload failed: ${response?.data?.message || 'Unknown error'}`);
      }
    } catch (error) {
      setUploadStatus(`❌ Upload failed: ${error?.response?.data?.detail || error?.message || 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ backgroundColor: '#0a0a0f', minHeight: '100vh', color: '#ffffff', fontFamily: 'Space Grotesk, sans-serif' }}>
      <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '2rem' }}>
        {/* Header - Exact Streamlit Style */}
        <div className="trading-header" style={{
          background: 'linear-gradient(145deg, rgba(25, 25, 45, 0.9), rgba(35, 35, 55, 0.6))',
          border: '2px solid rgba(0, 255, 255, 0.2)',
          borderRadius: '20px',
          padding: '2rem',
          marginBottom: '2rem',
          boxShadow: '0 0 20px rgba(0, 255, 255, 0.1)'
        }}>
          <h1 style={{ margin: 0, fontSize: '2.5rem', fontFamily: 'Orbitron, monospace' }}>
            📊 DATA UPLOAD CENTER
          </h1>
          <p style={{ fontSize: '1.2rem', margin: '1rem 0 0 0', color: 'rgba(255,255,255,0.8)' }}>
            Load and Process Market Data
          </p>
        </div>

        {/* Main Content Area */}
        <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '2rem' }}>
          {/* Left Column - Upload */}
          <div>
            {/* Upload Section */}
            <h2 style={{ fontSize: '1.8rem', marginBottom: '1rem', color: '#ffffff' }}>Upload OHLC Data</h2>
            <div style={{ marginBottom: '2rem', color: '#b8bcc8' }}>
              <p style={{ marginBottom: '1rem' }}>
                Upload your historical price data in CSV format. The file should contain columns for Date, Open, High, Low, Close, and optionally Volume.
              </p>
              <p style={{ fontWeight: 'bold', marginBottom: '0.5rem' }}>Supported formats:</p>
              <ul style={{ marginLeft: '1.5rem', lineHeight: '1.6' }}>
                <li>Date formats: YYYY-MM-DD, MM/DD/YYYY, DD/MM/YYYY</li>
                <li>Column names: Date/Datetime, Open, High, Low, Close, Volume (case-insensitive)</li>
              </ul>
            </div>

            {/* File Upload */}
            <div style={{ 
              border: '2px dashed rgba(0, 255, 255, 0.3)', 
              borderRadius: '8px', 
              padding: '2rem', 
              textAlign: 'center',
              backgroundColor: 'rgba(25, 25, 45, 0.5)',
              marginBottom: '2rem'
            }}>
              <input
                type="file"
                accept=".csv"
                onChange={handleFileUpload}
                disabled={loading}
                style={{
                  display: 'block',
                  width: '100%',
                  padding: '1rem',
                  border: '1px solid rgba(0, 255, 255, 0.3)',
                  borderRadius: '4px',
                  backgroundColor: 'rgba(0, 0, 0, 0.3)',
                  color: '#ffffff',
                  fontSize: '1rem'
                }}
              />
              <p style={{ marginTop: '1rem', color: '#b8bcc8', fontSize: '0.9rem' }}>
                Choose a CSV file with OHLC data
              </p>
            </div>

            {/* Status Message */}
            {uploadStatus && (
              <div style={{
                padding: '1rem',
                borderRadius: '8px',
                backgroundColor: uploadStatus.includes('✅') ? 'rgba(0, 255, 65, 0.1)' : 'rgba(255, 0, 128, 0.1)',
                border: `1px solid ${uploadStatus.includes('✅') ? 'rgba(0, 255, 65, 0.3)' : 'rgba(255, 0, 128, 0.3)'}`,
                color: uploadStatus.includes('✅') ? '#00ff41' : '#ff0080',
                marginBottom: '2rem'
              }}>
                {uploadStatus}
              </div>
            )}

            {/* Data Storage Options */}
            <h3 style={{ fontSize: '1.5rem', marginBottom: '1rem', color: '#ffffff' }}>📊 Data Storage Options</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              <div style={{
                padding: '1.5rem',
                backgroundColor: 'rgba(25, 25, 45, 0.5)',
                border: '1px solid rgba(0, 255, 255, 0.2)',
                borderRadius: '8px'
              }}>
                <h4 style={{ color: '#00ffff', marginBottom: '0.5rem' }}>Training Data</h4>
                <p style={{ color: '#b8bcc8', fontSize: '0.9rem' }}>
                  Historical data for model training and backtesting
                </p>
              </div>
              <div style={{
                padding: '1.5rem',
                backgroundColor: 'rgba(25, 25, 45, 0.5)',
                border: '1px solid rgba(0, 255, 255, 0.2)',
                borderRadius: '8px'
              }}>
                <h4 style={{ color: '#00ffff', marginBottom: '0.5rem' }}>Live Data</h4>
                <p style={{ color: '#b8bcc8', fontSize: '0.9rem' }}>
                  Recent market data for live predictions
                </p>
              </div>
            </div>
          </div>

          {/* Right Column - Current Datasets (Sidebar style) */}
          <div style={{
            backgroundColor: 'rgba(21, 21, 32, 0.8)',
            border: '1px solid rgba(0, 255, 255, 0.2)',
            borderRadius: '8px',
            padding: '1.5rem',
            height: 'fit-content'
          }}>
            {/* Header with Refresh */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h3 style={{ color: '#ffffff', margin: 0 }}>📋 Current Datasets</h3>
              <button
                onClick={loadData}
                style={{
                  backgroundColor: 'transparent',
                  border: '1px solid rgba(0, 255, 255, 0.3)',
                  borderRadius: '4px',
                  color: '#00ffff',
                  padding: '0.5rem',
                  cursor: 'pointer',
                  fontSize: '1rem'
                }}
                title="Refresh dataset list"
              >
                🔄
              </button>
            </div>

            {/* Dataset List */}
            {datasets.length > 0 ? (
              <div>
                {datasets.map((dataset, index) => (
                  <div key={index} style={{
                    backgroundColor: 'rgba(0, 255, 65, 0.1)',
                    border: '1px solid rgba(0, 255, 65, 0.3)',
                    borderRadius: '4px',
                    padding: '0.75rem',
                    marginBottom: '0.5rem',
                    color: '#00ff41'
                  }}>
                    📊 {dataset.name} ({dataset.rows} rows)
                  </div>
                ))}
              </div>
            ) : (
              <div style={{
                backgroundColor: 'rgba(0, 255, 255, 0.1)',
                border: '1px solid rgba(0, 255, 255, 0.3)',
                borderRadius: '4px',
                padding: '0.75rem',
                color: '#00ffff'
              }}>
                📊 No datasets uploaded yet
              </div>
            )}

            {/* Database Status */}
            <div style={{ marginTop: '1.5rem', paddingTop: '1rem', borderTop: '1px solid rgba(0, 255, 255, 0.2)' }}>
              <h4 style={{ color: '#ffffff', marginBottom: '0.5rem' }}>Database Status:</h4>
              <div style={{ color: '#b8bcc8', fontSize: '0.9rem', lineHeight: '1.5' }}>
                <div>Total Datasets: {databaseInfo?.total_datasets || 0}</div>
                <div>Total Records: {databaseInfo?.total_records || 0}</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DataUpload;