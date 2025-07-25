
/**
 * Data Upload page - Complete Streamlit functionality migration with cyberpunk theme
 */

import { useState } from 'react';
import Card from '../components/common/Card';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { dataAPI } from '../services/api';

export default function DataUpload() {
  const [file, setFile] = useState(null);
  const [datasetName, setDatasetName] = useState('');
  const [datasetPurpose, setDatasetPurpose] = useState('training');
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState('');
  const [dragOver, setDragOver] = useState(false);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    setFile(selectedFile);

    if (selectedFile && !datasetName) {
      // Auto-generate dataset name from filename
      const name = selectedFile.name.replace(/\.[^/.]+$/, "");
      setDatasetName(name);
    }
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragOver(false);
    
    const droppedFile = event.dataTransfer.files[0];
    if (droppedFile && (droppedFile.type === 'text/csv' || droppedFile.name.endsWith('.csv'))) {
      setFile(droppedFile);
      if (!datasetName) {
        const name = droppedFile.name.replace(/\.[^/.]+$/, "");
        setDatasetName(name);
      }
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (event) => {
    event.preventDefault();
    setDragOver(false);
  };

  const handleUpload = async () => {
    if (!file) {
      setMessage('Please select a file to upload');
      setMessageType('error');
      return;
    }

    if (!datasetName.trim()) {
      setMessage('Please provide a dataset name');
      setMessageType('error');
      return;
    }

    setUploading(true);
    setMessage('');

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('dataset_name', datasetName.trim());
      formData.append('dataset_purpose', datasetPurpose);

      const response = await dataAPI.uploadData(formData);

      if (response.status === 200) {
        setMessage(`Successfully uploaded ${datasetName} for ${datasetPurpose}`);
        setMessageType('success');
        setFile(null);
        setDatasetName('');
        // Reset file input
        const fileInput = document.querySelector('input[type="file"]');
        if (fileInput) fileInput.value = '';
      }
    } catch (error) {
      console.error('Upload error:', error);
      setMessage(`Upload failed: ${error.response?.data?.detail || error.message}`);
      setMessageType('error');
    } finally {
      setUploading(false);
    }
  };

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

        {/* Upload Form */}
        <div className="cyber-card mb-6">
          <h2 className="cyber-subtitle mb-6">Upload OHLC Data</h2>
          
          <div className="mb-6">
            <p className="cyber-text mb-4">
              Upload your historical price data in CSV format. The file should contain columns for Date, Open, High, Low, Close, and optionally Volume.
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
              <div className="cyber-alert cyber-alert-info">
                <strong>Supported formats:</strong><br/>
                Date formats: YYYY-MM-DD, MM/DD/YYYY, DD/MM/YYYY
              </div>
              <div className="cyber-alert cyber-alert-info">
                <strong>Column names:</strong><br/>
                Date/Datetime, Open, High, Low, Close, Volume (case-insensitive)
              </div>
              <div className="cyber-alert cyber-alert-warning">
                <strong>File size limit:</strong><br/>
                Maximum 100MB for optimal performance
              </div>
            </div>
          </div>

          {/* Dataset Configuration */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            <div>
              <label className="block text-sm font-medium mb-2 cyber-subtitle">
                🎯 Dataset Purpose
              </label>
              <select
                value={datasetPurpose}
                onChange={(e) => setDatasetPurpose(e.target.value)}
                className="cyber-select w-full"
              >
                <option value="training">Training - Used for model training</option>
                <option value="pre_seed">Pre-seed - Used for live data seeding</option>
                <option value="validation">Validation - Used for model validation</option>
                <option value="testing">Testing - Used for model testing</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2 cyber-subtitle">
                📊 Dataset Name
              </label>
              <input
                type="text"
                value={datasetName}
                onChange={(e) => setDatasetName(e.target.value)}
                placeholder={`${datasetPurpose}_dataset`}
                className="cyber-input w-full"
              />
            </div>
          </div>

          {/* File Upload Area */}
          <div 
            className={`file-upload-area ${dragOver ? 'dragover' : ''} mb-6`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onClick={() => document.getElementById('file-input').click()}
          >
            <input
              id="file-input"
              type="file"
              onChange={handleFileChange}
              accept=".csv,.xlsx,.xls"
              className="hidden"
            />
            
            {file ? (
              <div className="text-center">
                <div className="cyber-mono text-2xl mb-4">📁</div>
                <p className="cyber-subtitle mb-2">Selected File:</p>
                <p className="cyber-text text-lg font-semibold">{file.name}</p>
                <p className="cyber-mono mt-2">
                  Size: {(file.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
            ) : (
              <div className="text-center">
                <div className="cyber-mono text-4xl mb-4">⬆️</div>
                <p className="cyber-subtitle text-xl mb-2">Drop your CSV file here</p>
                <p className="cyber-text">or click to browse files</p>
                <p className="cyber-mono mt-4 text-sm">
                  Supported: CSV, Excel (.xlsx, .xls)
                </p>
              </div>
            )}
          </div>

          {/* Upload Button */}
          <button
            onClick={handleUpload}
            disabled={!file || !datasetName.trim() || uploading}
            className={`cyber-button w-full py-4 text-lg ${
              (!file || !datasetName.trim()) ? 'opacity-50 cursor-not-allowed' : ''
            }`}
          >
            {uploading ? (
              <div className="flex items-center justify-center">
                <div className="cyber-spinner mr-3"></div>
                <span>Processing Data...</span>
              </div>
            ) : (
              <>
                <span className="mr-2">💾</span>
                Upload Dataset
              </>
            )}
          </button>

          {/* Message Display */}
          {message && (
            <div className={`mt-6 cyber-alert ${
              messageType === 'success' ? 'cyber-alert-success' : 'cyber-alert-error'
            }`}>
              <div className="flex items-center">
                <span className="mr-2">
                  {messageType === 'success' ? '✅' : '❌'}
                </span>
                {message}
              </div>
            </div>
          )}
        </div>

        {/* Guidelines Card */}
        <div className="cyber-card">
          <h2 className="cyber-subtitle mb-6">📋 Upload Guidelines</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="cyber-subtitle text-lg mb-4">📄 File Format Requirements</h3>
              <div className="space-y-3">
                <div className="flex items-start">
                  <span className="cyber-mono mr-3">•</span>
                  <p className="cyber-text">CSV files should have headers in the first row</p>
                </div>
                <div className="flex items-start">
                  <span className="cyber-mono mr-3">•</span>
                  <p className="cyber-text">For OHLCV data, include columns: timestamp, open, high, low, close, volume</p>
                </div>
                <div className="flex items-start">
                  <span className="cyber-mono mr-3">•</span>
                  <p className="cyber-text">Timestamp should be in a recognizable format (ISO 8601 recommended)</p>
                </div>
              </div>
            </div>

            <div>
              <h3 className="cyber-subtitle text-lg mb-4">⚡ Quality Standards</h3>
              <div className="space-y-3">
                <div className="flex items-start">
                  <span className="cyber-mono mr-3">•</span>
                  <p className="cyber-text">Ensure data is clean and contains no missing critical values</p>
                </div>
                <div className="flex items-start">
                  <span className="cyber-mono mr-3">•</span>
                  <p className="cyber-text">All price values must be positive numbers</p>
                </div>
                <div className="flex items-start">
                  <span className="cyber-mono mr-3">•</span>
                  <p className="cyber-text">High ≥ Low, High ≥ Open, High ≥ Close constraints</p>
                </div>
              </div>
            </div>
          </div>

          {/* Sample Data Format */}
          <div className="mt-8">
            <h3 className="cyber-subtitle text-lg mb-4">📊 Expected Data Format</h3>
            <div className="cyber-table-container">
              <table className="cyber-table">
                <thead>
                  <tr>
                    <th>Date</th>
                    <th>Open</th>
                    <th>High</th>
                    <th>Low</th>
                    <th>Close</th>
                    <th>Volume</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>2023-01-01</td>
                    <td>100.0</td>
                    <td>105.0</td>
                    <td>99.0</td>
                    <td>104.0</td>
                    <td>1000000</td>
                  </tr>
                  <tr>
                    <td>2023-01-02</td>
                    <td>101.0</td>
                    <td>106.0</td>
                    <td>100.0</td>
                    <td>105.0</td>
                    <td>1100000</td>
                  </tr>
                  <tr>
                    <td>2023-01-03</td>
                    <td>102.0</td>
                    <td>107.0</td>
                    <td>101.0</td>
                    <td>106.0</td>
                    <td>1200000</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Next Steps */}
        <div className="mt-8 cyber-alert cyber-alert-info">
          <div className="flex items-center">
            <span className="cyber-mono text-2xl mr-4">📋</span>
            <div>
              <strong className="cyber-subtitle">Next Steps:</strong>
              <p className="cyber-text mt-1">
                Once your data is loaded and processed, go to the <strong>Model Training</strong> page to train the XGBoost models.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
