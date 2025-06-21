import React, { useState } from 'react';
import axios from 'axios';
import { API_BASE_URL } from '../config/api';

const DataUpload = () => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
    setUploadResult(null);
  };

  const uploadFile = async () => {
    if (!file) {
      alert('Please select a file first');
      return;
    }

    setUploading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await axios.post(`${API_BASE_URL}/api/upload-data`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      if (response.data.success) {
        setUploadResult({
          success: true,
          message: response.data.message
        });
      } else {
        setUploadResult({
          success: false,
          message: response.data.error || 'Upload failed'
        });
      }
    } catch (error) {
      setUploadResult({
        success: false,
        message: error.response?.data?.error || 'Upload failed'
      });
    } finally {
      setUploading(false);
    }
  };

  

  return (
    <div className="container">
      <div className="header">
        <h1>📊 DATA UPLOAD CENTER</h1>
        <p>Load and Process Market Data</p>
      </div>

      <div className="card">
        <h2>Upload OHLC Data</h2>
        <p style={{marginBottom: '2rem', color: '#b8bcc8'}}>
          Upload your historical price data in CSV format. The file should contain columns for Date, Open, High, Low, Close, and optionally Volume.
        </p>

        <div style={{marginBottom: '2rem'}}>
          <strong>Supported formats:</strong>
          <ul style={{marginTop: '0.5rem', color: '#b8bcc8'}}>
            <li>Date formats: YYYY-MM-DD, MM/DD/YYYY, DD/MM/YYYY</li>
            <li>Column names: Date/Datetime, Open, High, Low, Close, Volume (case-insensitive)</li>
          </ul>
        </div>

        <div style={{marginBottom: '2rem'}}>
          <input
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            style={{
              padding: '1rem',
              border: '2px dashed #00ffff',
              borderRadius: '8px',
              background: 'rgba(0, 255, 255, 0.05)',
              color: '#ffffff',
              width: '100%'
            }}
          />
        </div>

        {file && (
          <div className="alert alert-success" style={{marginBottom: '2rem'}}>
            File selected: {file.name} ({(file.size / 1024).toFixed(2)} KB)
          </div>
        )}

        <button
          className="btn btn-primary"
          onClick={uploadFile}
          disabled={!file || uploading}
          style={{opacity: (!file || uploading) ? 0.5 : 1}}
        >
          {uploading ? 'Uploading...' : '📤 Upload File'}
        </button>

        {uploadResult && (
          <div className={`alert ${uploadResult.success ? 'alert-success' : 'alert-error'}`} style={{marginTop: '2rem'}}>
            {uploadResult.success ? '✅' : '❌'} {uploadResult.message}
          </div>
        )}
      </div>

      <div className="card">
        <h3>Expected Data Format</h3>
        <table className="table">
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

        <div style={{marginTop: '2rem'}}>
          <strong>Column Requirements:</strong>
          <ul style={{marginTop: '0.5rem', color: '#b8bcc8'}}>
            <li><strong>Date/Datetime:</strong> Any standard date format</li>
            <li><strong>Open:</strong> Opening price (numeric)</li>
            <li><strong>High:</strong> Highest price (numeric)</li>
            <li><strong>Low:</strong> Lowest price (numeric)</li>
            <li><strong>Close:</strong> Closing price (numeric)</li>
            <li><strong>Volume:</strong> Trading volume (optional, numeric)</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default DataUpload;