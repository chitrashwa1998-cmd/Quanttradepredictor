/**
 * Data Upload page - Simplified working version
 */

import { useState, useCallback } from 'react';
import Card from '../components/common/Card';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { dataAPI } from '../services/api';

const DataUpload = () => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const [datasetName, setDatasetName] = useState('');
  const [dragActive, setDragActive] = useState(false);

  // Handle file selection
  const handleFileSelect = (selectedFile) => {
    if (selectedFile) {
      // Validate file type
      const validTypes = ['text/csv', 'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'];
      if (!validTypes.includes(selectedFile.type) && !selectedFile.name.endsWith('.csv')) {
        setUploadStatus('❌ Please select a CSV or Excel file');
        return;
      }

      setFile(selectedFile);
      setUploadStatus(`📄 Selected: ${selectedFile.name} (${(selectedFile.size / 1024 / 1024).toFixed(2)} MB)`);
      
      // Auto-generate dataset name from filename
      if (!datasetName) {
        const baseName = selectedFile.name.replace(/\.(csv|xlsx?|xls)$/i, '');
        setDatasetName(baseName);
      }
    }
  };

  // Handle drag and drop
  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  }, []);

  // Upload file
  const uploadFile = async () => {
    if (!file) {
      setUploadStatus('❌ Please select a file first');
      return;
    }

    if (!datasetName.trim()) {
      setUploadStatus('❌ Please enter a dataset name');
      return;
    }

    try {
      setUploading(true);
      setUploadStatus('📤 Uploading file...');

      const formData = new FormData();
      formData.append('file', file);
      formData.append('dataset_name', datasetName.trim());

      const response = await dataAPI.uploadData(formData);

      if (response.success) {
        setUploadStatus(`✅ Successfully uploaded! ${response.rows} rows stored as '${response.dataset_name}'`);
        setFile(null);
        setDatasetName('');
      } else {
        setUploadStatus(`❌ Upload failed: ${response.message || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Upload error:', error);
      setUploadStatus(`❌ Upload failed: ${error.response?.data?.detail || error.message}`);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-violet-800">
      <div className="container mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-600 mb-4">
            Data Upload
          </h1>
          <p className="text-gray-300 text-lg">
            Upload your OHLC trading data for analysis and model training
          </p>
        </div>

        {/* Status */}
        {uploadStatus && (
          <div className="mb-6">
            <div className={`p-4 rounded-lg ${
              uploadStatus.includes('✅') ? 'bg-green-900/30 border border-green-500/30 text-green-400' :
              uploadStatus.includes('❌') ? 'bg-red-900/30 border border-red-500/30 text-red-400' :
              'bg-blue-900/30 border border-blue-500/30 text-blue-400'
            }`}>
              {uploadStatus}
            </div>
          </div>
        )}

        <Card className="max-w-2xl mx-auto">
          <h2 className="text-2xl font-bold text-cyan-400 mb-6">📁 Upload OHLC Data</h2>

          {/* File Upload Area */}
          <div
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
              dragActive
                ? 'border-cyan-500 bg-cyan-900/20'
                : 'border-gray-600 hover:border-gray-500'
            }`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            {file ? (
              <div className="space-y-4">
                <div className="text-6xl">📄</div>
                <div>
                  <div className="text-xl font-bold text-white">{file.name}</div>
                  <div className="text-gray-400">
                    {(file.size / 1024 / 1024).toFixed(2)} MB
                  </div>
                </div>
                <button
                  onClick={() => {
                    setFile(null);
                    setUploadStatus('');
                  }}
                  className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded transition-colors"
                >
                  Remove File
                </button>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="text-6xl text-gray-400">📁</div>
                <div>
                  <div className="text-xl font-bold text-gray-300 mb-2">
                    Drop your CSV file here or click to browse
                  </div>
                  <div className="text-gray-400 text-sm">
                    Supported formats: CSV, Excel (.xlsx, .xls)
                  </div>
                </div>
                <input
                  type="file"
                  accept=".csv,.xlsx,.xls"
                  onChange={(e) => handleFileSelect(e.target.files[0])}
                  className="hidden"
                  id="file-input"
                />
                <label
                  htmlFor="file-input"
                  className="inline-block px-6 py-3 bg-cyan-600 hover:bg-cyan-700 text-white font-medium rounded-lg cursor-pointer transition-colors"
                >
                  Select File
                </label>
              </div>
            )}
          </div>

          {/* Dataset Name Input */}
          <div className="mt-6">
            <label className="block text-cyan-400 font-medium mb-2">
              Dataset Name:
            </label>
            <input
              type="text"
              value={datasetName}
              onChange={(e) => setDatasetName(e.target.value)}
              placeholder="Enter a name for your dataset..."
              className="w-full p-3 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:border-cyan-500 focus:outline-none"
            />
            <div className="text-sm text-gray-400 mt-1">
              This name will be used to identify your dataset in the system
            </div>
          </div>

          {/* Data Format Requirements */}
          <div className="mt-6 p-4 bg-blue-900/20 border border-blue-500/30 rounded-lg">
            <h3 className="text-blue-400 font-bold mb-2">📋 Data Format Requirements</h3>
            <div className="text-sm text-gray-300 space-y-1">
              <p>• <strong>Required columns:</strong> Open, High, Low, Close</p>
              <p>• <strong>Optional columns:</strong> Volume, DateTime/Date, Time</p>
              <p>• <strong>File formats:</strong> CSV, Excel (.xlsx, .xls)</p>
              <p>• <strong>Max file size:</strong> 100MB</p>
            </div>
          </div>

          {/* Upload Button */}
          <div className="mt-6">
            <button
              onClick={uploadFile}
              disabled={!file || !datasetName.trim() || uploading}
              className="w-full px-6 py-3 bg-green-600 hover:bg-green-700 text-white font-medium rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
            >
              {uploading && <LoadingSpinner size="sm" />}
              <span>
                {uploading ? 'Uploading...' : 'Upload Data'}
              </span>
            </button>
          </div>
        </Card>

        {/* Data Preview Section - Placeholder */}
        <Card className="max-w-4xl mx-auto mt-8">
          <h2 className="text-2xl font-bold text-cyan-400 mb-4">📊 Data Preview</h2>
          <div className="text-center py-8 text-gray-400">
            <p>Upload a file to see data preview and validation results</p>
          </div>
        </Card>
      </div>
    </div>
  );
};

export default DataUpload;