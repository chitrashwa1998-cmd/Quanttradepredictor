/**
 * Data Upload page - Complete Streamlit functionality migration
 */

import { useState } from 'react';
import Card from '../components/common/Card';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { dataAPI } from '../services/api';

export default function DataUpload() {
  const [file, setFile] = useState(null);
  const [datasetName, setDatasetName] = useState('');
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState('');

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    setFile(selectedFile);

    if (selectedFile && !datasetName) {
      // Auto-generate dataset name from filename
      const name = selectedFile.name.replace(/\.[^/.]+$/, "");
      setDatasetName(name);
    }
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

      const response = await dataAPI.uploadData(formData);

      if (response.status === 200) {
        setMessage(`Successfully uploaded ${datasetName}`);
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
    <div className="min-h-screen bg-gray-900 p-6">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-white mb-4">📊 Data Upload</h1>
          <p className="text-gray-300">Upload your trading data files for analysis and model training</p>
        </div>

        <Card className="mb-6">
          <h2 className="text-xl font-semibold text-white mb-4">Upload New Dataset</h2>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Dataset Name
              </label>
              <input
                type="text"
                value={datasetName}
                onChange={(e) => setDatasetName(e.target.value)}
                placeholder="Enter dataset name"
                className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-md text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Select File
              </label>
              <input
                type="file"
                onChange={handleFileChange}
                accept=".csv,.xlsx,.xls"
                className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-md text-white file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-700"
              />
              <p className="text-sm text-gray-400 mt-1">
                Supported formats: CSV, Excel (.xlsx, .xls)
              </p>
            </div>

            {file && (
              <div className="bg-gray-800 p-3 rounded-md">
                <p className="text-sm text-gray-300">
                  <span className="font-medium">Selected file:</span> {file.name}
                </p>
                <p className="text-sm text-gray-400">
                  Size: {(file.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
            )}

            <button
              onClick={handleUpload}
              disabled={!file || !datasetName.trim() || uploading}
              className="w-full py-3 px-4 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors flex items-center justify-center"
            >
              {uploading ? (
                <>
                  <LoadingSpinner size="sm" className="mr-2" />
                  Uploading...
                </>
              ) : (
                'Upload Dataset'
              )}
            </button>
          </div>

          {message && (
            <div className={`mt-4 p-3 rounded-md ${
              messageType === 'success' 
                ? 'bg-green-900 text-green-300 border border-green-700' 
                : 'bg-red-900 text-red-300 border border-red-700'
            }`}>
              {message}
            </div>
          )}
        </Card>

        <Card>
          <h2 className="text-xl font-semibold text-white mb-4">Upload Guidelines</h2>
          <div className="space-y-3 text-gray-300">
            <div className="flex items-start">
              <span className="text-blue-400 mr-2">•</span>
              <p>CSV files should have headers in the first row</p>
            </div>
            <div className="flex items-start">
              <span className="text-blue-400 mr-2">•</span>
              <p>For OHLCV data, include columns: timestamp, open, high, low, close, volume</p>
            </div>
            <div className="flex items-start">
              <span className="text-blue-400 mr-2">•</span>
              <p>Timestamp should be in a recognizable format (ISO 8601 recommended)</p>
            </div>
            <div className="flex items-start">
              <span className="text-blue-400 mr-2">•</span>
              <p>Maximum file size: 100MB</p>
            </div>
            <div className="flex items-start">
              <span className="text-blue-400 mr-2">•</span>
              <p>Ensure data is clean and contains no missing critical values</p>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}