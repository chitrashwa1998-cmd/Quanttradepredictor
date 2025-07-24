/**
 * Data Upload page - Upload and manage training datasets
 */

import { useState, useCallback } from 'react';
import { modelsAPI } from '../services/api';
import LoadingSpinner from '../components/common/LoadingSpinner';

export default function DataUpload() {
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  }, []);

  const handleFileUpload = async (file) => {
    if (!file) return;

    // Validate file type
    if (!file.name.toLowerCase().endsWith('.csv')) {
      setError('Please upload a CSV file');
      return;
    }

    try {
      setUploading(true);
      setError(null);
      setUploadResult(null);

      const result = await modelsAPI.uploadData(file);
      setUploadResult(result);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setUploading(false);
    }
  };

  const onFileInputChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileUpload(e.target.files[0]);
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold cyber-text mb-4">Upload Training Data</h1>
        <p className="text-gray-300">
          Upload historical OHLC data in CSV format for model training
        </p>
      </div>

      {/* Upload Area */}
      <div className="cyber-bg cyber-border rounded-lg p-8">
        <h2 className="text-xl font-semibold cyber-blue mb-6">File Upload</h2>
        
        <div
          className={`border-2 border-dashed rounded-lg p-12 text-center transition-colors ${
            dragActive
              ? 'border-cyber-blue bg-cyber-blue bg-opacity-10'
              : 'border-gray-600 hover:border-cyber-blue'
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          {uploading ? (
            <LoadingSpinner size="lg" text="Uploading file..." />
          ) : (
            <>
              <div className="cyber-text text-4xl mb-4">📄</div>
              <h3 className="text-lg font-medium text-white mb-2">
                Drop your CSV file here
              </h3>
              <p className="text-gray-400 mb-4">
                or click to browse files
              </p>
              <input
                type="file"
                accept=".csv"
                onChange={onFileInputChange}
                className="hidden"
                id="file-upload"
              />
              <label
                htmlFor="file-upload"
                className="cyber-border rounded-md py-2 px-4 text-cyber-blue hover:cyber-glow transition-all duration-200 cursor-pointer inline-block"
              >
                Browse Files
              </label>
            </>
          )}
        </div>

        {/* Requirements */}
        <div className="mt-6 bg-gray-800 rounded-lg p-4">
          <h3 className="text-sm font-semibold text-cyber-yellow mb-2">
            File Requirements:
          </h3>
          <ul className="text-sm text-gray-300 space-y-1">
            <li>• File format: CSV</li>
            <li>• Required columns: Date, Open, High, Low, Close</li>
            <li>• Date format: YYYY-MM-DD</li>
            <li>• Numeric values for OHLC data</li>
            <li>• No missing values in required columns</li>
          </ul>
        </div>
      </div>

      {/* Upload Result */}
      {uploadResult && (
        <div className="cyber-bg cyber-border rounded-lg p-6 slide-in-up">
          <h2 className="text-xl font-semibold text-cyber-green mb-4">
            ✅ Upload Successful
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h3 className="font-semibold text-cyber-blue mb-2">Dataset Info</h3>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Name:</span>
                  <span className="text-white font-mono">{uploadResult.dataset_name}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Rows:</span>
                  <span className="text-white font-mono">{uploadResult.rows?.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Columns:</span>
                  <span className="text-white font-mono">{uploadResult.columns?.length}</span>
                </div>
              </div>
            </div>
            <div>
              <h3 className="font-semibold text-cyber-blue mb-2">Date Range</h3>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Start:</span>
                  <span className="text-white font-mono">{uploadResult.date_range?.start}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">End:</span>
                  <span className="text-white font-mono">{uploadResult.date_range?.end}</span>
                </div>
              </div>
            </div>
          </div>
          
          <div className="mt-4 pt-4 border-t border-gray-700">
            <p className="text-sm text-gray-400">
              Dataset uploaded successfully and ready for model training.
            </p>
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="cyber-bg border border-cyber-red rounded-lg p-6 slide-in-up">
          <h2 className="text-xl font-semibold text-cyber-red mb-4">
            ❌ Upload Failed
          </h2>
          <p className="text-gray-300">{error}</p>
        </div>
      )}
    </div>
  );
}