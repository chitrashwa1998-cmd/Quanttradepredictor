/**
 * Data Upload page - Complete Streamlit functionality migration
 */

import { useState, useEffect, useCallback } from 'react';
import Card from '../components/common/Card';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { dataAPI } from '../services/api';

const DataUpload = () => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const [datasetName, setDatasetName] = useState('');
  const [dragActive, setDragActive] = useState(false);
  const [datasets, setDatasets] = useState([]);
  const [datasetPurpose, setDatasetPurpose] = useState('training');
  const [preserveFullData, setPreserveFullData] = useState(true);
  const [previewData, setPreviewData] = useState(null);
  const [validationResults, setValidationResults] = useState(null);
  const [processingResults, setProcessingResults] = useState(null);

  // Load datasets on component mount
  useEffect(() => {
    loadDatasets();
  }, []);

  const loadDatasets = async () => {
    try {
      const response = await dataAPI.getDatasets().catch(() => ({ data: [] }));
      setDatasets(response.data || []);
    } catch (error) {
      console.error('Error loading datasets:', error);
    }
  };

  // Handle file selection with comprehensive validation
  const handleFileSelect = (selectedFile) => {
    if (selectedFile) {
      // File type validation
      const validTypes = ['text/csv', 'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'];
      const validExtensions = ['.csv', '.xlsx', '.xls'];
      const hasValidType = validTypes.includes(selectedFile.type) || validExtensions.some(ext => selectedFile.name.toLowerCase().endsWith(ext));
      
      if (!hasValidType) {
        setUploadStatus('❌ Please select a CSV or Excel file (.csv, .xlsx, .xls)');
        return;
      }

      // File size validation (25MB limit)
      const maxSize = 25 * 1024 * 1024; // 25MB
      if (selectedFile.size > maxSize) {
        setUploadStatus(`❌ File too large. Maximum size: ${(maxSize / 1024 / 1024).toFixed(1)}MB. Current size: ${(selectedFile.size / 1024 / 1024).toFixed(1)}MB`);
        return;
      }

      setFile(selectedFile);
      setUploadStatus(`📄 Selected: ${selectedFile.name} (${(selectedFile.size / 1024 / 1024).toFixed(2)} MB)`);
      
      // Auto-generate dataset name from filename if not set
      if (!datasetName) {
        const baseName = selectedFile.name.replace(/\.(csv|xlsx?|xls)$/i, '');
        setDatasetName(baseName);
      }

      // Preview file content for validation
      previewFileData(selectedFile);
    }
  };

  // Preview and validate file data (matching Streamlit functionality)
  const previewFileData = async (file) => {
    try {
      const text = await file.text();
      const lines = text.split('\n').slice(0, 10); // First 10 lines
      setPreviewData(lines);

      // Basic validation checks
      const firstLine = lines[0];
      const headers = firstLine.split(',').map(h => h.trim().toLowerCase());
      
      const requiredColumns = ['open', 'high', 'low', 'close'];
      const hasAllRequired = requiredColumns.every(col => 
        headers.some(h => h.includes(col))
      );

      const hasDateColumn = headers.some(h => 
        ['date', 'datetime', 'timestamp', 'time'].some(dateCol => h.includes(dateCol))
      );

      const validation = {
        hasRequiredColumns: hasAllRequired,
        hasDateColumn: hasDateColumn,
        detectedColumns: headers,
        estimatedRows: text.split('\n').length - 1,
        fileSize: (file.size / 1024 / 1024).toFixed(2) + ' MB'
      };

      setValidationResults(validation);

      if (!hasAllRequired) {
        setUploadStatus('⚠️ Warning: Missing required OHLC columns (Open, High, Low, Close)');
      } else if (!hasDateColumn) {
        setUploadStatus('⚠️ Warning: No date/time column detected');
      } else {
        setUploadStatus('✅ File validation passed. Ready to upload.');
      }

    } catch (error) {
      setUploadStatus('❌ Error reading file content');
      console.error('File preview error:', error);
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

  // Upload file with comprehensive processing
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
      setUploadStatus('📤 Processing and uploading file...');

      const formData = new FormData();
      formData.append('file', file);
      formData.append('dataset_name', datasetName.trim());
      formData.append('purpose', datasetPurpose);
      formData.append('preserve_full_data', preserveFullData.toString());

      const response = await dataAPI.uploadData(formData);

      if (response.success) {
        const results = {
          dataset_name: response.dataset_name,
          rows_processed: response.rows,
          columns: response.columns,
          upload_time: new Date().toLocaleString(),
          file_size: (file.size / 1024 / 1024).toFixed(2) + ' MB',
          processing_notes: response.processing_notes || []
        };

        setProcessingResults(results);
        setUploadStatus(`✅ Successfully uploaded! ${response.rows.toLocaleString()} rows stored as '${response.dataset_name}'`);
        
        // Clear form
        setFile(null);
        setDatasetName('');
        setPreviewData(null);
        setValidationResults(null);
        
        // Reload datasets
        await loadDatasets();
        
      } else {
        setUploadStatus(`❌ Upload failed: ${response.message || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Upload error:', error);
      const errorMessage = error.response?.data?.detail || error.message || 'Upload failed';
      setUploadStatus(`❌ Upload failed: ${errorMessage}`);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-violet-800">
      <div className="container mx-auto px-4 py-8">
        {/* Header - Matching original Streamlit design */}
        <div className="text-center mb-8">
          <h1 className="text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-600 mb-4">
            📊 DATA UPLOAD CENTER
          </h1>
          <p className="text-xl text-gray-300 mb-2">
            Load and Process Market Data
          </p>
          <p className="text-gray-400">
            Upload your historical price data in CSV format with comprehensive validation and processing
          </p>
        </div>

        {/* Status */}
        {uploadStatus && (
          <div className="mb-6">
            <div className={`p-4 rounded-lg ${
              uploadStatus.includes('✅') ? 'bg-green-900/30 border border-green-500/30 text-green-400' :
              uploadStatus.includes('❌') ? 'bg-red-900/30 border border-red-500/30 text-red-400' :
              uploadStatus.includes('⚠️') ? 'bg-yellow-900/30 border border-yellow-500/30 text-yellow-400' :
              'bg-blue-900/30 border border-blue-500/30 text-blue-400'
            }`}>
              {uploadStatus}
            </div>
          </div>
        )}

        {/* File Upload Section */}
        <Card className="mb-6">
          <h2 className="text-2xl font-bold text-cyan-400 mb-6">Upload OHLC Data</h2>
          
          {/* File Upload Area */}
          <div
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-all duration-200 ${
              dragActive
                ? 'border-cyan-500 bg-cyan-900/20 scale-105'
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
                    setPreviewData(null);
                    setValidationResults(null);
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
                    Supported formats: CSV, Excel (.xlsx, .xls) • Max size: 25MB
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

          {/* Data Format Requirements - Matching Streamlit */}
          <div className="mt-6 p-4 bg-blue-900/20 border border-blue-500/30 rounded-lg">
            <h3 className="text-blue-400 font-bold mb-2">📋 Data Format Requirements</h3>
            <div className="text-sm text-gray-300 space-y-1">
              <p><strong>Supported formats:</strong></p>
              <p>• Date formats: YYYY-MM-DD, MM/DD/YYYY, DD/MM/YYYY</p>
              <p>• Column names: Date/Datetime, Open, High, Low, Close, Volume (case-insensitive)</p>
              <p><strong>Required columns:</strong> Open, High, Low, Close</p>
              <p><strong>Optional columns:</strong> Volume, DateTime/Date, Time</p>
              <p><strong>File formats:</strong> CSV, Excel (.xlsx, .xls)</p>
              <p><strong>Max file size:</strong> 25MB</p>
            </div>
          </div>
        </Card>

        {/* Data Storage Options - Matching Streamlit */}
        <Card className="mb-6">
          <h2 className="text-2xl font-bold text-cyan-400 mb-6">📊 Data Storage Options</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            <div>
              <label className="flex items-center space-x-3">
                <input
                  type="checkbox"
                  checked={preserveFullData}
                  onChange={(e) => setPreserveFullData(e.target.checked)}
                  className="form-checkbox h-5 w-5 text-cyan-600"
                />
                <div>
                  <div className="font-medium text-white">Preserve Full Dataset</div>
                  <div className="text-sm text-gray-400">Keep all data points without sampling. Recommended for most datasets.</div>
                </div>
              </label>
            </div>
            
            <div className={`p-4 rounded-lg ${preserveFullData ? 'bg-green-900/20 border border-green-500/30' : 'bg-yellow-900/20 border border-yellow-500/30'}`}>
              {preserveFullData ? (
                <div className="text-green-400">
                  <div className="font-bold">✅ Full dataset will be preserved</div>
                  <div className="text-sm">All data points will be stored</div>
                </div>
              ) : (
                <div className="text-yellow-400">
                  <div className="font-bold">⚠️ Large datasets will be intelligently sampled</div>
                  <div className="text-sm">Maximum 50,000 rows will be stored</div>
                </div>
              )}
            </div>
          </div>

          {/* Dataset Purpose */}
          <div className="mb-6">
            <label className="block text-cyan-400 font-medium mb-2">🎯 Dataset Purpose:</label>
            <select
              value={datasetPurpose}
              onChange={(e) => setDatasetPurpose(e.target.value)}
              className="w-full p-3 bg-gray-800 border border-gray-600 rounded-lg text-white focus:border-cyan-500 focus:outline-none"
            >
              <option value="training">Training - For model training and development</option>
              <option value="pre_seed">Pre-seed - Initial dataset for testing</option>
              <option value="validation">Validation - For model validation and testing</option>
              <option value="testing">Testing - For final model evaluation</option>
            </select>
          </div>

          {/* Dataset Name Input */}
          <div>
            <label className="block text-cyan-400 font-medium mb-2">Dataset Name:</label>
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
        </Card>

        {/* File Preview and Validation */}
        {(previewData || validationResults) && (
          <Card className="mb-6">
            <h2 className="text-2xl font-bold text-cyan-400 mb-4">👁️ File Preview & Validation</h2>
            
            {validationResults && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                <div className="bg-gray-800/50 p-3 rounded">
                  <div className="text-sm text-gray-400">Required Columns</div>
                  <div className={`text-lg font-bold ${validationResults.hasRequiredColumns ? 'text-green-400' : 'text-red-400'}`}>
                    {validationResults.hasRequiredColumns ? '✅ Found' : '❌ Missing'}
                  </div>
                </div>
                <div className="bg-gray-800/50 p-3 rounded">
                  <div className="text-sm text-gray-400">Date Column</div>
                  <div className={`text-lg font-bold ${validationResults.hasDateColumn ? 'text-green-400' : 'text-yellow-400'}`}>
                    {validationResults.hasDateColumn ? '✅ Found' : '⚠️ Not Found'}
                  </div>
                </div>
                <div className="bg-gray-800/50 p-3 rounded">
                  <div className="text-sm text-gray-400">Estimated Rows</div>
                  <div className="text-lg font-bold text-white">{validationResults.estimatedRows.toLocaleString()}</div>
                </div>
                <div className="bg-gray-800/50 p-3 rounded">
                  <div className="text-sm text-gray-400">File Size</div>
                  <div className="text-lg font-bold text-white">{validationResults.fileSize}</div>
                </div>
              </div>
            )}

            {previewData && (
              <div>
                <h3 className="text-lg font-bold text-white mb-2">📄 File Preview (First 10 lines):</h3>
                <div className="bg-gray-900 p-4 rounded-lg overflow-x-auto">
                  <pre className="text-sm text-gray-300 whitespace-pre-wrap">
                    {previewData.join('\n')}
                  </pre>
                </div>
              </div>
            )}
          </Card>
        )}

        {/* Upload Button */}
        <Card className="mb-6">
          <button
            onClick={uploadFile}
            disabled={!file || !datasetName.trim() || uploading}
            className="w-full px-6 py-4 bg-gradient-to-r from-green-600 to-green-700 hover:from-green-700 hover:to-green-800 text-white font-bold text-lg rounded-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-3"
          >
            {uploading && <LoadingSpinner size="sm" />}
            <span className="text-2xl">📤</span>
            <span>
              {uploading ? 'Processing and Uploading...' : 'Upload Data'}
            </span>
          </button>
        </Card>

        {/* Processing Results */}
        {processingResults && (
          <Card className="mb-6">
            <h2 className="text-2xl font-bold text-green-400 mb-4">✅ Upload Complete</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="bg-green-900/30 p-4 rounded">
                <div className="text-sm text-green-300">Dataset Name</div>
                <div className="text-lg font-bold text-white">{processingResults.dataset_name}</div>
              </div>
              <div className="bg-blue-900/30 p-4 rounded">
                <div className="text-sm text-blue-300">Rows Processed</div>
                <div className="text-lg font-bold text-white">{processingResults.rows_processed?.toLocaleString()}</div>
              </div>
              <div className="bg-purple-900/30 p-4 rounded">
                <div className="text-sm text-purple-300">File Size</div>
                <div className="text-lg font-bold text-white">{processingResults.file_size}</div>
              </div>
              <div className="bg-yellow-900/30 p-4 rounded">
                <div className="text-sm text-yellow-300">Upload Time</div>
                <div className="text-lg font-bold text-white">{processingResults.upload_time}</div>
              </div>
            </div>
          </Card>
        )}

        {/* Current Datasets - Matching Streamlit sidebar */}
        <Card>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-2xl font-bold text-cyan-400">📋 Current Datasets</h2>
            <button
              onClick={loadDatasets}
              className="px-3 py-1 bg-gray-700 hover:bg-gray-600 text-white rounded transition-colors"
              title="Refresh dataset list"
            >
              🔄
            </button>
          </div>
          
          {datasets.length > 0 ? (
            <div className="space-y-3">
              {datasets.map((dataset, index) => (
                <div key={index} className="bg-gray-800/50 p-4 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <div className="font-bold text-white">{dataset.name}</div>
                    <div className="text-sm text-gray-400">{dataset.rows?.toLocaleString()} rows</div>
                  </div>
                  <div className="text-sm text-gray-400">
                    {dataset.start_date} to {dataset.end_date}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    Updated: {dataset.updated_at}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-gray-400">
              <div className="text-4xl mb-2">📊</div>
              <p>No datasets uploaded yet</p>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
};

export default DataUpload;