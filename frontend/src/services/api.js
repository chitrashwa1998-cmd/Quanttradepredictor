/**
 * API service for TribexAlpha backend communication
 * Handles all HTTP requests to FastAPI backend
 */

import axios from 'axios';

// For Replit environment, use proxy configuration
const getAPIBaseURL = () => {
  // Use empty string since routes already include /api prefix
  return '';
};

const API_BASE_URL = getAPIBaseURL();

// Create axios instance with default config
const axiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 second timeout
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
axiosInstance.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
axiosInstance.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error('API Response Error:', error);
    if (error.response?.status === 401) {
      // Handle unauthorized access
      console.warn('Unauthorized access - redirecting to login');
    }
    return Promise.reject(error);
  }
);

const api = axiosInstance;

// Health check
export const healthCheck = async () => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    throw new Error(`Health check failed: ${error.message}`);
  }
};

// Predictions API
export const predictionsAPI = {
  // Make single prediction
  predict: async (modelName, data) => {
    const response = await api.post('/api/predictions/predict', {
      model_name: modelName,
      data: data
    });
    return response.data;
  },

  // Get model status
  getModelsStatus: async () => {
    const response = await api.get('/api/predictions/models/status');
    return response.data;
  },

  // Get live predictions
  getLivePredictions: async () => {
    const response = await api.get('/api/predictions/live');
    return response.data;
  },

  // Batch predictions
  batchPredict: async (requests) => {
    const response = await api.post('/api/predictions/batch', requests);
    return response.data;
  }
};

// Models API
export const modelsAPI = {
  // Train model
  trainModel: async (modelName, datasetName = null, parameters = null) => {
    const response = await api.post('/api/models/train', {
      model_name: modelName,
      dataset_name: datasetName,
      parameters: parameters
    });
    return response.data;
  },

  // Upload training data
  uploadData: async (file, datasetName = null) => {
    const formData = new FormData();
    formData.append('file', file);
    if (datasetName) {
      formData.append('dataset_name', datasetName);
    }

    const response = await api.post('/api/models/upload-data', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  // List all models
  listModels: async () => {
    const response = await api.get('/api/models/list');
    return response.data;
  },

  // Delete model
  deleteModel: async (modelName) => {
    const response = await api.delete(`/api/models/${modelName}`);
    return response.data;
  },

  // Get model info
  getModelInfo: async (modelName) => {
    const response = await api.get(`/api/models/${modelName}/info`);
    return response.data;
  }
};

// Data API
export const dataAPI = {
  // Upload data
  uploadData: async (formData) => {
    const response = await api.post('/api/data/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response;
  },

  // List datasets
  listDatasets: async () => {
    const response = await api.get('/api/data/datasets');
    return response.data;
  },

  // Get dataset
  getDataset: async (datasetName, limit = null, offset = 0) => {
    const params = { offset };
    if (limit) params.limit = limit;

    const response = await api.get(`/api/data/datasets/${datasetName}`, { params });
    return response.data;
  },

  // Delete dataset
  deleteDataset: async (datasetName) => {
    const response = await api.delete(`/api/data/datasets/${datasetName}`);
    return response;
  },

  // Rename dataset
  renameDataset: async (oldName, newName) => {
    const response = await api.post(`/api/data/datasets/${oldName}/rename`, { new_name: newName });
    return response;
  },

  // Load dataset with options
  loadDataset: async (datasetName, options = {}) => {
    const params = new URLSearchParams();
    if (options.limit) params.append('limit', options.limit);
    if (options.start_date) params.append('start_date', options.start_date);
    if (options.end_date) params.append('end_date', options.end_date);

    const response = await api.get(`/api/data/datasets/${datasetName}/load?${params}`);
    return response;
  },

  // Export dataset
  exportDataset: async (datasetName) => {
    const response = await api.get(`/api/data/datasets/${datasetName}/export`, {
      responseType: 'blob'
    });
    return response;
  },

  // Rename dataset
  renameDataset: async (oldName, newName) => {
    const response = await api.post(`/api/data/datasets/${oldName}/rename?new_name=${newName}`);
    return response;
  },

  // Delete model results
  deleteModelResults: async (modelName) => {
    const response = await api.delete(`/api/data/model-results/${modelName}`);
    return response;
  },

  // Delete predictions
  deletePredictions: async (modelName) => {
    const response = await api.delete(`/api/data/predictions/${modelName}`);
    return response;
  },

  // Sync metadata
  syncMetadata: async () => {
    const response = await api.post('/api/data/sync-metadata');
    return response;
  },

  // Clean database
  cleanDatabase: async () => {
    const response = await api.post('/api/data/clean-database');
    return response;
  },

  // Get key content
  getKeyContent: async (key) => {
    const response = await api.get(`/api/data/keys/${key}`);
    return response;
  },

  // Get database info
  getDatabaseInfo: async () => {
    const response = await api.get('/api/data/database/info');
    return response.data;
  },

  // Get datasets (alias for listDatasets for compatibility)
  getDatasets: async () => {
    const response = await api.get('/api/data/datasets');
    return response;
  },

  // Clear all data
  clearAllData: async () => {
    const response = await api.delete('/api/data/datasets');
    return response.data;
  },

  // Clean data mode
  cleanDataMode: async () => {
    const response = await api.post('/api/data/clean-mode');
    return response.data;
  },

  // Sync metadata
  syncMetadata: async () => {
    const response = await api.post('/api/data/sync-metadata');
    return response.data;
  },

  // Clean database
  cleanDatabase: async () => {
    const response = await api.post('/api/data/clean-database');
    return response.data;
  },

  // Get key content
  getKeyContent: async (key) => {
    const response = await api.get(`/api/data/keys/${key}`);
    return response;
  }
};

// WebSocket API
export const createWebSocket = (endpoint, onMessage, onError = null, onClose = null) => {
  const wsUrl = `${API_BASE_URL.replace('http', 'ws')}/api/ws/${endpoint}`;
  const ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    console.log(`WebSocket connected: ${endpoint}`);
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
    } catch (error) {
      console.error('WebSocket message parse error:', error);
    }
  };

  ws.onerror = (error) => {
    console.error(`WebSocket error on ${endpoint}:`, error);
    if (onError) onError(error);
  };

  ws.onclose = () => {
    console.log(`WebSocket closed: ${endpoint}`);
    if (onClose) onClose();
  };

  return ws;
};

export default api;