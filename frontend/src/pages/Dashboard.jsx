/**
 * Dashboard page - Main overview of the trading platform
 */

import { useState, useEffect } from 'react';
import { dataAPI, predictionsAPI } from '../services/api';
import LoadingSpinner from '../components/common/LoadingSpinner';

export default function Dashboard() {
  const [loading, setLoading] = useState(true);
  const [dbInfo, setDbInfo] = useState(null);
  const [modelsStatus, setModelsStatus] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const [dbResponse, modelsResponse] = await Promise.all([
          dataAPI.getDatabaseInfo(),
          predictionsAPI.getModelsStatus()
        ]);
        
        setDbInfo(dbResponse.info);
        setModelsStatus(modelsResponse.status);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <LoadingSpinner size="lg" text="Loading dashboard..." />
      </div>
    );
  }

  if (error) {
    return (
      <div className="cyber-bg cyber-border rounded-lg p-6">
        <h2 className="text-xl font-bold text-cyber-red mb-4">Error</h2>
        <p className="text-gray-300">{error}</p>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-4xl font-bold cyber-text mb-4">
          TribexAlpha Trading Platform
        </h1>
        <p className="text-xl text-gray-300">
          Advanced quantitative trading with ML-powered predictions
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Database Stats */}
        <div className="cyber-bg cyber-border rounded-lg p-6">
          <h3 className="text-lg font-semibold text-cyber-blue mb-3">Database</h3>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-400">Datasets:</span>
              <span className="text-white font-mono">
                {dbInfo?.total_datasets || 0}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Records:</span>
              <span className="text-white font-mono">
                {dbInfo?.total_records?.toLocaleString() || 0}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Backend:</span>
              <span className="text-cyber-green text-sm">
                {dbInfo?.backend || 'Unknown'}
              </span>
            </div>
          </div>
        </div>

        {/* Models Stats */}
        <div className="cyber-bg cyber-border rounded-lg p-6">
          <h3 className="text-lg font-semibold text-cyber-purple mb-3">Models</h3>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-400">Total:</span>
              <span className="text-white font-mono">
                {modelsStatus?.models ? Object.keys(modelsStatus.models).length : 0}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Loaded:</span>
              <span className="text-cyber-green font-mono">
                {modelsStatus?.models ? 
                  Object.values(modelsStatus.models).filter(m => m.loaded).length : 0}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Status:</span>
              <span className={`text-sm ${modelsStatus?.initialized ? 'text-cyber-green' : 'text-cyber-red'}`}>
                {modelsStatus?.initialized ? 'Ready' : 'Initializing'}
              </span>
            </div>
          </div>
        </div>

        {/* System Health */}
        <div className="cyber-bg cyber-border rounded-lg p-6">
          <h3 className="text-lg font-semibold text-cyber-yellow mb-3">System</h3>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-400">Backend:</span>
              <span className="text-cyber-green text-sm">Online</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Database:</span>
              <span className="text-cyber-green text-sm">Connected</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Version:</span>
              <span className="text-white font-mono">2.0.0</span>
            </div>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="cyber-bg cyber-border rounded-lg p-6">
          <h3 className="text-lg font-semibold text-cyber-green mb-3">Quick Actions</h3>
          <div className="space-y-2">
            <button className="w-full text-left text-sm text-cyber-blue hover:text-white transition-colors">
              → Upload Data
            </button>
            <button className="w-full text-left text-sm text-cyber-blue hover:text-white transition-colors">
              → Train Models
            </button>
            <button className="w-full text-left text-sm text-cyber-blue hover:text-white transition-colors">
              → View Predictions
            </button>
            <button className="w-full text-left text-sm text-cyber-blue hover:text-white transition-colors">
              → Start Live Trading
            </button>
          </div>
        </div>
      </div>

      {/* Models Overview */}
      {modelsStatus?.models && (
        <div className="cyber-bg cyber-border rounded-lg p-6">
          <h2 className="text-2xl font-bold cyber-text mb-6">Models Overview</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {Object.entries(modelsStatus.models).map(([name, info]) => (
              <div key={name} className="bg-gray-800 rounded-lg p-4">
                <h3 className="font-semibold text-cyber-blue capitalize mb-2">
                  {name}
                </h3>
                <div className="space-y-1 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Status:</span>
                    <span className={info.loaded ? 'text-cyber-green' : 'text-cyber-red'}>
                      {info.loaded ? 'Loaded' : 'Not Loaded'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Features:</span>
                    <span className="text-white">{info.features?.length || 0}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recent Activity placeholder */}
      <div className="cyber-bg cyber-border rounded-lg p-6">
        <h2 className="text-2xl font-bold cyber-text mb-6">Recent Activity</h2>
        <div className="text-center text-gray-400 py-8">
          <p>Activity monitoring will be available once trading begins</p>
        </div>
      </div>
    </div>
  );
}