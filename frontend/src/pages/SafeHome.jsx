/**
 * Safe fallback Home page - minimal implementation to avoid crashes
 */

import { useState, useEffect } from 'react';
import Card from '../components/common/Card';

const SafeHome = () => {
  const [status, setStatus] = useState('loading');

  useEffect(() => {
    // Simple status check without complex API calls
    const checkStatus = async () => {
      try {
        const response = await fetch('/api/health');
        if (response.ok) {
          setStatus('connected');
        } else {
          setStatus('error');
        }
      } catch (error) {
        setStatus('error');
      }
    };

    checkStatus();
  }, []);

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-4xl font-bold cyber-text mb-4">
          🚀 TribexAlpha Trading Platform
        </h1>
        <p className="text-xl text-gray-300">
          Advanced Quantitative Trading with AI-Powered Market Analysis
        </p>
      </div>

      {/* Status Card */}
      <Card>
        <div className="text-center p-6">
          <div className="text-6xl mb-4">
            {status === 'loading' && '⏳'}
            {status === 'connected' && '✅'}
            {status === 'error' && '❌'}
          </div>
          <h2 className="text-2xl font-bold cyber-blue mb-2">
            System Status
          </h2>
          <p className="text-gray-300">
            {status === 'loading' && 'Connecting to backend...'}
            {status === 'connected' && 'All systems operational'}
            {status === 'error' && 'Backend connection error'}
          </p>
        </div>
      </Card>

      {/* Quick Navigation */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <div className="text-center p-4">
            <div className="text-3xl mb-2">📊</div>
            <h3 className="font-semibold cyber-blue">Dashboard</h3>
            <p className="text-sm text-gray-400">System overview</p>
          </div>
        </Card>

        <Card>
          <div className="text-center p-4">
            <div className="text-3xl mb-2">📁</div>
            <h3 className="font-semibold cyber-blue">Data Upload</h3>
            <p className="text-sm text-gray-400">Import CSV data</p>
          </div>
        </Card>

        <Card>
          <div className="text-center p-4">
            <div className="text-3xl mb-2">🤖</div>
            <h3 className="font-semibold cyber-blue">Model Training</h3>
            <p className="text-sm text-gray-400">Train ML models</p>
          </div>
        </Card>

        <Card>
          <div className="text-center p-4">
            <div className="text-3xl mb-2">📡</div>
            <h3 className="font-semibold cyber-blue">Live Data</h3>
            <p className="text-sm text-gray-400">Real-time trading</p>
          </div>
        </Card>
      </div>

      {/* Key Features */}
      <Card>
        <h2 className="text-2xl font-bold cyber-blue mb-4">🎯 Platform Features</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h3 className="text-lg font-semibold text-cyber-green mb-2">
              🔮 AI-Powered Predictions
            </h3>
            <ul className="text-gray-300 space-y-1 text-sm">
              <li>• Volatility forecasting with XGBoost</li>
              <li>• Direction prediction models</li>
              <li>• Profit probability analysis</li>
              <li>• Reversal detection algorithms</li>
            </ul>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-cyber-green mb-2">
              📈 Real-Time Trading
            </h3>
            <ul className="text-gray-300 space-y-1 text-sm">
              <li>• Live Upstox WebSocket integration</li>
              <li>• Real-time market data processing</li>
              <li>• Automated prediction pipeline</li>
              <li>• Historical data fetching</li>
            </ul>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default SafeHome;