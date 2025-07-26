/**
 * Home page - Simplified working version
 */

import { useState, useEffect } from 'react';
import Card from '../components/common/Card';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { dataAPI, predictionsAPI } from '../services/api';

const Home = () => {
  const [databaseInfo, setDatabaseInfo] = useState(null);
  const [modelsStatus, setModelsStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        
        const [dbInfo, modelStatus] = await Promise.all([
          dataAPI.getDatabaseInfo().catch(() => ({ data: {} })),
          predictionsAPI.getModelsStatus().catch(() => ({ data: {} }))
        ]);
        
        setDatabaseInfo(dbInfo.data || {});
        setModelsStatus(modelStatus.data || {});
      } catch (err) {
        console.error('Home data fetch error:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-violet-800 flex items-center justify-center">
        <LoadingSpinner />
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-violet-800">
        <div className="container mx-auto px-4 py-8">
          <Card className="max-w-2xl mx-auto">
            <h3 className="text-2xl font-bold text-red-400 mb-4">Error Loading Dashboard</h3>
            <p className="text-gray-300">{error}</p>
          </Card>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-violet-800">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-6xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-purple-500 to-pink-500 mb-6">
            TribexAlpha
          </h1>
          <p className="text-2xl text-gray-300 mb-4">
            Advanced Quantitative Trading Platform
          </p>
          <p className="text-lg text-gray-400 max-w-3xl mx-auto">
            Leverage cutting-edge machine learning algorithms for volatility forecasting, 
            direction prediction, and comprehensive market analysis with real-time insights.
          </p>
        </div>

        {/* Database Stats */}
        <Card className="mb-8">
          <h2 className="text-3xl font-bold text-cyan-400 mb-6">📊 System Overview</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="bg-gradient-to-br from-blue-900/50 to-cyan-900/50 p-6 rounded-xl border border-cyan-500/30">
              <div className="text-3xl font-bold text-cyan-400 mb-2">
                {databaseInfo?.total_datasets || 0}
              </div>
              <div className="text-gray-300 text-lg">Datasets</div>
            </div>
            <div className="bg-gradient-to-br from-green-900/50 to-emerald-900/50 p-6 rounded-xl border border-green-500/30">
              <div className="text-3xl font-bold text-green-400 mb-2">
                {databaseInfo?.total_records?.toLocaleString() || 0}
              </div>
              <div className="text-gray-300 text-lg">Records</div>
            </div>
            <div className="bg-gradient-to-br from-purple-900/50 to-pink-900/50 p-6 rounded-xl border border-purple-500/30">
              <div className="text-3xl font-bold text-purple-400 mb-2">
                {databaseInfo?.total_trained_models || 0}
              </div>
              <div className="text-gray-300 text-lg">Trained Models</div>
            </div>
            <div className="bg-gradient-to-br from-yellow-900/50 to-orange-900/50 p-6 rounded-xl border border-yellow-500/30">
              <div className="text-3xl font-bold text-yellow-400 mb-2">
                {databaseInfo?.total_predictions || 0}
              </div>
              <div className="text-gray-300 text-lg">Predictions</div>
            </div>
          </div>
        </Card>

        {/* Model Status */}
        <Card className="mb-8">
          <h2 className="text-3xl font-bold text-purple-400 mb-6">🤖 Model Status</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {[
              { name: 'Volatility', key: 'volatility', icon: '📈', color: 'blue' },
              { name: 'Direction', key: 'direction', icon: '🎯', color: 'green' },
              { name: 'Profit Probability', key: 'profit_probability', icon: '💰', color: 'yellow' },
              { name: 'Reversal', key: 'reversal', icon: '🔄', color: 'purple' }
            ].map((model) => (
              <div key={model.key} className={`bg-gray-800/50 p-4 rounded-lg border border-${model.color}-500/30`}>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-lg">{model.icon}</span>
                  <span className={`px-2 py-1 rounded-full text-xs ${
                    modelsStatus?.[model.key]?.trained 
                      ? 'bg-green-600 text-white' 
                      : 'bg-red-600 text-white'
                  }`}>
                    {modelsStatus?.[model.key]?.trained ? 'Trained' : 'Not Trained'}
                  </span>
                </div>
                <div className={`text-${model.color}-400 font-medium`}>{model.name}</div>
              </div>
            ))}
          </div>
        </Card>

        {/* Features */}
        <Card className="mb-8">
          <h2 className="text-3xl font-bold text-cyan-400 mb-6">⚡ Platform Features</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div className="text-center p-6">
              <div className="text-4xl mb-4">📊</div>
              <h3 className="text-xl font-bold text-white mb-2">Data Management</h3>
              <p className="text-gray-400">Upload, validate, and manage your OHLC trading data with comprehensive preprocessing.</p>
            </div>
            <div className="text-center p-6">
              <div className="text-4xl mb-4">🤖</div>
              <h3 className="text-xl font-bold text-white mb-2">ML Training</h3>
              <p className="text-gray-400">Train advanced machine learning models using XGBoost, CatBoost, and ensemble methods.</p>
            </div>
            <div className="text-center p-6">
              <div className="text-4xl mb-4">🔮</div>
              <h3 className="text-xl font-bold text-white mb-2">Predictions</h3>
              <p className="text-gray-400">Generate real-time predictions with comprehensive statistical analysis and insights.</p>
            </div>
            <div className="text-center p-6">
              <div className="text-4xl mb-4">📈</div>
              <h3 className="text-xl font-bold text-white mb-2">Backtesting</h3>
              <p className="text-gray-400">Validate your strategies with historical data and performance metrics.</p>
            </div>
            <div className="text-center p-6">
              <div className="text-4xl mb-4">🎯</div>
              <h3 className="text-xl font-bold text-white mb-2">Live Trading</h3>
              <p className="text-gray-400">Connect to live market data feeds for real-time analysis and trading signals.</p>
            </div>
            <div className="text-center p-6">
              <div className="text-4xl mb-4">🗄️</div>
              <h3 className="text-xl font-bold text-white mb-2">Database</h3>
              <p className="text-gray-400">Robust PostgreSQL backend with optimized storage and retrieval systems.</p>
            </div>
          </div>
        </Card>

        {/* Quick Actions */}
        <Card>
          <h2 className="text-3xl font-bold text-green-400 mb-6">🚀 Quick Actions</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <a
              href="/data-upload"
              className="block p-6 bg-gradient-to-r from-blue-600 to-blue-700 rounded-lg hover:from-blue-700 hover:to-blue-800 transition-colors text-center"
            >
              <div className="text-3xl mb-2">📁</div>
              <div className="text-white font-bold">Upload Data</div>
            </a>
            <a
              href="/model-training"
              className="block p-6 bg-gradient-to-r from-purple-600 to-purple-700 rounded-lg hover:from-purple-700 hover:to-purple-800 transition-colors text-center"
            >
              <div className="text-3xl mb-2">🎯</div>
              <div className="text-white font-bold">Train Models</div>
            </a>
            <a
              href="/predictions"
              className="block p-6 bg-gradient-to-r from-green-600 to-green-700 rounded-lg hover:from-green-700 hover:to-green-800 transition-colors text-center"
            >
              <div className="text-3xl mb-2">🔮</div>
              <div className="text-white font-bold">Generate Predictions</div>
            </a>
            <a
              href="/database-manager"
              className="block p-6 bg-gradient-to-r from-red-600 to-red-700 rounded-lg hover:from-red-700 hover:to-red-800 transition-colors text-center"
            >
              <div className="text-3xl mb-2">🗄️</div>
              <div className="text-white font-bold">Manage Database</div>
            </a>
          </div>
        </Card>
      </div>
    </div>
  );
};

export default Home;