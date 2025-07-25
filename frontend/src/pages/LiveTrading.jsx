/**
 * Live Data page - Real-time Upstox WebSocket Integration
 * Exact replica of original Streamlit functionality
 */

import { useState, useEffect, useRef } from 'react';
import Card from '../components/common/Card';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { createWebSocket } from '../services/api';
import axios from 'axios';

export default function LiveTrading() {
  // State management - matching original Streamlit functionality
  const [isLiveConnected, setIsLiveConnected] = useState(false);
  const [isPredictionPipelineActive, setIsPredictionPipelineActive] = useState(false);
  const [activeTab, setActiveTab] = useState('config');
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');
  
  // Credentials state
  const [credentials, setCredentials] = useState({
    access_token: '',
    api_key: ''
  });
  
  // Historical data credentials
  const [histCredentials, setHistCredentials] = useState({
    access_token: '',
    api_key: ''
  });
  
  // Instrument selection state
  const [selectedInstruments, setSelectedInstruments] = useState(['NIFTY 50', 'BANK NIFTY']);
  const [selectedHistInstruments, setSelectedHistInstruments] = useState(['NIFTY 50']);
  const [customInstrument, setCustomInstrument] = useState('');
  
  // Live data state
  const [liveData, setLiveData] = useState({});
  const [predictions, setPredictions] = useState({});
  const [connectionStatus, setConnectionStatus] = useState('Disconnected');
  
  // WebSocket refs
  const wsRef = useRef(null);
  const predictionsWs = useRef(null);

  // Popular instruments mapping (from original Streamlit)
  const popularInstruments = {
    "NIFTY 50": "NSE_INDEX|Nifty 50",
    "BANK NIFTY": "NSE_INDEX|Nifty Bank",
    "NIFTY IT": "NSE_INDEX|Nifty IT", 
    "NIFTY FMCG": "NSE_INDEX|Nifty FMCG",
    "RELIANCE": "NSE_EQ|INE002A01018",
    "TCS": "NSE_EQ|INE467B01029",
    "HDFC BANK": "NSE_EQ|INE040A01034",
    "INFOSYS": "NSE_EQ|INE009A01021"
  };

  // Historical data instruments
  const histInstruments = {
    "NIFTY 50": "NSE_INDEX|Nifty 50",
    "BANK NIFTY": "NSE_INDEX|Nifty Bank", 
    "NIFTY IT": "NSE_INDEX|Nifty IT",
    "NIFTY FMCG": "NSE_INDEX|Nifty FMCG",
    "RELIANCE": "NSE_EQ|INE002A01018",
    "TCS": "NSE_EQ|INE467B01029",
    "HDFC BANK": "NSE_EQ|INE040A01034",
    "INFOSYS": "NSE_EQ|INE009A01021"
  };

  // Interval options for historical data
  const intervalOptions = {
    "1 minute": "1minute",
    "3 minute": "3minute", 
    "5 minute": "5minute",
    "15 minute": "15minute",
    "30 minute": "30minute",
    "1 hour": "1hour",
    "1 day": "day"
  };

  // Connect to live data feed
  const connectToLiveData = async () => {
    if (!credentials.access_token || !credentials.api_key) {
      setStatus('❌ Please provide both Access Token and API Key');
      return;
    }

    try {
      setLoading(true);
      setStatus('🔌 Connecting to Upstox WebSocket...');

      const instrumentKeys = selectedInstruments.map(inst => 
        popularInstruments[inst] || inst
      );

      const response = await axios.post('/api/live-data/connect', {
        access_token: credentials.access_token,
        api_key: credentials.api_key,
        instruments: instrumentKeys
      });

      if (response.data.success) {
        setIsLiveConnected(true);
        setConnectionStatus('Connected');
        setStatus('✅ Connected successfully to Upstox WebSocket!');
        
        // Start prediction pipeline
        const pipelineResponse = await axios.post('/api/live-data/start-predictions');
        
        if (pipelineResponse.data.success) {
          setIsPredictionPipelineActive(true);
          setStatus('✅ Live prediction pipeline started!');
        }
        
      } else {
        setStatus(`❌ Connection failed: ${response.data.message}`);
      }
    } catch (error) {
      setStatus(`❌ Connection error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Disconnect from live data
  const disconnectFromLiveData = async () => {
    try {
      await axios.post('/api/live-data/disconnect');
      await axios.post('/api/live-data/stop-predictions');
      
      setIsLiveConnected(false);
      setIsPredictionPipelineActive(false);
      setConnectionStatus('Disconnected');
      setStatus('🔌 Disconnected from live data feed');
      setLiveData({});
      setPredictions({});
    } catch (error) {
      setStatus(`❌ Disconnect error: ${error.message}`);
    }
  };

  // Fetch historical data
  const fetchHistoricalData = async (instrument, interval = '5minute', days = 30) => {
    if (!histCredentials.access_token || !histCredentials.api_key) {
      setStatus('❌ Please provide historical data credentials');
      return;
    }

    try {
      setLoading(true);
      setStatus(`📊 Fetching ${days} days of ${interval} data for ${instrument}...`);

      const response = await axios.post('/api/live-data/fetch-historical', {
        access_token: histCredentials.access_token,
        api_key: histCredentials.api_key,
        instruments: [histInstruments[instrument] || instrument]
      });

      if (response.data.success) {
        setStatus(`✅ Successfully fetched ${response.data.records_saved} records!`);
      } else {
        setStatus(`❌ Historical fetch failed: ${response.data.message}`);
      }
    } catch (error) {
      setStatus(`❌ Historical fetch error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // WebSocket connection for real-time updates
  useEffect(() => {
    if (isLiveConnected) {
      // Connect to live data WebSocket for real-time updates
      wsRef.current = createWebSocket(
        'live-data',
        (data) => {
          if (data.type === 'market_data') {
            setLiveData(prev => ({
              ...prev,
              [data.data.symbol]: data.data
            }));
          } else if (data.type === 'predictions') {
            setPredictions(prev => ({
              ...prev,
              [data.data.instrument]: data.data
            }));
          }
        },
        (error) => {
          console.error('WebSocket error:', error);
          setStatus('❌ WebSocket connection lost');
        },
        () => {
          setConnectionStatus('Disconnected');
          setIsLiveConnected(false);
        }
      );
    }

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [isLiveConnected]);

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold cyber-text mb-4">📡 Live Data</h1>
          <p className="text-gray-300">
            Real-time Upstox WebSocket Integration with Live Predictions
          </p>
        </div>
        
        {/* Connection Status */}
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <div
              className={`w-3 h-3 rounded-full ${
                isLiveConnected ? 'bg-cyber-green animate-pulse' : 'bg-cyber-red'
              }`}
            />
            <span className="text-sm text-gray-400">
              {connectionStatus}
            </span>
          </div>
        </div>
      </div>

      {/* Status Display */}
      {status && (
        <Card>
          <p className="text-center text-cyber-green">{status}</p>
        </Card>
      )}

      {/* Configuration Tabs */}
      <div className="cyber-bg cyber-border rounded-lg overflow-hidden">
        <div className="flex border-b border-gray-700">
          <button
            onClick={() => setActiveTab('config')}
            className={`flex-1 px-6 py-3 text-sm font-medium transition-colors ${
              activeTab === 'config'
                ? 'bg-cyber-blue text-black'
                : 'text-gray-400 hover:text-white'
            }`}
          >
            🔌 Live Data Config
          </button>
          <button
            onClick={() => setActiveTab('historical')}
            className={`flex-1 px-6 py-3 text-sm font-medium transition-colors ${
              activeTab === 'historical'
                ? 'bg-cyber-blue text-black'
                : 'text-gray-400 hover:text-white'
            }`}
          >
            📊 Historical Data Fetch
          </button>
        </div>

        <div className="p-6">
          {/* Live Data Configuration Tab */}
          {activeTab === 'config' && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Credentials Section */}
                <div>
                  <h3 className="text-xl font-semibold cyber-blue mb-4">
                    📱 Upstox API Credentials
                  </h3>
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-400 mb-2">
                        Access Token
                      </label>
                      <input
                        type="password"
                        value={credentials.access_token}
                        onChange={(e) => setCredentials(prev => ({ ...prev, access_token: e.target.value }))}
                        placeholder="Your Upstox access token"
                        className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-cyber-blue"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-400 mb-2">
                        API Key
                      </label>
                      <input
                        type="password"
                        value={credentials.api_key}
                        onChange={(e) => setCredentials(prev => ({ ...prev, api_key: e.target.value }))}
                        placeholder="Your Upstox API key"
                        className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-cyber-blue"
                      />
                    </div>
                  </div>
                </div>

                {/* Instrument Selection */}
                <div>
                  <h3 className="text-xl font-semibold cyber-blue mb-4">
                    📊 Instrument Configuration
                  </h3>
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-400 mb-2">
                        Select Instruments
                      </label>
                      <select
                        multiple
                        value={selectedInstruments}
                        onChange={(e) => setSelectedInstruments(Array.from(e.target.selectedOptions, option => option.value))}
                        className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-cyber-blue h-32"
                      >
                        {Object.keys(popularInstruments).map(instrument => (
                          <option key={instrument} value={instrument}>
                            {instrument}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-400 mb-2">
                        Custom Instrument Key
                      </label>
                      <input
                        type="text"
                        value={customInstrument}
                        onChange={(e) => setCustomInstrument(e.target.value)}
                        placeholder="e.g., NSE_EQ|INE002A01018"
                        className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-cyber-blue"
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Connection Controls */}
              <div className="border-t border-gray-700 pt-6">
                <h3 className="text-xl font-semibold cyber-blue mb-4">
                  🔌 Live Data Connection
                </h3>
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                  <button
                    onClick={connectToLiveData}
                    disabled={loading || !credentials.access_token || !credentials.api_key}
                    className="cyber-border rounded-md py-2 px-4 text-cyber-green hover:cyber-glow transition-all duration-200 disabled:opacity-50"
                  >
                    {loading ? '🔄 Connecting...' : '🚀 Connect'}
                  </button>
                  
                  <button
                    onClick={disconnectFromLiveData}
                    disabled={!isLiveConnected}
                    className="border border-cyber-red rounded-md py-2 px-4 text-cyber-red hover:bg-cyber-red hover:bg-opacity-10 transition-all duration-200 disabled:opacity-50"
                  >
                    🔌 Disconnect
                  </button>
                  
                  <button
                    onClick={() => window.location.reload()}
                    className="border border-gray-600 rounded-md py-2 px-4 text-gray-400 hover:text-white transition-all duration-200"
                  >
                    🔄 Refresh Status
                  </button>
                  
                  <div className="flex items-center justify-center space-x-2">
                    <div className={`w-3 h-3 rounded-full ${isLiveConnected ? 'bg-cyber-green animate-pulse' : 'bg-cyber-red'}`}></div>
                    <span className="text-sm text-gray-400">{connectionStatus}</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Historical Data Tab */}
          {activeTab === 'historical' && (
            <div className="space-y-6">
              <h3 className="text-xl font-semibold cyber-blue">
                📈 Fetch Historical Data from Upstox
              </h3>
              <p className="text-gray-400">
                Fetch historical 1-minute data for Nifty 50 and other instruments using Upstox API
              </p>
              
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Historical Credentials */}
                <div>
                  <h4 className="font-semibold text-cyber-green mb-3">API Credentials</h4>
                  <div className="space-y-3">
                    <input
                      type="password"
                      value={histCredentials.access_token}
                      onChange={(e) => setHistCredentials(prev => ({ ...prev, access_token: e.target.value }))}
                      placeholder="Access Token"
                      className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-cyber-blue"
                    />
                    <input
                      type="password"
                      value={histCredentials.api_key}
                      onChange={(e) => setHistCredentials(prev => ({ ...prev, api_key: e.target.value }))}
                      placeholder="API Key"
                      className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-cyber-blue"
                    />
                  </div>
                </div>

                {/* Instrument Selection */}
                <div>
                  <h4 className="font-semibold text-cyber-green mb-3">Instrument Selection</h4>
                  <select
                    multiple
                    value={selectedHistInstruments}
                    onChange={(e) => setSelectedHistInstruments(Array.from(e.target.selectedOptions, option => option.value))}
                    className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-cyber-blue h-32"
                  >
                    {Object.keys(histInstruments).map(instrument => (
                      <option key={instrument} value={instrument}>
                        {instrument}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Fetch Controls */}
                <div>
                  <h4 className="font-semibold text-cyber-green mb-3">Fetch Data</h4>
                  <div className="space-y-2">
                    {selectedHistInstruments.map(instrument => (
                      <button
                        key={instrument}
                        onClick={() => fetchHistoricalData(instrument)}
                        disabled={loading || !histCredentials.access_token || !histCredentials.api_key}
                        className="w-full text-left px-3 py-2 bg-gray-800 border border-gray-600 rounded-md text-cyber-blue hover:bg-gray-700 transition-all duration-200 disabled:opacity-50 text-sm"
                      >
                        📊 Fetch {instrument}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Live Data Status Dashboard */}
      {isLiveConnected && (
        <Card>
          <h3 className="text-xl font-semibold cyber-blue mb-4">
            📊 Live Prediction Pipeline Status
          </h3>
          <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
            <div className="text-center p-4 bg-gray-800 rounded-lg">
              <div className="text-2xl mb-2">{isLiveConnected ? '🟢' : '🔴'}</div>
              <div className="text-sm text-gray-400">Data Connection</div>
              <div className="text-xs text-cyber-green">{isLiveConnected ? 'Connected' : 'Disconnected'}</div>
            </div>
            
            <div className="text-center p-4 bg-gray-800 rounded-lg">
              <div className="text-2xl mb-2">{isPredictionPipelineActive ? '🟢' : '🔴'}</div>
              <div className="text-sm text-gray-400">Prediction Pipeline</div>
              <div className="text-xs text-cyber-green">{isPredictionPipelineActive ? 'Active' : 'Inactive'}</div>
            </div>
            
            <div className="text-center p-4 bg-gray-800 rounded-lg">
              <div className="text-2xl mb-2">🟢</div>
              <div className="text-sm text-gray-400">Trained Models</div>
              <div className="text-xs text-cyber-green">4/4 Available</div>
            </div>
            
            <div className="text-center p-4 bg-gray-800 rounded-lg">
              <div className="text-2xl mb-2">📊</div>
              <div className="text-sm text-gray-400">Subscribed Instruments</div>
              <div className="text-xs text-cyber-green">{selectedInstruments.length}</div>
            </div>
            
            <div className="text-center p-4 bg-gray-800 rounded-lg">
              <div className="text-2xl mb-2">🔮</div>
              <div className="text-sm text-gray-400">Live Predictions</div>
              <div className="text-xs text-cyber-green">{Object.keys(predictions).length}</div>
            </div>
          </div>
        </Card>
      )}

      {/* Live Data Display */}
      {Object.keys(liveData).length > 0 && (
        <Card>
          <h3 className="text-xl font-semibold cyber-blue mb-4">📈 Live Market Data</h3>
          <div className="space-y-4">
            {Object.entries(liveData).map(([symbol, data]) => (
              <div key={symbol} className="p-4 bg-gray-800 border border-gray-700 rounded-lg">
                <div className="flex justify-between items-center">
                  <h4 className="text-cyber-green font-semibold">{symbol}</h4>
                  <span className="text-cyber-blue font-mono text-xl">
                    ₹{data.price?.toFixed(2)}
                  </span>
                </div>
                <div className="mt-2 text-sm text-gray-400">
                  Change: <span className={data.change >= 0 ? 'text-cyber-green' : 'text-cyber-red'}>
                    {data.change?.toFixed(2)}
                  </span> | Volume: {data.volume?.toLocaleString()}
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Live Predictions Display */}
      {Object.keys(predictions).length > 0 && (
        <Card>
          <h3 className="text-xl font-semibold cyber-blue mb-4">🔮 Live Predictions</h3>
          <div className="space-y-4">
            {Object.entries(predictions).map(([instrument, prediction]) => (
              <div key={instrument} className="p-4 bg-gray-800 border border-gray-700 rounded-lg">
                <h4 className="text-cyber-green font-semibold mb-2">{instrument}</h4>
                <div className="grid grid-cols-3 gap-4 text-sm">
                  <div>
                    <span className="text-gray-400">Direction: </span>
                    <span className="text-cyber-blue">{prediction.direction || 'N/A'}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Volatility: </span>
                    <span className="text-cyber-blue">{prediction.volatility || 'N/A'}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Confidence: </span>
                    <span className="text-cyber-blue">{prediction.confidence || 'N/A'}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Information Card */}
      <Card>
        <div className="space-y-4">
          <div className="bg-blue-900 bg-opacity-30 p-4 rounded-lg">
            <h4 className="text-cyber-blue font-semibold mb-2">📋 Upstox Historical Data Features:</h4>
            <ul className="text-sm text-gray-300 space-y-1">
              <li>• Supports 1-minute to daily intervals</li>
              <li>• Up to 1 year of historical data</li>
              <li>• Real-time API integration</li>
              <li>• Direct CSV download</li>
              <li>• Database storage option</li>
              <li>• Interactive charts</li>
            </ul>
            <h5 className="text-cyber-blue font-semibold mt-3 mb-1">🔑 API Requirements:</h5>
            <ul className="text-sm text-gray-300 space-y-1">
              <li>• Valid Upstox access token (refreshed daily)</li>
              <li>• Active API subscription for historical data</li>
            </ul>
          </div>

          <div className="bg-green-900 bg-opacity-30 p-4 rounded-lg">
            <h4 className="text-cyber-green font-semibold mb-2">🌱 Live Data Continuation Feature:</h4>
            <p className="text-sm text-gray-300 mb-2">
              <strong>How it works:</strong> Upload your historical data with name pattern: `live_NSE_INDEX_Nifty_50`. 
              When live data starts, it automatically loads your historical data as foundation. 
              Live ticks continue building OHLC from that point forward. 
              Result: 250+ rows for predictions from day 1 instead of starting with 0.
            </p>
            <p className="text-sm text-gray-300">
              <strong>Naming pattern:</strong> `livenifty50`, `liveniftybank`, `livereliance`, etc.
            </p>
          </div>
        </div>
      </Card>
    </div>
  );
}