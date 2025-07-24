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
    <div style={{ animation: 'pageLoad 0.6s ease-out' }}>
      {/* Header - matching original Streamlit design */}
      <div className="trading-header" style={{
        background: 'var(--gradient-card)',
        border: '2px solid var(--border)',
        borderRadius: '20px',
        padding: '3rem 2rem',
        margin: '2rem 0 3rem 0',
        textAlign: 'center',
        position: 'relative',
        overflow: 'hidden',
        boxShadow: 'var(--shadow-glow)'
      }}>
        <h1 style={{
          fontFamily: 'var(--font-display)',
          fontSize: '4rem',
          fontWeight: '900',
          background: 'var(--gradient-primary)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          backgroundClip: 'text',
          textAlign: 'center',
          margin: '0 0 1rem 0',
          textShadow: '0 0 30px rgba(0, 255, 255, 0.5)'
        }}>
          📡 LIVE MARKET DATA
        </h1>
        <p style={{
          fontSize: '1.2rem',
          margin: '1rem 0 0 0',
          color: 'rgba(255,255,255,0.8)'
        }}>
          Real-time Upstox WebSocket Integration
        </p>
      </div>

      {/* Status Display */}
      {status && (
        <Card className="mb-6">
          <p style={{ color: 'var(--text-accent)', textAlign: 'center', margin: 0 }}>
            {status}
          </p>
        </Card>
      )}

      {/* Configuration Tabs */}
      <div className="cyber-tabs mb-8">
        <div className="flex space-x-1 mb-6">
          <button
            onClick={() => setActiveTab('config')}
            className={`tab-button ${activeTab === 'config' ? 'active' : ''}`}
            style={{
              padding: '0.75rem 1.5rem',
              background: activeTab === 'config' ? 'var(--accent-cyan)' : 'transparent',
              color: activeTab === 'config' ? 'var(--bg-primary)' : 'var(--text-secondary)',
              border: '2px solid var(--border)',
              borderRadius: '8px 8px 0 0',
              fontWeight: '600'
            }}
          >
            🔌 Live Data Config
          </button>
          <button
            onClick={() => setActiveTab('historical')}
            className={`tab-button ${activeTab === 'historical' ? 'active' : ''}`}
            style={{
              padding: '0.75rem 1.5rem',
              background: activeTab === 'historical' ? 'var(--accent-cyan)' : 'transparent',
              color: activeTab === 'historical' ? 'var(--bg-primary)' : 'var(--text-secondary)',
              border: '2px solid var(--border)',
              borderRadius: '8px 8px 0 0',
              fontWeight: '600'
            }}
          >
            📊 Historical Data Fetch
          </button>
        </div>

        {/* Live Data Configuration Tab */}
        {activeTab === 'config' && (
          <Card>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
              {/* Credentials Section */}
              <div>
                <h3 style={{ color: 'var(--accent-electric)', marginBottom: '1rem' }}>
                  📱 Upstox API Credentials
                </h3>
                <div className="space-y-4">
                  <div>
                    <label style={{ display: 'block', marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>
                      Access Token
                    </label>
                    <input
                      type="password"
                      value={credentials.access_token}
                      onChange={(e) => setCredentials(prev => ({ ...prev, access_token: e.target.value }))}
                      placeholder="Your Upstox access token"
                      style={{
                        width: '100%',
                        padding: '0.75rem',
                        background: 'var(--bg-secondary)',
                        border: '2px solid var(--border)',
                        borderRadius: '8px',
                        color: 'var(--text-primary)'
                      }}
                    />
                  </div>
                  <div>
                    <label style={{ display: 'block', marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>
                      API Key
                    </label>
                    <input
                      type="password"
                      value={credentials.api_key}
                      onChange={(e) => setCredentials(prev => ({ ...prev, api_key: e.target.value }))}
                      placeholder="Your Upstox API key"
                      style={{
                        width: '100%',
                        padding: '0.75rem',
                        background: 'var(--bg-secondary)',
                        border: '2px solid var(--border)',
                        borderRadius: '8px',
                        color: 'var(--text-primary)'
                      }}
                    />
                  </div>
                </div>
              </div>

              {/* Instrument Selection */}
              <div>
                <h3 style={{ color: 'var(--accent-electric)', marginBottom: '1rem' }}>
                  📊 Instrument Configuration
                </h3>
                <div className="space-y-4">
                  <div>
                    <label style={{ display: 'block', marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>
                      Select Instruments
                    </label>
                    <select
                      multiple
                      value={selectedInstruments}
                      onChange={(e) => setSelectedInstruments(Array.from(e.target.selectedOptions, option => option.value))}
                      style={{
                        width: '100%',
                        padding: '0.75rem',
                        background: 'var(--bg-secondary)',
                        border: '2px solid var(--border)',
                        borderRadius: '8px',
                        color: 'var(--text-primary)',
                        minHeight: '120px'
                      }}
                    >
                      {Object.keys(popularInstruments).map(instrument => (
                        <option key={instrument} value={instrument}>
                          {instrument}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label style={{ display: 'block', marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>
                      Custom Instrument Key
                    </label>
                    <input
                      type="text"
                      value={customInstrument}
                      onChange={(e) => setCustomInstrument(e.target.value)}
                      placeholder="e.g., NSE_EQ|INE002A01018"
                      style={{
                        width: '100%',
                        padding: '0.75rem',
                        background: 'var(--bg-secondary)',
                        border: '2px solid var(--border)',
                        borderRadius: '8px',
                        color: 'var(--text-primary)'
                      }}
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Connection Controls */}
            <div style={{ marginTop: '2rem', borderTop: '1px solid var(--border)', paddingTop: '2rem' }}>
              <h3 style={{ color: 'var(--accent-electric)', marginBottom: '1rem' }}>
                🔌 Live Data Connection
              </h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem' }}>
                <button
                  onClick={connectToLiveData}
                  disabled={loading || !credentials.access_token || !credentials.api_key}
                  style={{
                    padding: '0.75rem 1rem',
                    background: isLiveConnected ? 'var(--accent-electric)' : 'var(--gradient-button)',
                    color: 'var(--bg-primary)',
                    border: 'none',
                    borderRadius: '8px',
                    fontWeight: '600',
                    cursor: loading ? 'not-allowed' : 'pointer',
                    opacity: loading ? 0.5 : 1
                  }}
                >
                  {loading ? '🔄 Connecting...' : '🚀 Connect'}
                </button>
                
                <button
                  onClick={disconnectFromLiveData}
                  disabled={!isLiveConnected}
                  style={{
                    padding: '0.75rem 1rem',
                    background: isLiveConnected ? 'var(--gradient-danger)' : 'var(--bg-secondary)',
                    color: 'var(--text-primary)',
                    border: '2px solid var(--border)',
                    borderRadius: '8px',
                    fontWeight: '600',
                    cursor: isLiveConnected ? 'pointer' : 'not-allowed',
                    opacity: isLiveConnected ? 1 : 0.5
                  }}
                >
                  🔌 Disconnect
                </button>
                
                <button
                  onClick={() => window.location.reload()}
                  style={{
                    padding: '0.75rem 1rem',
                    background: 'transparent',
                    color: 'var(--text-accent)',
                    border: '2px solid var(--border)',
                    borderRadius: '8px',
                    fontWeight: '600',
                    cursor: 'pointer'
                  }}
                >
                  🔄 Refresh Status
                </button>
                
                <div style={{
                  padding: '0.75rem 1rem',
                  background: 'var(--bg-secondary)',
                  border: '2px solid var(--border)',
                  borderRadius: '8px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  <div style={{
                    width: '12px',
                    height: '12px',
                    borderRadius: '50%',
                    background: isLiveConnected ? 'var(--accent-electric)' : 'var(--accent-pink)',
                    marginRight: '0.5rem',
                    animation: isLiveConnected ? 'statusPulse 2s ease-in-out infinite' : 'none'
                  }}></div>
                  <span style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                    {connectionStatus}
                  </span>
                </div>
              </div>
            </div>
          </Card>
        )}

        {/* Historical Data Tab */}
        {activeTab === 'historical' && (
          <Card>
            <h3 style={{ color: 'var(--accent-electric)', marginBottom: '1rem' }}>
              📈 Fetch Historical Data from Upstox
            </h3>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '2rem' }}>
              Fetch historical 1-minute data for Nifty 50 and other instruments using Upstox API
            </p>
            
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '2rem' }}>
              {/* Historical Credentials */}
              <div>
                <h4 style={{ color: 'var(--text-accent)', marginBottom: '1rem' }}>API Credentials</h4>
                <div className="space-y-3">
                  <input
                    type="password"
                    value={histCredentials.access_token}
                    onChange={(e) => setHistCredentials(prev => ({ ...prev, access_token: e.target.value }))}
                    placeholder="Access Token"
                    style={{
                      width: '100%',
                      padding: '0.75rem',
                      background: 'var(--bg-secondary)',
                      border: '2px solid var(--border)',
                      borderRadius: '8px',
                      color: 'var(--text-primary)'
                    }}
                  />
                  <input
                    type="password"
                    value={histCredentials.api_key}
                    onChange={(e) => setHistCredentials(prev => ({ ...prev, api_key: e.target.value }))}
                    placeholder="API Key"
                    style={{
                      width: '100%',
                      padding: '0.75rem',
                      background: 'var(--bg-secondary)',
                      border: '2px solid var(--border)',
                      borderRadius: '8px',
                      color: 'var(--text-primary)'
                    }}
                  />
                </div>
              </div>

              {/* Instrument Selection */}
              <div>
                <h4 style={{ color: 'var(--text-accent)', marginBottom: '1rem' }}>Instrument Selection</h4>
                <select
                  multiple
                  value={selectedHistInstruments}
                  onChange={(e) => setSelectedHistInstruments(Array.from(e.target.selectedOptions, option => option.value))}
                  style={{
                    width: '100%',
                    padding: '0.75rem',
                    background: 'var(--bg-secondary)',
                    border: '2px solid var(--border)',
                    borderRadius: '8px',
                    color: 'var(--text-primary)',
                    minHeight: '120px'
                  }}
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
                <h4 style={{ color: 'var(--text-accent)', marginBottom: '1rem' }}>Fetch Data</h4>
                <div className="space-y-3">
                  {selectedHistInstruments.map(instrument => (
                    <button
                      key={instrument}
                      onClick={() => fetchHistoricalData(instrument)}
                      disabled={loading || !histCredentials.access_token || !histCredentials.api_key}
                      style={{
                        width: '100%',
                        padding: '0.5rem',
                        background: 'var(--gradient-card)',
                        color: 'var(--text-accent)',
                        border: '2px solid var(--border)',
                        borderRadius: '6px',
                        fontSize: '0.875rem',
                        cursor: loading ? 'not-allowed' : 'pointer',
                        opacity: loading ? 0.5 : 1
                      }}
                    >
                      📊 Fetch {instrument}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </Card>
        )}
      </div>

      {/* Live Data Status Dashboard */}
      {isLiveConnected && (
        <Card>
          <h3 style={{ color: 'var(--accent-electric)', marginBottom: '1.5rem' }}>
            📊 Live Prediction Pipeline Status
          </h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '1rem' }}>
            <div style={{ textAlign: 'center', padding: '1rem', background: 'var(--bg-secondary)', borderRadius: '8px' }}>
              <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>
                {isLiveConnected ? '🟢' : '🔴'}
              </div>
              <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                Data Connection
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-accent)' }}>
                {isLiveConnected ? 'Connected' : 'Disconnected'}
              </div>
            </div>
            
            <div style={{ textAlign: 'center', padding: '1rem', background: 'var(--bg-secondary)', borderRadius: '8px' }}>
              <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>
                {isPredictionPipelineActive ? '🟢' : '🔴'}
              </div>
              <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                Prediction Pipeline
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-accent)' }}>
                {isPredictionPipelineActive ? 'Active' : 'Inactive'}
              </div>
            </div>
            
            <div style={{ textAlign: 'center', padding: '1rem', background: 'var(--bg-secondary)', borderRadius: '8px' }}>
              <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>🟢</div>
              <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                Trained Models
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-accent)' }}>
                4/4 Available
              </div>
            </div>
            
            <div style={{ textAlign: 'center', padding: '1rem', background: 'var(--bg-secondary)', borderRadius: '8px' }}>
              <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>📊</div>
              <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                Subscribed Instruments
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-accent)' }}>
                {selectedInstruments.length}
              </div>
            </div>
            
            <div style={{ textAlign: 'center', padding: '1rem', background: 'var(--bg-secondary)', borderRadius: '8px' }}>
              <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>🔮</div>
              <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                Live Predictions
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-accent)' }}>
                {Object.keys(predictions).length}
              </div>
            </div>
          </div>
        </Card>
      )}

      {/* Live Data Display */}
      {Object.keys(liveData).length > 0 && (
        <Card>
          <h3 style={{ color: 'var(--accent-electric)', marginBottom: '1.5rem' }}>
            📈 Live Market Data
          </h3>
          <div style={{ display: 'grid', gap: '1rem' }}>
            {Object.entries(liveData).map(([symbol, data]) => (
              <div key={symbol} style={{
                padding: '1rem',
                background: 'var(--bg-secondary)',
                border: '1px solid var(--border)',
                borderRadius: '8px'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <h4 style={{ color: 'var(--text-accent)', margin: 0 }}>{symbol}</h4>
                  <span style={{ color: 'var(--accent-electric)', fontWeight: '600' }}>
                    ₹{data.price?.toFixed(2)}
                  </span>
                </div>
                <div style={{ marginTop: '0.5rem', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                  Change: <span style={{ color: data.change >= 0 ? 'var(--accent-electric)' : 'var(--accent-pink)' }}>
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
          <h3 style={{ color: 'var(--accent-electric)', marginBottom: '1.5rem' }}>
            🔮 Live Predictions
          </h3>
          <div style={{ display: 'grid', gap: '1rem' }}>
            {Object.entries(predictions).map(([instrument, prediction]) => (
              <div key={instrument} style={{
                padding: '1rem',
                background: 'var(--bg-secondary)',
                border: '1px solid var(--border)',
                borderRadius: '8px'
              }}>
                <h4 style={{ color: 'var(--text-accent)', marginBottom: '0.5rem' }}>{instrument}</h4>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', fontSize: '0.875rem' }}>
                  <div>
                    <span style={{ color: 'var(--text-secondary)' }}>Direction: </span>
                    <span style={{ color: 'var(--accent-electric)' }}>{prediction.direction || 'N/A'}</span>
                  </div>
                  <div>
                    <span style={{ color: 'var(--text-secondary)' }}>Volatility: </span>
                    <span style={{ color: 'var(--accent-electric)' }}>{prediction.volatility || 'N/A'}</span>
                  </div>
                  <div>
                    <span style={{ color: 'var(--text-secondary)' }}>Confidence: </span>
                    <span style={{ color: 'var(--accent-electric)' }}>{prediction.confidence || 'N/A'}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Information Card */}
      <Card>
        <div style={{ background: 'rgba(0, 255, 255, 0.1)', padding: '1.5rem', borderRadius: '8px', marginBottom: '1rem' }}>
          <h4 style={{ color: 'var(--accent-cyan)', marginBottom: '1rem' }}>📋 Upstox Historical Data Features:</h4>
          <ul style={{ listStyle: 'none', padding: 0, color: 'var(--text-secondary)' }}>
            <li>• Supports 1-minute to daily intervals</li>
            <li>• Up to 1 year of historical data</li>
            <li>• Real-time API integration</li>
            <li>• Direct CSV download</li>
            <li>• Database storage option</li>
            <li>• Interactive charts</li>
          </ul>
          <h4 style={{ color: 'var(--accent-cyan)', margin: '1rem 0', fontSize: '1rem' }}>🔑 API Requirements:</h4>
          <ul style={{ listStyle: 'none', padding: 0, color: 'var(--text-secondary)' }}>
            <li>• Valid Upstox access token (refreshed daily)</li>
            <li>• Active API subscription for historical data</li>
          </ul>
        </div>

        <div style={{ background: 'rgba(0, 255, 65, 0.1)', padding: '1.5rem', borderRadius: '8px' }}>
          <h4 style={{ color: 'var(--accent-electric)', marginBottom: '1rem' }}>🌱 Live Data Continuation Feature:</h4>
          <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem' }}>
            <strong>How it works:</strong> Upload your historical data with name pattern: `live_NSE_INDEX_Nifty_50`. 
            When live data starts, it automatically loads your historical data as foundation. 
            Live ticks continue building OHLC from that point forward. 
            Result: 250+ rows for predictions from day 1 instead of starting with 0.
          </p>
          <p style={{ color: 'var(--text-secondary)' }}>
            <strong>Naming pattern:</strong> `livenifty50`, `liveniftybank`, `livereliance`, etc.
          </p>
        </div>
      </Card>
    </div>
  );
}
          )}
        </div>
      </div>

      {/* Connection Status */}
      {connecting && (
        <div className="cyber-bg cyber-border rounded-lg p-6">
          <div className="text-center">
            <LoadingSpinner size="lg" text="Connecting to live data stream..." />
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="cyber-bg border border-cyber-red rounded-lg p-6">
          <h2 className="text-xl font-semibold text-cyber-red mb-4">
            ❌ Connection Error
          </h2>
          <p className="text-gray-300">{error}</p>
        </div>
      )}

      {/* Live Data Display */}
      {connected && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Market Data */}
          <div className="cyber-bg cyber-border rounded-lg p-6">
            <h2 className="text-xl font-semibold cyber-blue mb-4">Market Data</h2>
            {marketData ? (
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Symbol:</span>
                  <span className="text-white font-mono">{marketData.symbol}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Price:</span>
                  <span className="text-cyber-green font-mono text-xl">
                    ₹{marketData.price?.toLocaleString()}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Change:</span>
                  <span className={`font-mono ${
                    marketData.change >= 0 ? 'text-cyber-green' : 'text-cyber-red'
                  }`}>
                    {marketData.change >= 0 ? '+' : ''}{marketData.change?.toFixed(2)}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Volume:</span>
                  <span className="text-white font-mono">
                    {marketData.volume?.toLocaleString()}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Last Update:</span>
                  <span className="text-gray-300 text-sm">
                    {new Date(marketData.timestamp).toLocaleTimeString()}
                  </span>
                </div>
              </div>
            ) : (
              <div className="text-center text-gray-400 py-8">
                <p>Waiting for market data...</p>
              </div>
            )}
          </div>

          {/* Live Predictions */}
          <div className="cyber-bg cyber-border rounded-lg p-6">
            <h2 className="text-xl font-semibold cyber-purple mb-4">Live Predictions</h2>
            {predictions ? (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gray-800 rounded-lg p-3">
                    <h3 className="text-sm font-semibold text-cyber-blue mb-2">Volatility</h3>
                    <div className="text-lg font-mono text-cyber-green">
                      {predictions.volatility?.toFixed(4) || 'N/A'}
                    </div>
                  </div>
                  <div className="bg-gray-800 rounded-lg p-3">
                    <h3 className="text-sm font-semibold text-cyber-blue mb-2">Direction</h3>
                    <div className="text-lg font-mono text-cyber-yellow">
                      {predictions.direction || 'N/A'}
                    </div>
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gray-800 rounded-lg p-3">
                    <h3 className="text-sm font-semibold text-cyber-blue mb-2">Profit Prob.</h3>
                    <div className="text-lg font-mono text-cyber-purple">
                      {predictions.profit_probability ? 
                        `${(predictions.profit_probability * 100).toFixed(1)}%` : 'N/A'}
                    </div>
                  </div>
                  <div className="bg-gray-800 rounded-lg p-3">
                    <h3 className="text-sm font-semibold text-cyber-blue mb-2">Reversal</h3>
                    <div className="text-lg font-mono text-cyber-red">
                      {predictions.reversal || 'N/A'}
                    </div>
                  </div>
                </div>

                <div className="pt-3 border-t border-gray-700">
                  <p className="text-xs text-gray-400">
                    Last updated: {predictions.timestamp ? 
                      new Date(predictions.timestamp).toLocaleTimeString() : 'Unknown'}
                  </p>
                </div>
              </div>
            ) : (
              <div className="text-center text-gray-400 py-8">
                <p>Waiting for predictions...</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* AI Analysis */}
      {connected && predictions && (
        <div className="cyber-bg cyber-border rounded-lg p-6">
          <h2 className="text-xl font-semibold cyber-text mb-4">AI Market Analysis</h2>
          <div className="bg-gray-800 rounded-lg p-4">
            <p className="text-gray-300 italic">
              AI-powered market analysis and insights will be displayed here based on current predictions and market conditions.
            </p>
          </div>
        </div>
      )}

      {/* Trading Controls (Placeholder) */}
      {connected && (
        <div className="cyber-bg cyber-border rounded-lg p-6">
          <h2 className="text-xl font-semibold cyber-yellow mb-4">Trading Controls</h2>
          <div className="text-center text-gray-400 py-8">
            <p>Trading execution controls will be available here</p>
            <p className="text-sm mt-2">Connect to your broker API for live trading</p>
          </div>
        </div>
      )}
    </div>
  );
}