/**
 * Live Trading page - Real-time predictions and market data
 */

import { useState, useEffect, useRef } from 'react';
import { createWebSocket } from '../services/api';
import LoadingSpinner from '../components/common/LoadingSpinner';

export default function LiveTrading() {
  const [connected, setConnected] = useState(false);
  const [predictions, setPredictions] = useState(null);
  const [marketData, setMarketData] = useState(null);
  const [error, setError] = useState(null);
  const [connecting, setConnecting] = useState(false);
  
  const predictionsWs = useRef(null);
  const marketDataWs = useRef(null);

  const connectWebSockets = () => {
    setConnecting(true);
    setError(null);

    // Connect to live predictions WebSocket
    predictionsWs.current = createWebSocket(
      'live-predictions',
      (data) => {
        if (data.type === 'predictions') {
          setPredictions(data.data);
        } else if (data.type === 'status') {
          setConnected(data.connected);
        }
      },
      (error) => {
        setError('Failed to connect to predictions stream');
        setConnected(false);
      },
      () => {
        setConnected(false);
        setConnecting(false);
      }
    );

    // Connect to market data WebSocket
    marketDataWs.current = createWebSocket(
      'market-data',
      (data) => {
        if (data.type === 'market_data') {
          setMarketData(data.data);
        }
      },
      (error) => {
        console.warn('Market data connection failed:', error);
      }
    );

    // Send ping every 30 seconds to keep connection alive
    const pingInterval = setInterval(() => {
      if (predictionsWs.current?.readyState === WebSocket.OPEN) {
        predictionsWs.current.send(JSON.stringify({ type: 'ping' }));
      }
    }, 30000);

    setTimeout(() => {
      setConnected(true);
      setConnecting(false);
    }, 2000);

    return () => clearInterval(pingInterval);
  };

  const disconnectWebSockets = () => {
    if (predictionsWs.current) {
      predictionsWs.current.close();
      predictionsWs.current = null;
    }
    if (marketDataWs.current) {
      marketDataWs.current.close();
      marketDataWs.current = null;
    }
    setConnected(false);
    setPredictions(null);
    setMarketData(null);
  };

  useEffect(() => {
    return () => {
      disconnectWebSockets();
    };
  }, []);

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold cyber-text mb-4">Live Trading</h1>
          <p className="text-gray-300">
            Real-time market data and ML-powered predictions
          </p>
        </div>
        
        {/* Connection Controls */}
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <div
              className={`w-3 h-3 rounded-full ${
                connected ? 'bg-cyber-green animate-pulse' : 'bg-cyber-red'
              }`}
            />
            <span className="text-sm text-gray-400">
              {connected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
          
          {!connected ? (
            <button
              onClick={connectWebSockets}
              disabled={connecting}
              className="cyber-border rounded-md py-2 px-4 text-cyber-green hover:cyber-glow transition-all duration-200 disabled:opacity-50"
            >
              {connecting ? 'Connecting...' : 'Connect'}
            </button>
          ) : (
            <button
              onClick={disconnectWebSockets}
              className="border border-cyber-red rounded-md py-2 px-4 text-cyber-red hover:bg-cyber-red hover:bg-opacity-10 transition-all duration-200"
            >
              Disconnect
            </button>
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