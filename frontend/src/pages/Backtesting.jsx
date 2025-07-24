/**
 * Backtesting page - Test trading strategies against historical data
 */

import { useState } from 'react';
import LoadingSpinner from '../components/common/LoadingSpinner';

export default function Backtesting() {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold cyber-text mb-4">Backtesting</h1>
        <p className="text-gray-300">
          Test your trading strategies against historical market data
        </p>
      </div>

      {/* Strategy Configuration */}
      <div className="cyber-bg cyber-border rounded-lg p-6">
        <h2 className="text-xl font-semibold cyber-blue mb-6">Strategy Configuration</h2>
        <div className="text-center text-gray-400 py-8">
          <p>Backtesting functionality will be available here</p>
          <p className="text-sm mt-2">Configure your strategy parameters and run historical tests</p>
        </div>
      </div>

      {/* Results Display */}
      <div className="cyber-bg cyber-border rounded-lg p-6">
        <h2 className="text-xl font-semibold cyber-purple mb-6">Backtest Results</h2>
        <div className="text-center text-gray-400 py-8">
          <p>Backtest performance metrics and charts will be displayed here</p>
        </div>
      </div>
    </div>
  );
}