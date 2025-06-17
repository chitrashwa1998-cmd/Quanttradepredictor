
import React from 'react';

const Backtesting = () => {
  return (
    <div className="container">
      <div className="header">
        <h1>📈 BACKTEST ENGINE</h1>
        <p>Strategy Performance Analysis</p>
      </div>

      <div className="card">
        <h3>🚧 Under Development</h3>
        <p style={{color: '#b8bcc8', lineHeight: 1.8}}>
          The backtesting engine is currently being developed for the React frontend. 
          This will include:
        </p>
        <ul style={{color: '#b8bcc8', lineHeight: 1.8, marginTop: '1rem'}}>
          <li>Strategy configuration and parameter tuning</li>
          <li>Historical performance analysis</li>
          <li>Risk metrics calculation</li>
          <li>Trade history visualization</li>
          <li>Performance comparison charts</li>
        </ul>
      </div>
    </div>
  );
};

export default Backtesting;
