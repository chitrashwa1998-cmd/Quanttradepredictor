import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import { API_BASE_URL } from '../config/api';

// Helper function to get model-specific colors
const getModelColor = (modelName) => {
  const colors = {
    'direction': '#00ff88',
    'magnitude': '#ff6b6b', 
    'profit_prob': '#4ecdc4',
    'volatility': '#ffe66d',
    'trend_sideways': '#a8e6cf',
    'reversal': '#ff8b94',
    'trading_signal': '#b4b7ff'
  };
  return colors[modelName] || '#00ffff';
};

const Predictions = () => {
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [period, setPeriod] = useState('30d');

  useEffect(() => {
    const loadModels = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/models/status`);
        const data = await response.json();
        console.log('Models response:', data); // Debug log
        const models = Object.keys(data.data?.trained_models || {});
        setAvailableModels(models);
        if (models.length > 0 && !selectedModel) {
          setSelectedModel(models[0]);
        }
      } catch (error) {
        console.error('Error fetching models:', error);
      }
    };
    loadModels();
  }, []);

  useEffect(() => {
    const loadPredictions = async () => {
      if (!selectedModel) return;

      setLoading(true);
      try {
        const timestamp = new Date().getTime();
        const response = await axios.get(`${API_BASE_URL}/predictions/${selectedModel}?period=${period}&t=${timestamp}`);
        const data = response.data;
        console.log(`Predictions for ${selectedModel}:`, data); // Debug log

        // Validate response structure
        if (!data.success) {
          throw new Error(data.error || 'Failed to fetch predictions');
        }

        // Log some prediction statistics for debugging
        if (data.predictions && data.predictions.length > 0) {
          const upCount = data.predictions.filter(p => p.prediction === 1).length;
          const downCount = data.predictions.filter(p => p.prediction === 0).length;
          console.log(`${selectedModel} - Up: ${upCount}, Down: ${downCount}, Total: ${data.predictions.length}`);

          // Log first few predictions to see differences
          console.log(`${selectedModel} first 5 predictions:`, data.predictions.slice(0, 5));

          // Log prediction distribution
          const predDistribution = data.predictions.reduce((acc, pred) => {
            acc[pred.prediction] = (acc[pred.prediction] || 0) + 1;
            return acc;
          }, {});
          console.log(`${selectedModel} prediction distribution:`, predDistribution);

          // Validate data quality
          const invalidDates = data.predictions.filter(p => !p.date || p.date === 'undefined').length;
          const invalidPrices = data.predictions.filter(p => !p.price || isNaN(p.price)).length;
          const invalidPredictions = data.predictions.filter(p => p.prediction === undefined || p.prediction === null).length;

          if (invalidDates > 0) console.warn(`${selectedModel}: ${invalidDates} invalid dates found`);
          if (invalidPrices > 0) console.warn(`${selectedModel}: ${invalidPrices} invalid prices found`);
          if (invalidPredictions > 0) console.warn(`${selectedModel}: ${invalidPredictions} invalid predictions found`);

          // Add model-specific styling to differentiate charts
          data.modelColor = getModelColor(selectedModel);
          data.modelName = selectedModel;
        } else {
          console.warn(`${selectedModel}: No predictions data received`);
        }

        setPredictions(data);
      } catch (error) {
        console.error('Error fetching predictions:', error);
        setPredictions(null);
        // Show user-friendly error message
        alert(`Failed to load predictions for ${selectedModel}: ${error.message}`);
      } finally {
        setLoading(false);
      }
    };

    loadPredictions();
  }, [selectedModel, period]);

  if (availableModels.length === 0) {
    return (
      <div className="container">
        <div className="header">
          <h1>🎯 Prediction Engine</h1>
          <p>Real-time Market Analysis & Forecasting</p>
        </div>

        <div className="alert alert-warning">
          ⚠️ No trained models found. Please go to the <strong>Model Training</strong> page to train models first.
        </div>

        <div className="card">
          <h3>📝 Steps to Get Started:</h3>
          <ol style={{color: '#b8bcc8', lineHeight: 1.8}}>
            <li>Upload trading data on the <strong>Data Upload</strong> page</li>
            <li>Train models on the <strong>Model Training</strong> page</li>
            <li>Return here to view predictions</li>
          </ol>
        </div>
      </div>
    );
  }

  const createPredictionChart = () => {
    if (!predictions || !predictions.predictions || predictions.predictions.length === 0) {
      console.log('No prediction data available for chart');
      return null;
    }

    const data = predictions.predictions;
    console.log(`Creating chart for ${selectedModel} with ${data.length} data points`);

    // Enhanced date formatting with IST timezone handling
    const formatDate = (dateStr) => {
      try {
        // Handle various date formats
        if (!dateStr || dateStr === 'undefined' || dateStr === 'null') {
          return new Date().toISOString();
        }

        let date;

        // If it's already a valid datetime string (YYYY-MM-DD HH:MM:SS)
        if (typeof dateStr === 'string' && dateStr.match(/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/)) {
          // Parse as IST time and convert to UTC for chart
          date = new Date(dateStr + ' +05:30'); // Add IST offset
          return date.toISOString();
        }

        // If it's already a valid ISO string
        if (typeof dateStr === 'string' && dateStr.includes('T')) {
          return new Date(dateStr).toISOString();
        }

        // If it's a numeric index (old format)
        if (!isNaN(dateStr)) {
          const index = parseInt(dateStr);
          // Create sequential 5-minute intervals from market start
          const baseDate = new Date();
          baseDate.setHours(9, 15, 0, 0); // 9:15 AM IST market start
          // Convert to UTC by subtracting IST offset (5.5 hours)
          baseDate.setHours(baseDate.getHours() - 5);
          baseDate.setMinutes(baseDate.getMinutes() - 30);

          date = new Date(baseDate.getTime() + (index * 5 * 60 * 1000)); // Add 5-minute intervals
          return date.toISOString();
        }

        // Fallback parsing
        date = new Date(dateStr);
        if (isNaN(date.getTime())) {
          // Create sequential dates based on position in array
          const now = new Date();
          const dataIndex = data.findIndex(d => d.date === dateStr);
          const minutesBack = (data.length - dataIndex) * 5; // 5-minute intervals
          return new Date(now.getTime() - (minutesBack * 60 * 1000)).toISOString();
        }

        return date.toISOString();
      } catch (e) {
        console.warn('Date formatting error:', e, 'for date:', dateStr);
        return new Date().toISOString();
      }
    };

    // Validate and clean data
    const validData = data.filter(d => d && d.price !== undefined && d.prediction !== undefined);

    if (validData.length === 0) {
      console.log('No valid data points found');
      return null;
    }

    // Price line
    const priceTrace = {
      x: validData.map(d => formatDate(d.date)),
      y: validData.map(d => parseFloat(d.price) || 0),
      type: 'scatter',
      mode: 'lines',
      name: 'Price',
      line: { color: predictions.modelColor || '#2E86C1', width: 2 },
      hovertemplate: 'Price: $%{y:.2f}<br>Date: %{x}<extra></extra>'
    };

    // Prediction markers
    const upPredictions = validData.filter(d => d.prediction === 1);
    const downPredictions = validData.filter(d => d.prediction === 0);

    const traces = [priceTrace];

    if (upPredictions.length > 0) {
      traces.push({
        x: upPredictions.map(d => formatDate(d.date)),
        y: upPredictions.map(d => parseFloat(d.price) || 0),
        type: 'scatter',
        mode: 'markers',
        name: 'Predicted Up',
        marker: { symbol: 'triangle-up', color: '#00ff41', size: 8 },
        hovertemplate: 'UP Signal<br>Price: $%{y:.2f}<br>Confidence: %{customdata:.3f}<extra></extra>',
        customdata: upPredictions.map(d => d.confidence || 0.5)
      });
    }

    if (downPredictions.length > 0) {
      traces.push({
        x: downPredictions.map(d => formatDate(d.date)),
        y: downPredictions.map(d => parseFloat(d.price) || 0),
        type: 'scatter',
        mode: 'markers',
        name: 'Predicted Down',
        marker: { symbol: 'triangle-down', color: '#ff0080', size: 8 },
        hovertemplate: 'DOWN Signal<br>Price: $%{y:.2f}<br>Confidence: %{customdata:.3f}<extra></extra>',
        customdata: downPredictions.map(d => d.confidence || 0.5)
      });
    }

    return {
      data: traces,
      layout: {
        title: {
          text: `${selectedModel.replace('_', ' ').toUpperCase()} Predictions (${validData.length} points)`,
          font: { color: '#ffffff', size: 18 }
        },
        xaxis: { 
          title: 'Date & Time', 
          color: '#ffffff',
          tickangle: -45,
          showgrid: true,
          gridcolor: 'rgba(255,255,255,0.1)'
        },
        yaxis: { 
          title: 'Price ($)', 
          color: '#ffffff',
          showgrid: true,
          gridcolor: 'rgba(255,255,255,0.1)'
        },
        height: 500,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#ffffff' },
        legend: { 
          bgcolor: 'rgba(0,0,0,0.5)',
          bordercolor: 'rgba(255,255,255,0.2)',
          borderwidth: 1
        },
        hovermode: 'closest'
      }
    };
  };

  const chartData = createPredictionChart();

  return (
    <div className="container">
      <div className="header">
        <h1>🎯 Prediction Engine</h1>
        <p>Real-time Market Analysis & Forecasting</p>
      </div>

      <div className="card">
        <div className="grid grid-3" style={{marginBottom: '2rem'}}>
          <div>
            <label style={{display: 'block', marginBottom: '0.5rem', color: '#00ffff'}}>
              🤖 Select AI Model
            </label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              style={{
                width: '100%',
                padding: '0.5rem',
                background: 'rgba(25, 25, 45, 0.9)',
                border: '1px solid rgba(0, 255, 255, 0.3)',
                borderRadius: '4px',
                color: '#ffffff'
              }}
            >
              {availableModels.map(model => (
                <option key={model} value={model}>
                  {model.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label style={{display: 'block', marginBottom: '0.5rem', color: '#00ffff'}}>
              📅 Time Period
            </label>
            <select
              value={period}
              onChange={(e) => setPeriod(e.target.value)}
              style={{
                width: '100%',
                padding: '0.5rem',
                background: 'rgba(25, 25, 45, 0.9)',
                border: '1px solid rgba(0, 255, 255, 0.3)',
                borderRadius: '4px',
                color: '#ffffff'
              }}
            >
              <option value="30d">Last 30 days</option>
              <option value="90d">Last 90 days</option>
              <option value="all">All data</option>
            </select>
          </div>

          <div className="metric-card">
            <div className="metric-label">Model Type</div>
            <div className="metric-value" style={{fontSize: '1rem'}}>
              {availableModels.length > 0 && selectedModel ? 
                (['magnitude', 'volatility'].includes(selectedModel) ? 'Regression' : 'Classification') : 
                'Unknown'
              }
            </div>
          </div>
        </div>

        {loading && <div className="loading"></div>}

        {chartData && !loading && (
          <Plot
            data={chartData.data}
            layout={chartData.layout}
            style={{width: '100%'}}
            config={{responsive: true}}
          />
        )}
      </div>

      {predictions && predictions.predictions && !loading && (
        <div className="card">
          <h3>📊 Prediction Statistics</h3>

          <div className="grid grid-4" style={{marginBottom: '2rem'}}>
            <div className="metric-card">
              <div className="metric-label">Total Predictions</div>
              <div className="metric-value">{predictions.total_predictions}</div>
            </div>

            <div className="metric-card">
              <div className="metric-label">Up Predictions</div>
              <div className="metric-value" style={{color: '#00ff41'}}>
                {predictions.predictions.filter(p => p.prediction === 1).length}
              </div>
            </div>

            <div className="metric-card">
              <div className="metric-label">Down Predictions</div>
              <div className="metric-value" style={{color: '#ff0080'}}>
                {predictions.predictions.filter(p => p.prediction === 0).length}
              </div>
            </div>

            <div className="metric-card">
              <div className="metric-label">Latest Signal</div>
              <div className="metric-value" style={{color: predictions.predictions[predictions.predictions.length - 1]?.prediction === 1 ? '#00ff41' : '#ff0080'}}>
                {predictions.predictions[predictions.predictions.length - 1]?.prediction === 1 ? 'UP' : 'DOWN'}
              </div>
            </div>
          </div>

          <h3>📋 Recent Predictions</h3>
          <div style={{overflowX: 'auto'}}>
            <table className="table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Price</th>
                  <th>Prediction</th>
                  <th>Confidence</th>
                </tr>
              </thead>
              <tbody>
                {predictions.predictions.slice(-20).reverse().map((pred, index) => {
                  const formatPredictionDate = (dateStr) => {
                    try {
                      // Handle various date formats
                      if (!dateStr || dateStr === 'undefined' || dateStr === 'null') {
                        return 'Invalid Date';
                      }

                      let date;

                      // If it's a number, treat as index - create proper date
                      if (!isNaN(dateStr)) {
                        const index = parseInt(dateStr);
                        // Assume 5-minute intervals for scalping
                        const baseDate = new Date();
                        baseDate.setHours(9, 15, 0, 0); // Market start time 9:15 AM
                        date = new Date(baseDate.getTime() + (index * 5 * 60 * 1000)); // Add 5-minute intervals
                      } else if (typeof dateStr === 'string') {
                        // If it's already a formatted date string
                        if (dateStr.includes('-') && dateStr.includes(':')) {
                          // Format: YYYY-MM-DD HH:MM:SS
                          date = new Date(dateStr);
                        } else {
                          date = new Date(dateStr);
                        }
                      } else {
                        date = new Date(dateStr);
                      }

                      if (isNaN(date.getTime())) {
                        return `Invalid: ${dateStr}`;
                      }

                      // Format to IST with proper market time context
                      const istTime = date.toLocaleString('en-IN', {
                        timeZone: 'Asia/Kolkata',
                        year: 'numeric',
                        month: '2-digit',
                        day: '2-digit',
                        hour: '2-digit',
                        minute: '2-digit',
                        hour12: false
                      });

                      // Add market session indicator
                      const hour = date.getHours();
                      const minute = date.getMinutes();
                      const timeInMinutes = hour * 60 + minute;
                      const marketStart = 9 * 60 + 15; // 9:15 AM
                      const marketEnd = 15 * 60 + 30;  // 3:30 PM

                      let sessionIndicator = '';
                      if (timeInMinutes >= marketStart && timeInMinutes <= marketEnd) {
                        sessionIndicator = ' 🟢';
                      } else {
                        sessionIndicator = ' 🔴';
                      }

                      return istTime + ' IST' + sessionIndicator;
                    } catch (e) {
                      console.warn('Date formatting error:', e, 'for date:', dateStr);
                      return `Error: ${dateStr}`;
                    }
                  };

                  return (
                    <tr key={index}>
                      <td>{formatPredictionDate(pred.date)}</td>
                      <td>${(pred.price || 0).toFixed(2)}</td>
                      <td style={{color: pred.prediction === 1 ? '#00ff41' : '#ff0080'}}>
                        {pred.prediction === 1 ? '📈 UP' : '📉 DOWN'}
                      </td>
                      <td>{pred.confidence ? pred.confidence.toFixed(3) : 'N/A'}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default Predictions;