import React, { useState, useEffect } from 'react';
import axios from 'axios';

// Connect to the working Flask server on port 8080
const API_BASE_URL = 'http://localhost:8080/api';
import Plot from 'react-plotly.js';

const Predictions = () => {
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [period, setPeriod] = useState('30d');

  useEffect(() => {
    const loadModels = async () => {
      try {
        const response = await fetch('/api/models/status');
        const data = await response.json();
        const models = Object.keys(data.models || []);
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
        const response = await axios.get(`${API_BASE_URL}/predictions/${selectedModel}?period=${period}`);
        const data = response.data;
        setPredictions(data);
      } catch (error) {
        console.error('Error fetching predictions:', error);
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
          ⚠️ No trained models found. Please go to the <strong>Model Training</strong> page first.
        </div>
      </div>
    );
  }

  const createPredictionChart = () => {
    if (!predictions || !predictions.predictions) return null;

    const data = predictions.predictions;

    // Price line
    const priceTrace = {
      x: data.map(d => d.date),
      y: data.map(d => d.price),
      type: 'scatter',
      mode: 'lines',
      name: 'Price',
      line: { color: '#2E86C1', width: 2 }
    };

    // Prediction markers
    const upPredictions = data.filter(d => d.prediction === 1);
    const downPredictions = data.filter(d => d.prediction === 0);

    const traces = [priceTrace];

    if (upPredictions.length > 0) {
      traces.push({
        x: upPredictions.map(d => d.date),
        y: upPredictions.map(d => d.price),
        type: 'scatter',
        mode: 'markers',
        name: 'Predicted Up',
        marker: { symbol: 'triangle-up', color: 'green', size: 8 }
      });
    }

    if (downPredictions.length > 0) {
      traces.push({
        x: downPredictions.map(d => d.date),
        y: downPredictions.map(d => d.price),
        type: 'scatter',
        mode: 'markers',
        name: 'Predicted Down',
        marker: { symbol: 'triangle-down', color: 'red', size: 8 }
      });
    }

    return {
      data: traces,
      layout: {
        title: `${selectedModel.replace('_', ' ').toUpperCase()} Predictions`,
        xaxis: { title: 'Date', color: '#ffffff' },
        yaxis: { title: 'Price ($)', color: '#ffffff' },
        height: 500,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#ffffff' },
        legend: { bgcolor: 'rgba(0,0,0,0)' }
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
            <div className="metric-value" style={{fontSize: '1rem'}}>Classification</div>
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
                {predictions.predictions.slice(-20).reverse().map((pred, index) => (
                  <tr key={index}>
                    <td>{new Date(pred.date).toLocaleDateString()}</td>
                    <td>${pred.price.toFixed(2)}</td>
                    <td style={{color: pred.prediction === 1 ? '#00ff41' : '#ff0080'}}>
                      {pred.prediction === 1 ? '📈 UP' : '📉 DOWN'}
                    </td>
                    <td>{pred.confidence ? pred.confidence.toFixed(3) : 'N/A'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default Predictions;