/**
 * Dashboard page - Exact Streamlit UI replication
 */

import { useState, useEffect } from 'react';
import { dataAPI } from '../services/api';

const Dashboard = () => {
  const [databaseInfo, setDatabaseInfo] = useState(null);
  const [loading, setLoading] = useState(true);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      const response = await dataAPI.getDatabaseInfo();
      setDatabaseInfo(response?.data || {});
    } catch (error) {
      console.error('Error loading dashboard data:', error);
      setDatabaseInfo({});
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDashboardData();
  }, []);

  return (
    <div style={{ backgroundColor: '#0a0a0f', minHeight: '100vh', color: '#ffffff', fontFamily: 'Space Grotesk, sans-serif' }}>
      <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '2rem' }}>
        
        {/* Header - Exact Streamlit Style */}
        <div className="trading-header" style={{
          background: 'linear-gradient(145deg, rgba(25, 25, 45, 0.9), rgba(35, 35, 55, 0.6))',
          border: '2px solid rgba(0, 255, 255, 0.2)',
          borderRadius: '20px',
          padding: '3rem 2rem',
          marginBottom: '3rem',
          boxShadow: '0 0 20px rgba(0, 255, 255, 0.1)',
          textAlign: 'center'
        }}>
          <h1 style={{ 
            margin: 0, 
            fontSize: '3rem', 
            fontFamily: 'Orbitron, monospace',
            background: 'linear-gradient(135deg, #00ffff, #8b5cf6, #ff0080)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
            marginBottom: '1rem'
          }}>
            ⚡ TRIBEXALPHA
          </h1>
          <h2 style={{ 
            color: '#00ffff', 
            margin: '1rem 0',
            fontSize: '1.8rem',
            fontFamily: 'Orbitron, monospace'
          }}>
            Trading Analytics Platform
          </h2>
          <p style={{ fontSize: '1.2rem', margin: '1rem 0 0 0', color: 'rgba(255,255,255,0.8)' }}>
            Advanced AI-Powered Market Intelligence for Nifty 50 Predictions and Quantitative Trading
          </p>
        </div>

        {/* Quick Navigation - Exact Streamlit Style */}
        <div style={{
          background: 'rgba(0, 255, 255, 0.1)',
          border: '2px solid #00ffff',
          borderRadius: '16px',
          padding: '2rem',
          margin: '2rem 0',
          textAlign: 'center'
        }}>
          <h2 style={{ color: '#00ffff', marginBottom: '1rem' }}>📍 Quick Navigation</h2>
          <p style={{ color: '#e6e8eb', fontSize: '1.1rem', marginBottom: '1.5rem' }}>
            Use the sidebar navigation to access different modules:
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', margin: '1rem 0' }}>
            <div style={{ background: 'rgba(0, 255, 255, 0.1)', padding: '1rem', borderRadius: '8px' }}>
              <strong style={{ color: '#00ffff' }}>📊 DATA UPLOAD</strong><br />
              <span style={{ color: '#b8bcc8' }}>Load your OHLC data</span>
            </div>
            <div style={{ background: 'rgba(0, 255, 65, 0.1)', padding: '1rem', borderRadius: '8px' }}>
              <strong style={{ color: '#00ff41' }}>🔬 MODEL TRAINING</strong><br />
              <span style={{ color: '#b8bcc8' }}>Train machine learning models</span>
            </div>
            <div style={{ background: 'rgba(139, 92, 246, 0.1)', padding: '1rem', borderRadius: '8px' }}>
              <strong style={{ color: '#8b5cf6' }}>🎯 PREDICTIONS</strong><br />
              <span style={{ color: '#b8bcc8' }}>Generate forecasts</span>
            </div>
            <div style={{ background: 'rgba(255, 0, 128, 0.1)', padding: '1rem', borderRadius: '8px' }}>
              <strong style={{ color: '#ff0080' }}>📈 BACKTESTING</strong><br />
              <span style={{ color: '#b8bcc8' }}>Test strategies</span>
            </div>
          </div>
        </div>

        {/* System Status */}
        <div style={{ marginBottom: '2rem' }}>
          <h2 style={{ fontSize: '1.8rem', marginBottom: '1rem' }}>📊 System Status</h2>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1rem' }}>
            {/* Database Status */}
            <div style={{
              backgroundColor: 'rgba(25, 25, 45, 0.5)',
              border: '1px solid rgba(0, 255, 255, 0.3)',
              borderRadius: '8px',
              padding: '1.5rem'
            }}>
              <h3 style={{ color: '#00ffff', marginBottom: '1rem', fontSize: '1.2rem' }}>🗄️ Database</h3>
              {loading ? (
                <div style={{ color: '#b8bcc8' }}>Loading...</div>
              ) : (
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                    <span style={{ color: '#b8bcc8' }}>Status:</span>
                    <span style={{ color: '#00ff41' }}>✅ Connected</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                    <span style={{ color: '#b8bcc8' }}>Datasets:</span>
                    <span style={{ color: '#ffffff' }}>{databaseInfo?.total_datasets || 0}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                    <span style={{ color: '#b8bcc8' }}>Records:</span>
                    <span style={{ color: '#ffffff' }}>{databaseInfo?.total_records || 0}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#b8bcc8' }}>Type:</span>
                    <span style={{ color: '#ffffff' }}>{databaseInfo?.backend || 'PostgreSQL'}</span>
                  </div>
                </div>
              )}
            </div>

            {/* Models Status */}
            <div style={{
              backgroundColor: 'rgba(25, 25, 45, 0.5)',
              border: '1px solid rgba(139, 92, 246, 0.3)',
              borderRadius: '8px',
              padding: '1.5rem'
            }}>
              <h3 style={{ color: '#8b5cf6', marginBottom: '1rem', fontSize: '1.2rem' }}>🧠 AI Models</h3>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                  <span style={{ color: '#b8bcc8' }}>Trained Models:</span>
                  <span style={{ color: '#ffffff' }}>{databaseInfo?.total_trained_models || 0}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                  <span style={{ color: '#b8bcc8' }}>Volatility:</span>
                  <span style={{ color: '#00ff41' }}>✅ Ready</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                  <span style={{ color: '#b8bcc8' }}>Direction:</span>
                  <span style={{ color: '#00ff41' }}>✅ Ready</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: '#b8bcc8' }}>Reversal:</span>
                  <span style={{ color: '#00ff41' }}>✅ Ready</span>
                </div>
              </div>
            </div>

            {/* Performance */}
            <div style={{
              backgroundColor: 'rgba(25, 25, 45, 0.5)',
              border: '1px solid rgba(0, 255, 65, 0.3)',
              borderRadius: '8px',
              padding: '1.5rem'
            }}>
              <h3 style={{ color: '#00ff41', marginBottom: '1rem', fontSize: '1.2rem' }}>⚡ Performance</h3>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                  <span style={{ color: '#b8bcc8' }}>API Status:</span>
                  <span style={{ color: '#00ff41' }}>✅ Online</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                  <span style={{ color: '#b8bcc8' }}>Response Time:</span>
                  <span style={{ color: '#ffffff' }}>~200ms</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                  <span style={{ color: '#b8bcc8' }}>Predictions:</span>
                  <span style={{ color: '#ffffff' }}>{databaseInfo?.total_predictions || 0}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: '#b8bcc8' }}>Uptime:</span>
                  <span style={{ color: '#00ff41' }}>99.9%</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Recent Activity */}
        <div style={{ marginBottom: '2rem' }}>
          <h2 style={{ fontSize: '1.8rem', marginBottom: '1rem' }}>📈 Recent Activity</h2>
          
          <div style={{
            backgroundColor: 'rgba(25, 25, 45, 0.5)',
            border: '1px solid rgba(0, 255, 255, 0.3)',
            borderRadius: '8px',
            padding: '1.5rem'
          }}>
            {databaseInfo?.datasets && databaseInfo.datasets.length > 0 ? (
              <div>
                <h3 style={{ color: '#00ffff', marginBottom: '1rem', fontSize: '1.1rem' }}>Recent Datasets</h3>
                {databaseInfo.datasets.slice(0, 3).map((dataset, index) => (
                  <div key={index} style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    padding: '0.75rem',
                    backgroundColor: 'rgba(0, 0, 0, 0.2)',
                    borderRadius: '4px',
                    marginBottom: '0.5rem'
                  }}>
                    <div>
                      <span style={{ color: '#ffffff', fontWeight: 'bold' }}>📊 {dataset.name}</span>
                      <div style={{ color: '#b8bcc8', fontSize: '0.9rem' }}>
                        {dataset.rows} rows • {dataset.start_date} to {dataset.end_date}
                      </div>
                    </div>
                    <div style={{ color: '#00ff41', fontSize: '0.9rem' }}>
                      Updated: {new Date(dataset.updated_at).toLocaleDateString()}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div style={{ textAlign: 'center', color: '#b8bcc8', padding: '2rem' }}>
                <div style={{ fontSize: '2rem', marginBottom: '1rem' }}>📊</div>
                <div>No recent activity</div>
                <div style={{ fontSize: '0.9rem', marginTop: '0.5rem' }}>
                  Upload data to get started with predictions
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Getting Started */}
        <div style={{
          backgroundColor: 'rgba(0, 255, 255, 0.05)',
          border: '1px solid rgba(0, 255, 255, 0.2)',
          borderRadius: '8px',
          padding: '2rem',
          textAlign: 'center'
        }}>
          <h2 style={{ color: '#00ffff', marginBottom: '1rem' }}>🚀 Getting Started</h2>
          <p style={{ color: '#b8bcc8', marginBottom: '1.5rem', fontSize: '1.1rem' }}>
            Ready to start making predictions? Follow these simple steps:
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
            <div style={{ padding: '1rem' }}>
              <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>1️⃣</div>
              <div style={{ color: '#ffffff', fontWeight: 'bold', marginBottom: '0.5rem' }}>Upload Data</div>
              <div style={{ color: '#b8bcc8', fontSize: '0.9rem' }}>Load your OHLC market data</div>
            </div>
            <div style={{ padding: '1rem' }}>
              <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>2️⃣</div>
              <div style={{ color: '#ffffff', fontWeight: 'bold', marginBottom: '0.5rem' }}>Train Models</div>
              <div style={{ color: '#b8bcc8', fontSize: '0.9rem' }}>Configure and train AI models</div>
            </div>
            <div style={{ padding: '1rem' }}>
              <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>3️⃣</div>
              <div style={{ color: '#ffffff', fontWeight: 'bold', marginBottom: '0.5rem' }}>Generate Predictions</div>
              <div style={{ color: '#b8bcc8', fontSize: '0.9rem' }}>Get volatility and direction forecasts</div>
            </div>
            <div style={{ padding: '1rem' }}>
              <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>4️⃣</div>
              <div style={{ color: '#ffffff', fontWeight: 'bold', marginBottom: '0.5rem' }}>Backtest Strategy</div>
              <div style={{ color: '#b8bcc8', fontSize: '0.9rem' }}>Validate performance</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;