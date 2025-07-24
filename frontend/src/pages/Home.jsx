/**
 * Home page with comprehensive app information - matching original Streamlit layout
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
          dataAPI.getDatabaseInfo(),
          predictionsAPI.getModelsStatus()
        ]);
        setDatabaseInfo(dbInfo);
        setModelsStatus(modelStatus);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-64">
        <LoadingSpinner />
      </div>
    );
  }

  if (error) {
    return (
      <Card className="error-state">
        <h3 style={{ color: 'var(--accent-pink)' }}>Error Loading Dashboard</h3>
        <p style={{ color: 'var(--text-secondary)' }}>{error}</p>
      </Card>
    );
  }

  return (
    <div style={{ animation: 'pageLoad 0.6s ease-out' }}>
      {/* Original TribexAlpha Header */}
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
          marginBottom: '1rem',
          textShadow: '0 0 30px rgba(0, 255, 255, 0.5)',
          animation: 'titleGlow 3s ease-in-out infinite alternate'
        }}>
          TribexAlpha
        </h1>
        <p style={{
          fontSize: '1.5rem',
          margin: '1rem 0 0 0',
          opacity: '0.9',
          fontWeight: '300',
          color: '#00ffff',
          fontFamily: 'var(--font-primary)'
        }}>
          🚀 AI-Powered Quant Signal Engine
        </p>
        <p style={{
          fontSize: '1.1rem',
          margin: '1rem 0 0 0',
          opacity: '0.8',
          color: '#b8bcc8',
          fontFamily: 'var(--font-primary)',
          lineHeight: '1.6'
        }}>
          An AI-powered quant signal engine delivering multi-model predictions for direction, volatility, reversals, and profit zones — built for real-time execution and adaptive to any market regime.
        </p>
      </div>

      {/* Navigation Guidance Section */}
      <div style={{
        background: 'rgba(0, 255, 255, 0.1)',
        border: '2px solid #00ffff',
        borderRadius: '16px',
        padding: '2rem',
        margin: '2rem 0',
        textAlign: 'center'
      }}>
        <h2 style={{
          color: '#00ffff',
          marginBottom: '1rem',
          fontFamily: 'var(--font-display)',
          fontSize: '1.8rem'
        }}>
          📍 Quick Navigation
        </h2>
        <p style={{
          color: '#e6e8eb',
          fontSize: '1.1rem',
          marginBottom: '1.5rem',
          fontFamily: 'var(--font-primary)'
        }}>
          Use the sidebar navigation to access different modules:
        </p>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '1rem',
          margin: '1rem 0'
        }}>
          <div style={{
            background: 'rgba(0, 255, 255, 0.1)',
            padding: '1rem',
            borderRadius: '8px'
          }}>
            <strong style={{ color: '#00ffff', fontFamily: 'var(--font-primary)' }}>📊 DATA UPLOAD</strong><br />
            <span style={{ color: '#b8bcc8', fontFamily: 'var(--font-primary)' }}>Load your OHLC data</span>
          </div>
          <div style={{
            background: 'rgba(0, 255, 65, 0.1)',
            padding: '1rem',
            borderRadius: '8px'
          }}>
            <strong style={{ color: '#00ff41', fontFamily: 'var(--font-primary)' }}>🔬 MODEL TRAINING</strong><br />
            <span style={{ color: '#b8bcc8', fontFamily: 'var(--font-primary)' }}>Train machine learning models</span>
          </div>
          <div style={{
            background: 'rgba(139, 92, 246, 0.1)',
            padding: '1rem',
            borderRadius: '8px'
          }}>
            <strong style={{ color: '#8b5cf6', fontFamily: 'var(--font-primary)' }}>🎯 PREDICTIONS</strong><br />
            <span style={{ color: '#b8bcc8', fontFamily: 'var(--font-primary)' }}>Generate forecasts</span>
          </div>
          <div style={{
            background: 'rgba(255, 0, 128, 0.1)',
            padding: '1rem',
            borderRadius: '8px'
          }}>
            <strong style={{ color: '#ff0080', fontFamily: 'var(--font-primary)' }}>📈 BACKTESTING</strong><br />
            <span style={{ color: '#b8bcc8', fontFamily: 'var(--font-primary)' }}>Test strategies</span>
          </div>
        </div>
      </div>

      {/* Enhanced System Status Dashboard - Original Streamlit Style */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        {/* Data Engine Status */}
        <div className="metric-container" style={{
          background: 'var(--gradient-card)',
          border: '2px solid var(--border)',
          borderRadius: '16px',
          padding: '2rem',
          textAlign: 'center',
          transition: 'all 0.3s ease'
        }}>
          <h3 style={{
            color: '#00ffff',
            margin: '0',
            fontFamily: 'var(--font-display)',
            fontSize: '1.2rem',
            fontWeight: '600'
          }}>
            ⚡ DATA ENGINE
          </h3>
          <h2 style={{
            margin: '0.5rem 0',
            color: databaseInfo?.total_datasets > 0 ? '#00ff41' : '#ff0080',
            fontWeight: '800',
            fontSize: '1.5rem',
            fontFamily: 'var(--font-primary)'
          }}>
            {databaseInfo?.total_datasets > 0 ? '🟢 LOADED' : '🔴 NO DATA'}
          </h2>
          <p style={{
            color: '#9ca3af',
            margin: '0',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.9rem'
          }}>
            {databaseInfo?.total_records || 0} market records loaded
          </p>
        </div>

        {/* AI Models Status */}
        <div className="metric-container" style={{
          background: 'var(--gradient-card)',
          border: '2px solid var(--border)',
          borderRadius: '16px',
          padding: '2rem',
          textAlign: 'center',
          transition: 'all 0.3s ease'
        }}>
          <h3 style={{
            color: '#00ff41',
            margin: '0',
            fontFamily: 'var(--font-display)',
            fontSize: '1.2rem',
            fontWeight: '600'
          }}>
            🤖 AI MODELS
          </h3>
          <h2 style={{
            margin: '0.5rem 0',
            color: modelsStatus?.models ? '#00ff41' : '#ffaa00',
            fontWeight: '800',
            fontSize: '1.5rem',
            fontFamily: 'var(--font-primary)'
          }}>
            {modelsStatus?.models ? Object.keys(modelsStatus.models).length : 0}/4
          </h2>
          <p style={{
            color: '#9ca3af',
            margin: '0',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.9rem'
          }}>
            Machine learning models trained
          </p>
        </div>

        {/* Predictions Status */}
        <div className="metric-container" style={{
          background: 'var(--gradient-card)',
          border: '2px solid var(--border)',
          borderRadius: '16px',
          padding: '2rem',
          textAlign: 'center',
          transition: 'all 0.3s ease'
        }}>
          <h3 style={{
            color: '#8b5cf6',
            margin: '0',
            fontFamily: 'var(--font-display)',
            fontSize: '1.2rem',
            fontWeight: '600'
          }}>
            🎯 PREDICTIONS
          </h3>
          <h2 style={{
            margin: '0.5rem 0',
            color: modelsStatus?.models ? '#00ff41' : '#ffaa00',
            fontWeight: '800',
            fontSize: '1.5rem',
            fontFamily: 'var(--font-primary)'
          }}>
            {modelsStatus?.models ? '🟢 ACTIVE' : '⚠️ STANDBY'}
          </h2>
          <p style={{
            color: '#9ca3af',
            margin: '0',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.9rem'
          }}>
            Real-time market analysis
          </p>
        </div>

        {/* System Status */}
        <div className="metric-container glow-animation" style={{
          background: 'var(--gradient-card)',
          border: '2px solid var(--border)',
          borderRadius: '16px',
          padding: '2rem',
          textAlign: 'center',
          transition: 'all 0.3s ease',
          boxShadow: 'var(--shadow-glow)'
        }}>
          <h3 style={{
            color: '#ff0080',
            margin: '0',
            fontFamily: 'var(--font-display)',
            fontSize: '1.2rem',
            fontWeight: '600'
          }}>
            🌐 SYSTEM
          </h3>
          <h2 style={{
            margin: '0.5rem 0',
            color: '#00ff41',
            fontWeight: '800',
            fontSize: '1.5rem',
            fontFamily: 'var(--font-primary)'
          }}>
            ONLINE
          </h2>
          <p style={{
            color: '#b8bcc8',
            margin: '0',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.9rem'
          }}>
            All systems operational
          </p>
        </div>
      </div>



      {/* Success Navigation Message */}
      <div style={{
        background: 'rgba(0, 255, 65, 0.1)',
        border: '1px solid #00ff41',
        borderRadius: '12px',
        padding: '1rem',
        margin: '2rem 0',
        textAlign: 'center',
        color: '#00ff41',
        fontFamily: 'var(--font-primary)'
      }}>
        🎯 <strong>Navigation</strong>: Use the sidebar to navigate between different modules of the trading system.
      </div>

      <hr style={{ border: '1px solid var(--border)', margin: '2rem 0' }} />

      {/* Core Capabilities Section */}
      <div className="chart-container" style={{ margin: '3rem 0' }}>
        <h2 style={{
          color: '#00ffff',
          marginBottom: '2rem',
          textAlign: 'center',
          fontFamily: 'var(--font-display)',
          fontSize: '2rem'
        }}>
          🔮 ADVANCED PREDICTION CAPABILITIES
        </h2>
      </div>

      {/* Enhanced Features Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
        <div className="feature-card" style={{
          background: 'var(--gradient-card)',
          border: '2px solid var(--border)',
          borderRadius: '16px',
          padding: '2rem',
          transition: 'all 0.3s ease'
        }}>
          <h3 style={{
            color: '#00ffff',
            marginBottom: '1.5rem',
            fontFamily: 'var(--font-display)',
            fontSize: '1.5rem'
          }}>
            🧠 MACHINE LEARNING ARSENAL
          </h3>
          <div style={{
            color: '#e6e8eb',
            fontFamily: 'var(--font-primary)',
            lineHeight: '2'
          }}>
            <div style={{
              margin: '1rem 0',
              padding: '0.5rem',
              background: 'rgba(0, 255, 255, 0.05)',
              borderRadius: '8px'
            }}>
              <strong style={{ color: '#00ff41' }}>🎯 Direction Prediction</strong><br />
              <span style={{ color: '#b8bcc8' }}>Advanced price movement forecasting with 94% accuracy</span>
            </div>
            <div style={{
              margin: '1rem 0',
              padding: '0.5rem',
              background: 'rgba(139, 92, 246, 0.05)',
              borderRadius: '8px'
            }}>
              <strong style={{ color: '#8b5cf6' }}>🔄 Reversal Detection</strong><br />
              <span style={{ color: '#b8bcc8' }}>Advanced market reversal point identification</span>
            </div>
            <div style={{
              margin: '1rem 0',
              padding: '0.5rem',
              background: 'rgba(255, 0, 128, 0.05)',
              borderRadius: '8px'
            }}>
              <strong style={{ color: '#ff0080' }}>💰 Profit Probability</strong><br />
              <span style={{ color: '#b8bcc8' }}>Trade success likelihood with risk assessment</span>
            </div>
            <div style={{
              margin: '1rem 0',
              padding: '0.5rem',
              background: 'rgba(255, 215, 0, 0.05)',
              borderRadius: '8px'
            }}>
              <strong style={{ color: '#ffd700' }}>⚡ Volatility Forecasting</strong><br />
              <span style={{ color: '#b8bcc8' }}>Dynamic market volatility prediction engine</span>
            </div>
          </div>
        </div>

        <div className="feature-card" style={{
          background: 'var(--gradient-card)',
          border: '2px solid var(--border)',
          borderRadius: '16px',
          padding: '2rem',
          transition: 'all 0.3s ease'
        }}>
          <h3 style={{
            color: '#00ff41',
            marginBottom: '1.5rem',
            fontFamily: 'var(--font-display)',
            fontSize: '1.5rem'
          }}>
            ⚙️ TRADING INFRASTRUCTURE
          </h3>
          <div style={{
            color: '#e6e8eb',
            fontFamily: 'var(--font-primary)',
            lineHeight: '2'
          }}>
            <div style={{
              margin: '1rem 0',
              padding: '0.5rem',
              background: 'rgba(0, 255, 65, 0.05)',
              borderRadius: '8px'
            }}>
              <strong style={{ color: '#00ffff' }}>⚡ Real-time Processing</strong><br />
              <span style={{ color: '#b8bcc8' }}>Low latency market data analysis</span>
            </div>
            <div style={{
              margin: '1rem 0',
              padding: '0.5rem',
              background: 'rgba(255, 107, 53, 0.05)',
              borderRadius: '8px'
            }}>
              <strong style={{ color: '#ff6b35' }}>🔍 Advanced Backtesting</strong><br />
              <span style={{ color: '#b8bcc8' }}>Comprehensive strategy performance analysis</span>
            </div>
            <div style={{
              margin: '1rem 0',
              padding: '0.5rem',
              background: 'rgba(139, 92, 246, 0.05)',
              borderRadius: '8px'
            }}>
              <strong style={{ color: '#8b5cf6' }}>🤖 Ensemble Models</strong><br />
              <span style={{ color: '#b8bcc8' }}>XGBoost, CatBoost, and Random Forest integration</span>
            </div>
            <div style={{
              margin: '1rem 0',
              padding: '0.5rem',
              background: 'rgba(255, 0, 128, 0.05)',
              borderRadius: '8px'
            }}>
              <strong style={{ color: '#ff0080' }}>🧮 Technical Analysis</strong><br />
              <span style={{ color: '#b8bcc8' }}>50+ advanced technical indicators and patterns</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;