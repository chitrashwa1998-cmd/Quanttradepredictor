/**
 * Header component with original Streamlit styling
 */

import { Link, useLocation } from 'react-router-dom';

const Header = () => {
  const location = useLocation();

  return (
    <div>
      {/* Main Header */}
      <div className="trading-header" style={{
        background: 'var(--gradient-card)',
        border: '2px solid var(--border)',
        borderRadius: '20px',
        padding: '3rem 2rem',
        margin: '2rem 0',
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
          fontFamily: 'var(--font-primary)',
          color: 'var(--accent-electric)',
          fontSize: '1.2rem',
          fontWeight: '500'
        }}>
          ⚡ Quantum Trading Engine
        </p>
      </div>

      {/* Navigation */}
      <nav style={{
        background: 'var(--gradient-card)',
        border: '2px solid var(--border)',
        borderRadius: '16px',
        padding: '1rem 2rem',
        margin: '1rem 0',
        boxShadow: 'var(--shadow)'
      }}>
        <div className="flex flex-wrap justify-center gap-4">
          {[
            { path: '/', label: 'Dashboard', icon: '📊' },
            { path: '/upload', label: 'Data Upload', icon: '📁' },
            { path: '/training', label: 'Model Training', icon: '🤖' },
            { path: '/predictions', label: 'Predictions', icon: '🔮' },
            { path: '/live', label: 'Live Trading', icon: '⚡' },
            { path: '/backtesting', label: 'Backtesting', icon: '📈' },
            { path: '/database', label: 'Database', icon: '🗄️' }
          ].map((item) => (
            <Link
              key={item.path}
              to={item.path}
              style={{
                background: location.pathname === item.path ? 'var(--card-bg-hover)' : 'var(--gradient-card)',
                border: '2px solid var(--border)',
                borderRadius: '12px',
                padding: '0.75rem 1.5rem',
                margin: '0.25rem',
                color: 'var(--text-primary)',
                textDecoration: 'none',
                fontFamily: 'var(--font-primary)',
                fontWeight: '500',
                transition: 'all 0.3s ease',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                borderLeft: location.pathname === item.path ? '4px solid var(--accent-cyan)' : '2px solid var(--border)'
              }}
              onMouseEnter={(e) => {
                e.target.style.background = 'var(--card-bg-hover)';
                e.target.style.borderColor = 'var(--border-hover)';
                e.target.style.transform = 'translateY(-2px)';
              }}
              onMouseLeave={(e) => {
                if (location.pathname !== item.path) {
                  e.target.style.background = 'var(--gradient-card)';
                  e.target.style.borderColor = 'var(--border)';
                }
                e.target.style.transform = 'translateY(0)';
              }}
            >
              <span>{item.icon}</span>
              <span>{item.label}</span>
            </Link>
          ))}
        </div>
      </nav>
    </div>
  );
};

export default Header;