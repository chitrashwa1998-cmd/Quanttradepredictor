/**
 * Sidebar component matching original Streamlit navigation
 */

import { Link, useLocation } from 'react-router-dom';
import { useState } from 'react';

const Sidebar = ({ isMobile = false, sidebarOpen = false, setSidebarOpen = () => {} }) => {
  const location = useLocation();
  const [isCollapsed, setIsCollapsed] = useState(false);

  const navigationItems = [
    { path: '/', label: 'Home', icon: '🏠' },
    { path: '/upload', label: 'Data Upload', icon: '📁' },
    { path: '/training', label: 'Model Training', icon: '🤖' },
    { path: '/predictions', label: 'Predictions', icon: '🔮' },
    { path: '/backtesting', label: 'Backtesting', icon: '📈' },
    { path: '/database', label: 'Database Manager', icon: '🗄️' },
    { path: '/live', label: 'Live Data', icon: '⚡' }
  ];

  return (
    <div style={{
      width: isCollapsed ? '80px' : '280px',
      minHeight: '100vh',
      background: 'var(--gradient-card)',
      border: '2px solid var(--border)',
      borderRadius: '0 20px 20px 0',
      padding: '2rem 1rem',
      position: 'fixed',
      left: '0',
      top: '0',
      zIndex: '1000',
      transition: 'all 0.3s ease',
      boxShadow: 'var(--shadow-glow)'
    }}>
      {/* Collapse Toggle */}
      <div style={{
        textAlign: 'right',
        marginBottom: '2rem'
      }}>
        <button
          onClick={() => setIsCollapsed(!isCollapsed)}
          style={{
            background: 'transparent',
            border: '2px solid var(--border)',
            borderRadius: '8px',
            color: 'var(--text-accent)',
            padding: '0.5rem',
            cursor: 'pointer',
            fontSize: '1.2rem',
            transition: 'all 0.3s ease'
          }}
          onMouseEnter={(e) => {
            e.target.style.borderColor = 'var(--border-hover)';
            e.target.style.color = 'var(--accent-cyan)';
          }}
          onMouseLeave={(e) => {
            e.target.style.borderColor = 'var(--border)';
            e.target.style.color = 'var(--text-accent)';
          }}
        >
          {isCollapsed ? '▶' : '◀'}
        </button>
      </div>

      {/* App Title */}
      {!isCollapsed && (
        <div style={{
          textAlign: 'center',
          marginBottom: '3rem',
          padding: '1rem 0'
        }}>
          <h2 style={{
            fontFamily: 'var(--font-display)',
            fontSize: '1.8rem',
            background: 'var(--gradient-primary)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
            marginBottom: '0.5rem'
          }}>
            TribexAlpha
          </h2>
          <p style={{
            color: 'var(--text-secondary)',
            fontSize: '0.9rem',
            fontFamily: 'var(--font-primary)'
          }}>
            Quantum Trading Engine
          </p>
        </div>
      )}

      {/* Navigation Items */}
      <nav>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          {navigationItems.map((item) => (
            <Link
              key={item.path}
              to={item.path}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '1rem',
                padding: '1rem',
                borderRadius: '12px',
                textDecoration: 'none',
                color: location.pathname === item.path ? 'var(--accent-cyan)' : 'var(--text-primary)',
                background: location.pathname === item.path ? 'var(--card-bg-hover)' : 'transparent',
                border: location.pathname === item.path ? '2px solid var(--accent-cyan)' : '2px solid transparent',
                fontFamily: 'var(--font-primary)',
                fontWeight: '500',
                fontSize: '1rem',
                transition: 'all 0.3s ease',
                borderLeft: location.pathname === item.path ? '4px solid var(--accent-cyan)' : '4px solid transparent'
              }}
              onMouseEnter={(e) => {
                if (location.pathname !== item.path) {
                  e.target.style.background = 'var(--card-bg)';
                  e.target.style.color = 'var(--accent-cyan)';
                  e.target.style.borderColor = 'var(--border-hover)';
                }
              }}
              onMouseLeave={(e) => {
                if (location.pathname !== item.path) {
                  e.target.style.background = 'transparent';
                  e.target.style.color = 'var(--text-primary)';
                  e.target.style.borderColor = 'transparent';
                }
              }}
            >
              <span style={{ fontSize: '1.2rem', minWidth: '24px' }}>{item.icon}</span>
              {!isCollapsed && <span>{item.label}</span>}
            </Link>
          ))}
        </div>
      </nav>

      {/* Status Section */}
      {!isCollapsed && (
        <div style={{
          position: 'absolute',
          bottom: '2rem',
          left: '1rem',
          right: '1rem',
          background: 'var(--card-bg)',
          border: '2px solid var(--border)',
          borderRadius: '12px',
          padding: '1rem',
          textAlign: 'center'
        }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '0.5rem',
            marginBottom: '0.5rem'
          }}>
            <span className="status-online">●</span>
            <span style={{
              color: 'var(--text-primary)',
              fontSize: '0.9rem',
              fontFamily: 'var(--font-primary)'
            }}>
              System Online
            </span>
          </div>
          <p style={{
            color: 'var(--text-secondary)',
            fontSize: '0.8rem',
            fontFamily: 'var(--font-mono)'
          }}>
            v2.0 React + FastAPI
          </p>
        </div>
      )}
    </div>
  );
};

export default Sidebar;