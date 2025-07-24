/**
 * Sidebar component matching original Streamlit navigation
 */

import { Link, useLocation } from 'react-router-dom';
import { useState } from 'react';

const Sidebar = ({ isMobile = false, sidebarOpen = false, setSidebarOpen = () => {} }) => {
  const location = useLocation();

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
    <div 
      className={isMobile ? (sidebarOpen ? 'sidebar-mobile open' : 'sidebar-mobile') : 'sidebar-desktop'}
      style={{
        width: '280px',
        minHeight: '100vh',
        background: 'var(--bg-secondary)',
        borderRight: '1px solid var(--border)',
        position: 'fixed',
        left: '0',
        top: '0',
        zIndex: '1000',
        transition: 'all 0.3s ease',
        overflow: 'hidden',
        ...(isMobile && {
          transform: sidebarOpen ? 'translateX(0)' : 'translateX(-100%)',
          boxShadow: '2px 0 8px rgba(0, 0, 0, 0.3)'
        })
      }}
    >
      {/* Scrollable Content */}
      <div style={{
        height: '100vh',
        overflowY: 'auto',
        overflowX: 'hidden',
        padding: '1.5rem 0'
      }}>
        {/* TribexAlpha Logo */}
        <div style={{
          textAlign: 'center',
          padding: '1.5rem',
          background: 'linear-gradient(135deg, #00ffff 0%, #8b5cf6 100%)',
          borderRadius: '16px',
          margin: '0 1rem 2rem 1rem'
        }}>
          <h2 style={{
            color: 'white',
            margin: '0',
            fontFamily: 'var(--font-display)',
            fontSize: '1.4rem',
            fontWeight: '700'
          }}>
            ⚡ TribexAlpha
          </h2>
        </div>

        {/* Navigation Items */}
        <nav style={{ padding: '0 1rem' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            {navigationItems.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                onClick={() => isMobile && setSidebarOpen(false)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.75rem',
                  padding: '0.75rem 1rem',
                  borderRadius: '8px',
                  textDecoration: 'none',
                  color: location.pathname === item.path ? 'var(--accent-cyan)' : 'var(--text-primary)',
                  background: location.pathname === item.path ? 'rgba(0, 255, 255, 0.1)' : 'transparent',
                  border: location.pathname === item.path ? '1px solid var(--accent-cyan)' : '1px solid transparent',
                  fontFamily: 'var(--font-primary)',
                  fontSize: '0.9rem',
                  fontWeight: '500',
                  transition: 'all 0.2s ease'
                }}
                onMouseEnter={(e) => {
                  if (location.pathname !== item.path) {
                    e.target.style.background = 'rgba(255, 255, 255, 0.05)';
                    e.target.style.color = 'var(--accent-cyan)';
                  }
                }}
                onMouseLeave={(e) => {
                  if (location.pathname !== item.path) {
                    e.target.style.background = 'transparent';
                    e.target.style.color = 'var(--text-primary)';
                  }
                }}
              >
                <span style={{ fontSize: '1rem', minWidth: '20px' }}>{item.icon}</span>
                <span>{item.label}</span>
              </Link>
            ))}
          </div>
        </nav>

        {/* Database Status */}
        <div style={{
          margin: '2rem 1rem 1rem 1rem',
          borderTop: '1px solid var(--border)',
          paddingTop: '1rem'
        }}>
          <div style={{
            background: 'rgba(51, 103, 145, 0.1)',
            border: '1px solid #336791',
            borderRadius: '8px',
            padding: '0.8rem',
            textAlign: 'center'
          }}>
            <div style={{ color: '#336791', fontSize: '1.2rem', marginBottom: '0.25rem' }}>🐘</div>
            <div style={{ color: '#336791', fontSize: '0.8rem', fontWeight: 'bold', marginBottom: '0.25rem' }}>
              PostgreSQL Row-Based
            </div>
            <div style={{ color: '#8b949e', fontSize: '0.7rem' }}>Connected</div>
          </div>
        </div>

        {/* Footer */}
        <div style={{
          position: 'absolute',
          bottom: '1rem',
          left: '1rem',
          right: '1rem',
          textAlign: 'center',
          color: 'var(--text-secondary)',
          fontSize: '0.7rem',
          fontFamily: 'var(--font-mono)'
        }}>
          v2.0 React + FastAPI
        </div>
      </div>
    </div>
  );
};

export default Sidebar;