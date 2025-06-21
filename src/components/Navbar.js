
import React from 'react';
import { Link, useLocation } from 'react-router-dom';

const Navbar = () => {
  const location = useLocation();

  const navItems = [
    { path: '/', label: '🏠 HOME' },
    { path: '/data-upload', label: '📊 DATA UPLOAD' },
    { path: '/model-training', label: '🔬 MODEL TRAINING' },
    { path: '/predictions', label: '🎯 PREDICTIONS' },
    { path: '/backtesting', label: '📈 BACKTESTING' },
    
    { path: '/database', label: '💾 DATABASE' }
  ];

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <Link to="/" className="sidebar-brand">
          ⚡ TribexAlpha
        </Link>
      </div>
      <nav className="sidebar-nav">
        <ul className="sidebar-links">
          {navItems.map((item) => (
            <li key={item.path}>
              <Link
                to={item.path}
                className={location.pathname === item.path ? 'active' : ''}
              >
                {item.label}
              </Link>
            </li>
          ))}
        </ul>
      </nav>
    </div>
  );
};

export default Navbar;
