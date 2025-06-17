
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
    { path: '/realtime-data', label: '📊 REALTIME DATA' },
    { path: '/database', label: '💾 DATABASE' }
  ];

  return (
    <nav className="nav">
      <div className="container">
        <div className="nav-container">
          <Link to="/" className="nav-brand">
            ⚡ TribexAlpha
          </Link>
          <ul className="nav-links">
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
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
