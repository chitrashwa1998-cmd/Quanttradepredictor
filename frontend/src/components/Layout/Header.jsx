/**
 * Header component with navigation and status
 */

import { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Bars3Icon, XMarkIcon } from '@heroicons/react/24/outline';
import { healthCheck } from '../../services/api';

const navigation = [
  { name: 'Dashboard', href: '/', current: false },
  { name: 'Upload Data', href: '/upload', current: false },
  { name: 'Train Models', href: '/training', current: false },
  { name: 'Predictions', href: '/predictions', current: false },
  { name: 'Live Trading', href: '/live', current: false },
  { name: 'Backtesting', href: '/backtesting', current: false },
  { name: 'Database', href: '/database', current: false },
];

export default function Header() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [isHealthy, setIsHealthy] = useState(false);
  const location = useLocation();

  // Check backend health
  useEffect(() => {
    const checkHealth = async () => {
      try {
        await healthCheck();
        setIsHealthy(true);
      } catch (error) {
        setIsHealthy(false);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 30000); // Check every 30 seconds

    return () => clearInterval(interval);
  }, []);

  // Update current navigation item
  const updatedNavigation = navigation.map(item => ({
    ...item,
    current: location.pathname === item.href
  }));

  return (
    <header className="cyber-bg border-b cyber-border">
      <nav className="mx-auto flex max-w-7xl items-center justify-between p-6 lg:px-8" aria-label="Global">
        {/* Logo */}
        <div className="flex lg:flex-1">
          <Link to="/" className="-m-1.5 p-1.5">
            <span className="text-2xl font-bold cyber-text">TribexAlpha</span>
            <span className="ml-2 text-sm text-gray-400">v2.0</span>
          </Link>
        </div>

        {/* Mobile menu button */}
        <div className="flex lg:hidden">
          <button
            type="button"
            className="-m-2.5 inline-flex items-center justify-center rounded-md p-2.5 text-gray-400 hover:text-cyber-blue"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          >
            <span className="sr-only">Open main menu</span>
            {mobileMenuOpen ? (
              <XMarkIcon className="h-6 w-6" aria-hidden="true" />
            ) : (
              <Bars3Icon className="h-6 w-6" aria-hidden="true" />
            )}
          </button>
        </div>

        {/* Desktop navigation */}
        <div className="hidden lg:flex lg:gap-x-8">
          {updatedNavigation.map((item) => (
            <Link
              key={item.name}
              to={item.href}
              className={`text-sm font-medium transition-colors duration-200 ${
                item.current
                  ? 'cyber-text'
                  : 'text-gray-300 hover:text-cyber-blue'
              }`}
            >
              {item.name}
            </Link>
          ))}
        </div>

        {/* Status indicator */}
        <div className="hidden lg:flex lg:flex-1 lg:justify-end items-center">
          <div className="flex items-center space-x-2">
            <div
              className={`w-2 h-2 rounded-full ${
                isHealthy ? 'bg-cyber-green' : 'bg-cyber-red'
              } animate-pulse`}
            />
            <span className="text-sm text-gray-400">
              {isHealthy ? 'Backend Online' : 'Backend Offline'}
            </span>
          </div>
        </div>
      </nav>

      {/* Mobile menu */}
      {mobileMenuOpen && (
        <div className="lg:hidden cyber-bg border-t cyber-border">
          <div className="space-y-2 px-6 pb-6 pt-6">
            {updatedNavigation.map((item) => (
              <Link
                key={item.name}
                to={item.href}
                className={`block rounded-md px-3 py-2 text-base font-medium transition-colors duration-200 ${
                  item.current
                    ? 'cyber-text cyber-bg'
                    : 'text-gray-300 hover:text-cyber-blue hover:bg-gray-800'
                }`}
                onClick={() => setMobileMenuOpen(false)}
              >
                {item.name}
              </Link>
            ))}
            
            {/* Mobile status */}
            <div className="flex items-center space-x-2 px-3 py-2">
              <div
                className={`w-2 h-2 rounded-full ${
                  isHealthy ? 'bg-cyber-green' : 'bg-cyber-red'
                } animate-pulse`}
              />
              <span className="text-sm text-gray-400">
                {isHealthy ? 'Backend Online' : 'Backend Offline'}
              </span>
            </div>
          </div>
        </div>
      )}
    </header>
  );
}