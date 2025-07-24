/**
 * Error boundary component for handling React component errors
 */

import { Component } from 'react';

class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    this.setState({
      error: error,
      errorInfo: errorInfo
    });
    
    // Log error to console for debugging
    console.error('Error caught by boundary:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-cyber-dark">
          <div className="max-w-md w-full cyber-bg cyber-border rounded-lg p-6">
            <div className="text-center">
              <h1 className="text-2xl font-bold text-cyber-red mb-4">
                Something went wrong
              </h1>
              <p className="text-gray-300 mb-6">
                An unexpected error occurred in the application.
              </p>
              
              {this.props.showDetails && (
                <div className="text-left bg-gray-900 rounded p-3 mb-4">
                  <pre className="text-xs text-gray-400 overflow-auto">
                    {this.state.error && this.state.error.toString()}
                  </pre>
                </div>
              )}
              
              <button
                onClick={() => window.location.reload()}
                className="w-full cyber-border rounded-md py-2 px-4 text-cyber-blue hover:cyber-glow transition-all duration-200"
              >
                Reload Application
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;