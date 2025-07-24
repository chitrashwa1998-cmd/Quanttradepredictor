/**
 * Layout component with original Streamlit sidebar layout
 */

import { useState, useEffect } from 'react';
import Sidebar from './Sidebar';

const Layout = ({ children }) => {
  const [isMobile, setIsMobile] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth <= 768);
    };
    
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  return (
    <div className="min-h-screen" style={{background: 'var(--bg-primary)', color: 'var(--text-primary)'}}>
      {/* Mobile Overlay */}
      {isMobile && sidebarOpen && (
        <div 
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.5)',
            zIndex: 999
          }}
          onClick={() => setSidebarOpen(false)}
        />
      )}
      
      <Sidebar isMobile={isMobile} sidebarOpen={sidebarOpen} setSidebarOpen={setSidebarOpen} />
      
      {/* Mobile Menu Button */}
      {isMobile && (
        <button
          onClick={() => setSidebarOpen(true)}
          style={{
            position: 'fixed',
            top: '1rem',
            left: '1rem',
            zIndex: 1001,
            background: 'var(--gradient-card)',
            border: '2px solid var(--border)',
            borderRadius: '8px',
            color: 'var(--text-accent)',
            padding: '0.5rem',
            fontSize: '1.2rem',
            cursor: 'pointer'
          }}
        >
          ☰
        </button>
      )}
      
      <main 
        className={isMobile ? 'main-content-mobile' : ''}
        style={{
          marginLeft: isMobile ? '0' : '280px',
          padding: isMobile ? '4rem 2rem 2rem 2rem' : '3rem 4rem',
          background: 'transparent',
          minHeight: '100vh'
        }}
      >
        {children}
      </main>
    </div>
  );
};

export default Layout;