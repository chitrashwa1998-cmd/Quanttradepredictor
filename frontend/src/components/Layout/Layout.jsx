/**
 * Layout component with original Streamlit styling
 */

import Header from './Header';

const Layout = ({ children }) => {
  return (
    <div className="min-h-screen" style={{background: 'var(--bg-primary)', color: 'var(--text-primary)'}}>
      <Header />
      <main style={{
        maxWidth: '1400px',
        margin: '0 auto',
        padding: '2rem 3rem',
        background: 'transparent'
      }}>
        {children}
      </main>
    </div>
  );
};

export default Layout;