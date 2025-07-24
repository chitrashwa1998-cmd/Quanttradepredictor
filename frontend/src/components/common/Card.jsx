/**
 * Card component with original Streamlit styling
 */

const Card = ({ children, className = '', hover = true, glow = false, ...props }) => {
  const cardStyle = {
    background: 'var(--gradient-card)',
    border: '2px solid var(--border)',
    borderRadius: '16px',
    padding: '2rem',
    margin: '1rem 0',
    boxShadow: 'var(--shadow)',
    transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
    position: 'relative',
    overflow: 'hidden',
    ...(glow && { boxShadow: 'var(--shadow-glow)' })
  };

  const handleMouseEnter = (e) => {
    if (hover) {
      e.target.style.borderColor = 'var(--border-hover)';
      e.target.style.background = 'var(--card-bg-hover)';
      e.target.style.transform = 'translateY(-5px)';
      e.target.style.boxShadow = '0 12px 40px rgba(0, 255, 255, 0.2)';
    }
  };

  const handleMouseLeave = (e) => {
    if (hover) {
      e.target.style.borderColor = 'var(--border)';
      e.target.style.background = 'var(--gradient-card)';
      e.target.style.transform = 'translateY(0)';
      e.target.style.boxShadow = glow ? 'var(--shadow-glow)' : 'var(--shadow)';
    }
  };

  return (
    <div
      className={`metric-container ${className}`}
      style={cardStyle}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      {...props}
    >
      {/* Hover border effect */}
      <div style={{
        content: '',
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        height: '3px',
        background: 'var(--gradient-primary)',
        opacity: 0,
        transition: 'opacity 0.3s ease'
      }}></div>
      {children}
    </div>
  );
};

export default Card;