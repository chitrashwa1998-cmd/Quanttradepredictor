/**
 * Database Manager page - Exact Streamlit UI replication
 */

import { useState, useEffect } from 'react';
import { dataAPI } from '../services/api';

const DatabaseManager = () => {
  const [databaseInfo, setDatabaseInfo] = useState(null);
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(false);
  const [actionStatus, setActionStatus] = useState('');

  // Load database information
  const loadDatabaseInfo = async () => {
    try {
      setLoading(true);
      const [dbResponse, datasetsResponse] = await Promise.all([
        dataAPI.getDatabaseInfo(),
        dataAPI.getDatasets()
      ]);
      
      setDatabaseInfo(dbResponse?.data || {});
      setDatasets(Array.isArray(datasetsResponse?.data) ? datasetsResponse.data : []);
    } catch (error) {
      console.error('Error loading database info:', error);
      setActionStatus(`❌ Error loading database info: ${error?.message || 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDatabaseInfo();
  }, []);

  return (
    <div style={{ backgroundColor: '#0a0a0f', minHeight: '100vh', color: '#ffffff', fontFamily: 'Space Grotesk, sans-serif' }}>
      <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '2rem' }}>
        
        {/* Header */}
        <h1 style={{ fontSize: '2.5rem', marginBottom: '0.5rem', fontFamily: 'Orbitron, monospace' }}>
          🗄️ Database Manager
        </h1>
        <p style={{ color: '#b8bcc8', fontSize: '1.1rem', marginBottom: '2rem' }}>
          Monitor and manage your PostgreSQL database and datasets.
        </p>

        {/* Status */}
        {actionStatus && (
          <div style={{
            padding: '1rem',
            borderRadius: '8px',
            marginBottom: '2rem',
            backgroundColor: actionStatus.includes('✅') ? 'rgba(0, 255, 65, 0.1)' : 'rgba(255, 0, 128, 0.1)',
            border: `1px solid ${actionStatus.includes('✅') ? 'rgba(0, 255, 65, 0.3)' : 'rgba(255, 0, 128, 0.3)'}`,
            color: actionStatus.includes('✅') ? '#00ff41' : '#ff0080'
          }}>
            {actionStatus}
          </div>
        )}

        {/* Database Overview */}
        <div style={{ marginBottom: '2rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <h2 style={{ fontSize: '1.8rem', margin: 0 }}>📊 Database Overview</h2>
            <button
              onClick={loadDatabaseInfo}
              style={{
                backgroundColor: '#00ffff',
                color: '#0a0a0f',
                border: 'none',
                padding: '0.5rem 1rem',
                borderRadius: '4px',
                fontSize: '0.9rem',
                fontWeight: 'bold',
                cursor: 'pointer'
              }}
              disabled={loading}
            >
              🔄 Refresh
            </button>
          </div>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
            <div style={{
              backgroundColor: 'rgba(25, 25, 45, 0.5)',
              border: '1px solid rgba(0, 255, 255, 0.3)',
              borderRadius: '8px',
              padding: '1.5rem',
              textAlign: 'center'
            }}>
              <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>🗄️</div>
              <div style={{ fontSize: '1.5rem', color: '#00ffff', fontWeight: 'bold', marginBottom: '0.5rem' }}>
                {loading ? '...' : databaseInfo?.database_type || 'PostgreSQL'}
              </div>
              <div style={{ color: '#b8bcc8', fontSize: '0.9rem' }}>Database Type</div>
            </div>
            
            <div style={{
              backgroundColor: 'rgba(25, 25, 45, 0.5)',
              border: '1px solid rgba(0, 255, 65, 0.3)',
              borderRadius: '8px',
              padding: '1.5rem',
              textAlign: 'center'
            }}>
              <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>📊</div>
              <div style={{ fontSize: '1.5rem', color: '#00ff41', fontWeight: 'bold', marginBottom: '0.5rem' }}>
                {loading ? '...' : databaseInfo?.total_datasets || 0}
              </div>
              <div style={{ color: '#b8bcc8', fontSize: '0.9rem' }}>Total Datasets</div>
            </div>
            
            <div style={{
              backgroundColor: 'rgba(25, 25, 45, 0.5)',
              border: '1px solid rgba(139, 92, 246, 0.3)',
              borderRadius: '8px',
              padding: '1.5rem',
              textAlign: 'center'
            }}>
              <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>📈</div>
              <div style={{ fontSize: '1.5rem', color: '#8b5cf6', fontWeight: 'bold', marginBottom: '0.5rem' }}>
                {loading ? '...' : databaseInfo?.total_records?.toLocaleString() || 0}
              </div>
              <div style={{ color: '#b8bcc8', fontSize: '0.9rem' }}>Total Records</div>
            </div>
            
            <div style={{
              backgroundColor: 'rgba(25, 25, 45, 0.5)',
              border: '1px solid rgba(255, 0, 128, 0.3)',
              borderRadius: '8px',
              padding: '1.5rem',
              textAlign: 'center'
            }}>
              <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>🤖</div>
              <div style={{ fontSize: '1.5rem', color: '#ff0080', fontWeight: 'bold', marginBottom: '0.5rem' }}>
                {loading ? '...' : databaseInfo?.total_trained_models || 0}
              </div>
              <div style={{ color: '#b8bcc8', fontSize: '0.9rem' }}>Trained Models</div>
            </div>
          </div>
        </div>

        {/* Datasets Management */}
        <div style={{ marginBottom: '2rem' }}>
          <h2 style={{ fontSize: '1.8rem', marginBottom: '1rem' }}>📋 Datasets Management</h2>
          
          {datasets.length > 0 ? (
            <div style={{
              backgroundColor: 'rgba(25, 25, 45, 0.5)',
              border: '1px solid rgba(0, 255, 255, 0.3)',
              borderRadius: '8px',
              padding: '1.5rem'
            }}>
              <div style={{ marginBottom: '1rem' }}>
                <h3 style={{ color: '#00ffff', marginBottom: '0.5rem' }}>Current Datasets</h3>
              </div>
              
              {datasets.map((dataset, index) => (
                <div key={index} style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  padding: '1rem',
                  backgroundColor: 'rgba(0, 0, 0, 0.2)',
                  borderRadius: '4px',
                  marginBottom: '0.5rem',
                  border: '1px solid rgba(0, 255, 255, 0.2)'
                }}>
                  <div>
                    <div style={{ color: '#ffffff', fontWeight: 'bold', marginBottom: '0.25rem' }}>
                      📊 {dataset.name}
                    </div>
                    <div style={{ color: '#b8bcc8', fontSize: '0.9rem' }}>
                      {dataset.rows?.toLocaleString() || 0} rows • {dataset.start_date} to {dataset.end_date}
                    </div>
                    <div style={{ color: '#b8bcc8', fontSize: '0.8rem' }}>
                      Created: {new Date(dataset.created_at).toLocaleDateString()} • 
                      Updated: {new Date(dataset.updated_at).toLocaleDateString()}
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: '0.5rem' }}>
                    <button
                      style={{
                        backgroundColor: 'rgba(0, 255, 255, 0.2)',
                        border: '1px solid rgba(0, 255, 255, 0.3)',
                        borderRadius: '4px',
                        color: '#00ffff',
                        padding: '0.5rem',
                        cursor: 'pointer',
                        fontSize: '0.8rem'
                      }}
                    >
                      📊 View
                    </button>
                    <button
                      style={{
                        backgroundColor: 'rgba(255, 0, 128, 0.2)',
                        border: '1px solid rgba(255, 0, 128, 0.3)',
                        borderRadius: '4px',
                        color: '#ff0080',
                        padding: '0.5rem',
                        cursor: 'pointer',
                        fontSize: '0.8rem'
                      }}
                    >
                      🗑️ Delete
                    </button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div style={{
              backgroundColor: 'rgba(255, 215, 0, 0.1)',
              border: '1px solid rgba(255, 215, 0, 0.3)',
              borderRadius: '8px',
              padding: '2rem',
              textAlign: 'center',
              color: '#ffd700'
            }}>
              <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>📊</div>
              <div style={{ marginBottom: '0.5rem' }}>No datasets found</div>
              <div style={{ color: '#b8bcc8', fontSize: '0.9rem' }}>
                Upload data through the Data Upload page to get started
              </div>
            </div>
          )}
        </div>

        {/* Database Actions */}
        <div style={{
          backgroundColor: 'rgba(25, 25, 45, 0.5)',
          border: '1px solid rgba(255, 0, 128, 0.3)',
          borderRadius: '8px',
          padding: '2rem'
        }}>
          <h2 style={{ fontSize: '1.5rem', marginBottom: '1rem', color: '#ff0080' }}>⚠️ Database Actions</h2>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1rem' }}>
            <div style={{ textAlign: 'center' }}>
              <h3 style={{ color: '#ffd700', marginBottom: '1rem' }}>🔧 Maintenance</h3>
              <button
                style={{
                  backgroundColor: 'rgba(255, 215, 0, 0.2)',
                  border: '1px solid rgba(255, 215, 0, 0.3)',
                  borderRadius: '4px',
                  color: '#ffd700',
                  padding: '0.75rem 1rem',
                  cursor: 'pointer',
                  width: '100%',
                  marginBottom: '0.5rem'
                }}
                disabled={loading}
              >
                🧹 Clean Database
              </button>
              <div style={{ color: '#b8bcc8', fontSize: '0.8rem' }}>
                Remove temporary and cached data
              </div>
            </div>
            
            <div style={{ textAlign: 'center' }}>
              <h3 style={{ color: '#00ffff', marginBottom: '1rem' }}>💾 Backup</h3>
              <button
                style={{
                  backgroundColor: 'rgba(0, 255, 255, 0.2)',
                  border: '1px solid rgba(0, 255, 255, 0.3)',
                  borderRadius: '4px',
                  color: '#00ffff',
                  padding: '0.75rem 1rem',
                  cursor: 'pointer',
                  width: '100%',
                  marginBottom: '0.5rem'
                }}
                disabled={loading}
              >
                💾 Export Data
              </button>
              <div style={{ color: '#b8bcc8', fontSize: '0.8rem' }}>
                Download database backup
              </div>
            </div>
            
            <div style={{ textAlign: 'center' }}>
              <h3 style={{ color: '#ff0080', marginBottom: '1rem' }}>🗑️ Reset</h3>
              <button
                style={{
                  backgroundColor: 'rgba(255, 0, 128, 0.2)',
                  border: '1px solid rgba(255, 0, 128, 0.3)',
                  borderRadius: '4px',
                  color: '#ff0080',
                  padding: '0.75rem 1rem',
                  cursor: 'pointer',
                  width: '100%',
                  marginBottom: '0.5rem'
                }}
                disabled={loading}
              >
                🚨 Clear All Data
              </button>
              <div style={{ color: '#b8bcc8', fontSize: '0.8rem' }}>
                ⚠️ This action cannot be undone
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DatabaseManager;