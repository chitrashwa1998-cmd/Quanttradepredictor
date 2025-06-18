const { useState, useEffect } = React;

// Main TribexAlpha App Component
function TribexAlpha() {
    const [activeTab, setActiveTab] = useState('home');
    const [marketData, setMarketData] = useState(null);
    const [predictions, setPredictions] = useState(null);
    const [modelStatus, setModelStatus] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const tabs = [
        { id: 'home', label: 'Home', icon: '🏠' },
        { id: 'upload', label: 'Data Upload', icon: '📊' },
        { id: 'training', label: 'Model Training', icon: '🤖' },
        { id: 'predictions', label: 'Predictions', icon: '🔮' },
        { id: 'backtesting', label: 'Backtesting', icon: '📈' },
        { id: 'database', label: 'Database Manager', icon: '💾' },
        { id: 'realtime', label: 'Real-time Data', icon: '⚡' }
    ];

    // Fetch data from Flask API
    const fetchData = async (endpoint) => {
        try {
            setLoading(true);
            const response = await axios.get(`/api/${endpoint}`);
            return response.data;
        } catch (err) {
            setError(`Failed to fetch ${endpoint}: ${err.message}`);
            return null;
        } finally {
            setLoading(false);
        }
    };

    // Load initial data
    useEffect(() => {
        const loadData = async () => {
            const [market, pred, models] = await Promise.all([
                fetchData('market-status'),
                fetchData('predictions'),
                fetchData('database-info')
            ]);
            
            setMarketData(market);
            setPredictions(pred);
            setModelStatus(models);
        };
        
        loadData();
        
        // Auto-refresh every 30 seconds during market hours
        const interval = setInterval(loadData, 30000);
        return () => clearInterval(interval);
    }, []);

    const renderContent = () => {
        switch (activeTab) {
            case 'home':
                return <HomeTab marketData={marketData} predictions={predictions} modelStatus={modelStatus} />;
            case 'upload':
                return <DataUploadTab />;
            case 'training':
                return <ModelTrainingTab modelStatus={modelStatus} setModelStatus={setModelStatus} />;
            case 'predictions':
                return <PredictionsTab predictions={predictions} setPredictions={setPredictions} />;
            case 'backtesting':
                return <BacktestingTab />;
            case 'database':
                return <DatabaseManagerTab />;
            case 'realtime':
                return <RealtimeDataTab marketData={marketData} setMarketData={setMarketData} />;
            default:
                return <HomeTab marketData={marketData} predictions={predictions} modelStatus={modelStatus} />;
        }
    };

    return (
        <div className="container">
            <div className="header">
                <h1>⚡ TribexAlpha</h1>
                <p>Quantitative Trading Intelligence Platform</p>
            </div>
            
            <div className="tabs">
                {tabs.map(tab => (
                    <div
                        key={tab.id}
                        className={`tab ${activeTab === tab.id ? 'active' : ''}`}
                        onClick={() => setActiveTab(tab.id)}
                    >
                        {tab.icon} {tab.label}
                    </div>
                ))}
            </div>

            <div className="content">
                {error && <div className="error">{error}</div>}
                {loading && <div className="loading">Loading...</div>}
                {renderContent()}
            </div>
        </div>
    );
}

// Home Tab Component
function HomeTab({ marketData, predictions, modelStatus }) {
    const formatTime = (timeStr) => {
        if (!timeStr) return 'Loading...';
        return new Date(timeStr).toLocaleString('en-IN', {
            timeZone: 'Asia/Kolkata',
            hour12: true,
            dateStyle: 'medium',
            timeStyle: 'short'
        });
    };

    return (
        <div>
            <h2>Trading Dashboard Overview</h2>
            
            <div className="card">
                <h3>Market Status</h3>
                {marketData ? (
                    <div>
                        <div className="metric">
                            <span className="metric-value">{marketData.current_time}</span>
                            <span className="metric-label">Current IST Time</span>
                        </div>
                        <div className="metric">
                            <span className={`metric-value ${marketData.is_open ? 'status-open' : 'status-closed'}`}>
                                {marketData.is_open ? 'OPEN' : 'CLOSED'}
                            </span>
                            <span className="metric-label">Market Status</span>
                        </div>
                        {marketData.nifty_data && (
                            <div className="metric">
                                <span className="metric-value">{marketData.nifty_data.current_price?.toFixed(2) || 'N/A'}</span>
                                <span className="metric-label">Nifty 50</span>
                            </div>
                        )}
                    </div>
                ) : (
                    <div>Loading market data...</div>
                )}
            </div>

            <div className="card">
                <h3>AI Model Status</h3>
                {modelStatus ? (
                    <div>
                        <div className="metric">
                            <span className="metric-value">{Object.keys(modelStatus.trained_models || {}).length}</span>
                            <span className="metric-label">Trained Models</span>
                        </div>
                        <div className="metric">
                            <span className="metric-value">{modelStatus.data_status === 'available' ? 'Ready' : 'No Data'}</span>
                            <span className="metric-label">Data Status</span>
                        </div>
                    </div>
                ) : (
                    <div>Loading model status...</div>
                )}
            </div>

            <div className="card">
                <h3>Latest Predictions</h3>
                {predictions && predictions.length > 0 ? (
                    <div>
                        {predictions.slice(0, 3).map((pred, idx) => (
                            <div key={idx} className="metric">
                                <span className="metric-value">{pred.signal || 'N/A'}</span>
                                <span className="metric-label">{pred.model_name} ({(pred.confidence * 100).toFixed(1)}%)</span>
                            </div>
                        ))}
                    </div>
                ) : (
                    <div>No predictions available. Train models first.</div>
                )}
            </div>
        </div>
    );
}

// Data Upload Tab Component
function DataUploadTab() {
    const [uploadStatus, setUploadStatus] = useState(null);
    const [datasets, setDatasets] = useState([]);

    const handleFileUpload = async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await axios.post('/api/upload-data', formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
            setUploadStatus({ type: 'success', message: response.data.message });
            loadDatasets();
        } catch (error) {
            setUploadStatus({ type: 'error', message: error.response?.data?.error || 'Upload failed' });
        }
    };

    const loadDatasets = async () => {
        try {
            const response = await axios.get('/api/database-info');
            setDatasets(response.data.datasets || []);
        } catch (error) {
            console.error('Failed to load datasets:', error);
        }
    };

    useEffect(() => {
        loadDatasets();
    }, []);

    return (
        <div>
            <h2>Data Upload & Management</h2>
            
            <div className="card">
                <h3>Upload OHLC Data</h3>
                <p>Upload CSV files with columns: Date/Time, Open, High, Low, Close, Volume</p>
                <input 
                    type="file" 
                    accept=".csv" 
                    onChange={handleFileUpload}
                    style={{ margin: '10px 0' }}
                />
                {uploadStatus && (
                    <div className={uploadStatus.type === 'success' ? 'success' : 'error'}>
                        {uploadStatus.message}
                    </div>
                )}
            </div>

            <div className="card">
                <h3>Available Datasets</h3>
                {datasets.length > 0 ? (
                    <div>
                        {datasets.map((dataset, idx) => (
                            <div key={idx} style={{ padding: '10px', border: '1px solid #ddd', margin: '5px 0', borderRadius: '5px' }}>
                                <strong>{dataset.name}</strong><br />
                                Rows: {dataset.rows} | Created: {new Date(dataset.created_at).toLocaleDateString()}
                            </div>
                        ))}
                    </div>
                ) : (
                    <div>No datasets available. Upload data to get started.</div>
                )}
            </div>
        </div>
    );
}

// Model Training Tab Component
function ModelTrainingTab({ modelStatus, setModelStatus }) {
    const [training, setTraining] = useState(false);
    const [trainingResults, setTrainingResults] = useState(null);

    const trainModels = async (modelNames) => {
        setTraining(true);
        try {
            const response = await axios.post('/api/train-models', { models: modelNames });
            setTrainingResults(response.data);
            setModelStatus(response.data.model_status);
        } catch (error) {
            setTrainingResults({ error: error.response?.data?.error || 'Training failed' });
        } finally {
            setTraining(false);
        }
    };

    const availableModels = [
        'direction_model',
        'volatility_model',
        'momentum_model',
        'mean_reversion_model'
    ];

    return (
        <div>
            <h2>AI Model Training</h2>
            
            <div className="card">
                <h3>Train Trading Models</h3>
                <p>Train XGBoost ensemble models for different trading strategies</p>
                
                <div style={{ margin: '20px 0' }}>
                    {availableModels.map(model => (
                        <button
                            key={model}
                            className="btn"
                            onClick={() => trainModels([model])}
                            disabled={training}
                            style={{ margin: '5px', display: 'block' }}
                        >
                            Train {model.replace('_', ' ').toUpperCase()}
                        </button>
                    ))}
                    
                    <button
                        className="btn"
                        onClick={() => trainModels(availableModels)}
                        disabled={training}
                        style={{ margin: '10px 5px', backgroundColor: '#16a34a' }}
                    >
                        Train All Models
                    </button>
                </div>

                {training && <div className="loading">Training models... This may take a few minutes.</div>}
                
                {trainingResults && (
                    <div className={trainingResults.error ? 'error' : 'success'}>
                        {trainingResults.error || 'Models trained successfully!'}
                    </div>
                )}
            </div>

            <div className="card">
                <h3>Model Status</h3>
                {modelStatus ? (
                    <div>
                        {Object.entries(modelStatus.trained_models || {}).map(([model, status]) => (
                            <div key={model} style={{ padding: '10px', border: '1px solid #ddd', margin: '5px 0', borderRadius: '5px' }}>
                                <strong>{model}</strong>: {status ? 'Trained' : 'Not Trained'}
                            </div>
                        ))}
                    </div>
                ) : (
                    <div>Loading model status...</div>
                )}
            </div>
        </div>
    );
}

// Predictions Tab Component
function PredictionsTab({ predictions, setPredictions }) {
    const [selectedModel, setSelectedModel] = useState('direction_model');
    const [loading, setLoading] = useState(false);

    const generatePredictions = async () => {
        setLoading(true);
        try {
            const response = await axios.get(`/api/predictions?model=${selectedModel}`);
            setPredictions(response.data.predictions || []);
        } catch (error) {
            console.error('Failed to generate predictions:', error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div>
            <h2>AI Trading Predictions</h2>
            
            <div className="card">
                <h3>Generate Predictions</h3>
                <select 
                    value={selectedModel} 
                    onChange={(e) => setSelectedModel(e.target.value)}
                    style={{ padding: '8px', margin: '10px 0', borderRadius: '5px' }}
                >
                    <option value="direction_model">Direction Model</option>
                    <option value="volatility_model">Volatility Model</option>
                    <option value="momentum_model">Momentum Model</option>
                    <option value="mean_reversion_model">Mean Reversion Model</option>
                </select>
                
                <button className="btn" onClick={generatePredictions} disabled={loading}>
                    Generate Predictions
                </button>
            </div>

            <div className="card">
                <h3>Latest Predictions</h3>
                {loading && <div className="loading">Generating predictions...</div>}
                
                {predictions && predictions.length > 0 ? (
                    <div style={{ overflowX: 'auto' }}>
                        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                            <thead>
                                <tr style={{ backgroundColor: '#f5f5f5' }}>
                                    <th style={{ padding: '10px', border: '1px solid #ddd' }}>Time</th>
                                    <th style={{ padding: '10px', border: '1px solid #ddd' }}>Signal</th>
                                    <th style={{ padding: '10px', border: '1px solid #ddd' }}>Confidence</th>
                                    <th style={{ padding: '10px', border: '1px solid #ddd' }}>Model</th>
                                </tr>
                            </thead>
                            <tbody>
                                {predictions.slice(0, 10).map((pred, idx) => (
                                    <tr key={idx}>
                                        <td style={{ padding: '10px', border: '1px solid #ddd' }}>
                                            {pred.timestamp || 'N/A'}
                                        </td>
                                        <td style={{ padding: '10px', border: '1px solid #ddd' }}>
                                            <span style={{ 
                                                color: pred.signal === 'BUY' ? '#22c55e' : pred.signal === 'SELL' ? '#ef4444' : '#6b7280',
                                                fontWeight: 'bold'
                                            }}>
                                                {pred.signal || 'HOLD'}
                                            </span>
                                        </td>
                                        <td style={{ padding: '10px', border: '1px solid #ddd' }}>
                                            {(pred.confidence * 100).toFixed(1)}%
                                        </td>
                                        <td style={{ padding: '10px', border: '1px solid #ddd' }}>
                                            {pred.model_name || selectedModel}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                ) : (
                    <div>No predictions available. Generate predictions first.</div>
                )}
            </div>
        </div>
    );
}

// Backtesting Tab Component
function BacktestingTab() {
    return (
        <div>
            <h2>Strategy Backtesting</h2>
            <div className="card">
                <h3>Backtest Results</h3>
                <p>Backtesting functionality will be implemented here.</p>
                <p>This will include strategy performance metrics, drawdown analysis, and risk assessment.</p>
            </div>
        </div>
    );
}

// Database Manager Tab Component
function DatabaseManagerTab() {
    const [dbInfo, setDbInfo] = useState(null);

    const loadDatabaseInfo = async () => {
        try {
            const response = await axios.get('/api/database-info');
            setDbInfo(response.data);
        } catch (error) {
            console.error('Failed to load database info:', error);
        }
    };

    useEffect(() => {
        loadDatabaseInfo();
    }, []);

    return (
        <div>
            <h2>Database Management</h2>
            
            <div className="card">
                <h3>Database Status</h3>
                {dbInfo ? (
                    <div>
                        <div className="metric">
                            <span className="metric-value">{dbInfo.datasets?.length || 0}</span>
                            <span className="metric-label">Datasets</span>
                        </div>
                        <div className="metric">
                            <span className="metric-value">{dbInfo.models?.length || 0}</span>
                            <span className="metric-label">Trained Models</span>
                        </div>
                        <div className="metric">
                            <span className="metric-value">{dbInfo.predictions?.length || 0}</span>
                            <span className="metric-label">Stored Predictions</span>
                        </div>
                    </div>
                ) : (
                    <div>Loading database information...</div>
                )}
            </div>
        </div>
    );
}

// Real-time Data Tab Component
function RealtimeDataTab({ marketData, setMarketData }) {
    const [refreshing, setRefreshing] = useState(false);

    const refreshData = async () => {
        setRefreshing(true);
        try {
            const response = await axios.get('/api/realtime-nifty');
            setMarketData(response.data);
        } catch (error) {
            console.error('Failed to refresh data:', error);
        } finally {
            setRefreshing(false);
        }
    };

    return (
        <div>
            <h2>Real-time Market Data</h2>
            
            <div className="card">
                <h3>Nifty 50 Live Data</h3>
                <button className="btn" onClick={refreshData} disabled={refreshing}>
                    {refreshing ? 'Refreshing...' : 'Refresh Data'}
                </button>
                
                {marketData && marketData.nifty_data ? (
                    <div>
                        <div className="metric">
                            <span className="metric-value">{marketData.nifty_data.current_price?.toFixed(2) || 'N/A'}</span>
                            <span className="metric-label">Current Price</span>
                        </div>
                        <div className="metric">
                            <span className="metric-value" style={{ 
                                color: (marketData.nifty_data.change || 0) >= 0 ? '#22c55e' : '#ef4444' 
                            }}>
                                {marketData.nifty_data.change?.toFixed(2) || 'N/A'}
                            </span>
                            <span className="metric-label">Change</span>
                        </div>
                        <div className="metric">
                            <span className="metric-value">{marketData.nifty_data.volume || 'N/A'}</span>
                            <span className="metric-label">Volume</span>
                        </div>
                    </div>
                ) : (
                    <div>Loading real-time data...</div>
                )}
            </div>
        </div>
    );
}

// Render the app
ReactDOM.render(<TribexAlpha />, document.getElementById('root'));