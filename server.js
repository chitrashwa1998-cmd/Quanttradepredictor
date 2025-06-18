const express = require('express');
const cors = require('cors');
const path = require('path');
const { spawn } = require('child_process');

const app = express();
const PORT = 5000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('.'));

// Store latest data
let latestData = {
    niftyData: null,
    predictions: null,
    lastUpdate: null
};

// Utility to get IST time
const getISTTime = () => {
    const now = new Date();
    const istOffset = 5.5 * 60; // IST is UTC+5:30
    const utc = now.getTime() + (now.getTimezoneOffset() * 60000);
    return new Date(utc + (istOffset * 60000));
};

// Check if market is open
const isMarketOpen = () => {
    const ist = getISTTime();
    const hour = ist.getHours();
    const minute = ist.getMinutes();
    const day = ist.getDay(); // 0 = Sunday, 6 = Saturday
    
    // Market open: Monday-Friday, 9:15 AM - 3:30 PM IST
    const isWeekday = day >= 1 && day <= 5;
    const isMarketHours = (hour > 9 || (hour === 9 && minute >= 15)) && 
                         (hour < 15 || (hour === 15 && minute <= 30));
    
    return isWeekday && isMarketHours;
};

// Function to call Python backend for real data
const fetchRealNiftyData = async () => {
    return new Promise((resolve, reject) => {
        const pythonProcess = spawn('python', ['-c', `
import sys
import os
sys.path.append('.')
try:
    from utils.realtime_data import IndianMarketData
    market_data = IndianMarketData()
    current_data = market_data.get_current_price("^NSEI")
    if current_data:
        print(f"{current_data['price']},{current_data['change']},{current_data['change_percent']},{current_data['volume']},{current_data['high']},{current_data['low']},{current_data['open']}")
    else:
        print("ERROR")
except Exception as e:
    print(f"ERROR: {str(e)}")
`]);

        let data = '';
        pythonProcess.stdout.on('data', (chunk) => {
            data += chunk.toString();
        });

        pythonProcess.on('close', (code) => {
            const output = data.trim();
            if (output.startsWith('ERROR') || code !== 0) {
                reject(new Error('Failed to fetch real data'));
                return;
            }

            try {
                const values = output.split(',');
                if (values.length >= 7) {
                    resolve({
                        price: parseFloat(values[0]),
                        change: parseFloat(values[1]),
                        changePercent: parseFloat(values[2]),
                        volume: parseInt(values[3]),
                        high: parseFloat(values[4]),
                        low: parseFloat(values[5]),
                        open: parseFloat(values[6])
                    });
                } else {
                    reject(new Error('Invalid data format'));
                }
            } catch (error) {
                reject(error);
            }
        });
    });
};

// Function to get AI predictions from Python models
const fetchRealPredictions = async () => {
    return new Promise((resolve, reject) => {
        const pythonProcess = spawn('python', ['-c', `
import sys
import os
sys.path.append('.')
try:
    from models.xgboost_models import QuantTradingModels
    from utils.database_adapter import get_trading_database
    
    # Load models
    models = QuantTradingModels()
    db = get_trading_database()
    
    # Try to load existing data for predictions
    data = db.load_ohlc_data()
    if data is not None and len(data) > 100:
        # Get latest predictions
        latest_data = data.tail(1)
        features = models.prepare_features(data.tail(100))
        if len(features) > 0:
            latest_features = features.tail(1)
            
            # Get predictions from available models
            direction_pred, direction_conf = models.predict('direction', latest_features)
            price_pred, price_conf = models.predict('price_target', latest_features)
            trend_pred, trend_conf = models.predict('trend', latest_features)
            
            direction = "UP" if direction_pred[0] > 0.5 else "DOWN"
            trend = "TRENDING" if trend_pred[0] > 0.5 else "SIDEWAYS"
            
            # Calculate volatility from recent data
            returns = data['Close'].pct_change().tail(20).std()
            volatility = "HIGH" if returns > 0.02 else "MEDIUM" if returns > 0.01 else "LOW"
            
            print(f"{direction},{direction_conf[0]:.3f},{price_pred[0]:.2f},{price_conf[0]:.3f},{trend},{trend_conf[0]:.3f},{volatility}")
        else:
            print("ERROR: No features available")
    else:
        print("ERROR: No data available")
except Exception as e:
    print(f"ERROR: {str(e)}")
`]);

        let data = '';
        pythonProcess.stdout.on('data', (chunk) => {
            data += chunk.toString();
        });

        pythonProcess.on('close', (code) => {
            const output = data.trim();
            if (output.startsWith('ERROR') || code !== 0) {
                reject(new Error('Failed to fetch predictions'));
                return;
            }

            try {
                const values = output.split(',');
                if (values.length >= 7) {
                    resolve({
                        direction: values[0],
                        directionConfidence: parseFloat(values[1]),
                        priceTarget: parseFloat(values[2]),
                        targetConfidence: parseFloat(values[3]),
                        trend: values[4],
                        trendConfidence: parseFloat(values[5]),
                        volatility: values[6]
                    });
                } else {
                    reject(new Error('Invalid prediction format'));
                }
            } catch (error) {
                reject(error);
            }
        });
    });
};

// API Routes
app.get('/api/nifty-data', async (req, res) => {
    try {
        const data = await fetchRealNiftyData();
        latestData.niftyData = data;
        latestData.lastUpdate = new Date();
        res.json({
            success: true,
            data: data,
            timestamp: latestData.lastUpdate
        });
    } catch (error) {
        console.error('Error fetching Nifty data:', error);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

app.get('/api/predictions', async (req, res) => {
    try {
        const predictions = await fetchRealPredictions();
        latestData.predictions = predictions;
        res.json({
            success: true,
            data: predictions,
            timestamp: new Date()
        });
    } catch (error) {
        console.error('Error fetching predictions:', error);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

app.get('/api/market-status', (req, res) => {
    const ist = getISTTime();
    res.json({
        success: true,
        data: {
            isOpen: isMarketOpen(),
            currentTime: ist.toISOString(),
            istTime: ist.toLocaleString('en-IN', { timeZone: 'Asia/Kolkata' })
        }
    });
});

app.get('/api/all-data', async (req, res) => {
    try {
        const [niftyData, predictions] = await Promise.all([
            fetchRealNiftyData().catch(e => null),
            fetchRealPredictions().catch(e => null)
        ]);

        res.json({
            success: true,
            data: {
                niftyData,
                predictions,
                marketStatus: {
                    isOpen: isMarketOpen(),
                    currentTime: getISTTime().toISOString()
                }
            },
            timestamp: new Date()
        });
    } catch (error) {
        console.error('Error fetching all data:', error);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// Serve React app
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// Start server
app.listen(PORT, '0.0.0.0', () => {
    console.log(`Trading Dashboard Server running on http://0.0.0.0:${PORT}`);
    console.log(`Access your dashboard at: http://0.0.0.0:${PORT}`);
});

// Auto-refresh data every 30 seconds during market hours
setInterval(async () => {
    if (isMarketOpen()) {
        try {
            const [niftyData, predictions] = await Promise.all([
                fetchRealNiftyData().catch(e => null),
                fetchRealPredictions().catch(e => null)
            ]);
            
            if (niftyData) latestData.niftyData = niftyData;
            if (predictions) latestData.predictions = predictions;
            latestData.lastUpdate = new Date();
            
            console.log(`Data updated at ${getISTTime().toLocaleTimeString('en-IN')}`);
        } catch (error) {
            console.error('Auto-refresh failed:', error);
        }
    }
}, 30000);