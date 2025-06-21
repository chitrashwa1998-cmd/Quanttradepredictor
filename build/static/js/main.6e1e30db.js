// TribexAlpha Trading Dashboard App

// React App Entry Point
(function() {
    'use strict';
    
    // Simple React-like app implementation
    function createElement(tag, props, ...children) {
        const element = document.createElement(tag);
        if (props) {
            Object.keys(props).forEach(key => {
                if (key.startsWith('on') && typeof props[key] === 'function') {
                    element.addEventListener(key.substring(2).toLowerCase(), props[key]);
                } else if (key === 'className') {
                    element.className = props[key];
                } else {
                    element.setAttribute(key, props[key]);
                }
            });
        }
        children.forEach(child => {
            if (typeof child === 'string') {
                element.appendChild(document.createTextNode(child));
            } else {
                element.appendChild(child);
            }
        });
        return element;
    }

    // API Configuration
    const API_BASE = window.location.protocol === 'https:' ? 'https://' + window.location.host : 'http://' + window.location.host;
    
    // App State
    let currentPage = 'home';
    let appState = {
        models: {},
        data: {},
        predictions: {},
        loading: false
    };

    // API Functions
    async function fetchAPI(endpoint) {
        try {
            const response = await fetch(`${API_BASE}/api${endpoint}`);
            return await response.json();
        } catch (error) {
            console.error('API Error:', error);
            return { success: false, error: error.message };
        }
    }

    // Component: Navbar
    function createNavbar() {
        const nav = createElement('nav', { className: 'navbar' },
            createElement('h1', {}, 'TribexAlpha Trading Dashboard'),
            createElement('div', { className: 'nav-links' },
                createElement('a', { 
                    href: '#', 
                    className: currentPage === 'home' ? 'nav-link active' : 'nav-link',
                    onclick: () => navigateTo('home')
                }, 'Home'),
                createElement('a', { 
                    href: '#', 
                    className: currentPage === 'upload' ? 'nav-link active' : 'nav-link',
                    onclick: () => navigateTo('upload')
                }, 'Data Upload'),
                createElement('a', { 
                    href: '#', 
                    className: currentPage === 'models' ? 'nav-link active' : 'nav-link',
                    onclick: () => navigateTo('models')
                }, 'Model Training'),
                createElement('a', { 
                    href: '#', 
                    className: currentPage === 'predictions' ? 'nav-link active' : 'nav-link',
                    onclick: () => navigateTo('predictions')
                }, 'Predictions'),
                createElement('a', { 
                    href: '#', 
                    className: currentPage === 'database' ? 'nav-link active' : 'nav-link',
                    onclick: () => navigateTo('database')
                }, 'Database')
            )
        );
        return nav;
    }

    // Component: Home Page
    function createHomePage() {
        return createElement('div', { className: 'container' },
            createElement('div', { className: 'card' },
                createElement('h2', {}, 'Welcome to TribexAlpha Trading Dashboard'),
                createElement('p', {}, 'Advanced AI-powered trading analysis platform for NIFTY 50 predictions.'),
                createElement('div', { className: 'stats-grid' },
                    createElement('div', { className: 'stat-card' },
                        createElement('div', { className: 'stat-value' }, '7'),
                        createElement('div', { className: 'stat-label' }, 'AI Models')
                    ),
                    createElement('div', { className: 'stat-card' },
                        createElement('div', { className: 'stat-value' }, '5min'),
                        createElement('div', { className: 'stat-label' }, 'Data Frequency')
                    ),
                    createElement('div', { className: 'stat-card' },
                        createElement('div', { className: 'stat-value' }, '90%+'),
                        createElement('div', { className: 'stat-label' }, 'Accuracy')
                    )
                )
            )
        );
    }

    // Component: Upload Page
    function createUploadPage() {
        return createElement('div', { className: 'container' },
            createElement('div', { className: 'card' },
                createElement('h2', {}, 'Data Upload'),
                createElement('div', { className: 'form-group' },
                    createElement('label', {}, 'Upload CSV File'),
                    createElement('input', { 
                        type: 'file', 
                        className: 'form-control',
                        accept: '.csv',
                        onchange: handleFileUpload
                    })
                ),
                createElement('button', { 
                    className: 'btn',
                    onclick: uploadData
                }, 'Upload Data')
            )
        );
    }

    // Component: Models Page
    function createModelsPage() {
        return createElement('div', { className: 'container' },
            createElement('div', { className: 'card' },
                createElement('h2', {}, 'AI Model Training'),
                createElement('p', {}, 'Train advanced machine learning models for trading predictions.'),
                createElement('button', { 
                    className: 'btn',
                    onclick: trainModels
                }, 'Train All Models'),
                createElement('div', { id: 'models-status' })
            )
        );
    }

    // Component: Predictions Page
    function createPredictionsPage() {
        return createElement('div', { className: 'container' },
            createElement('div', { className: 'card' },
                createElement('h2', {}, 'AI Predictions'),
                createElement('div', { className: 'form-group' },
                    createElement('label', {}, 'Select Model'),
                    createElement('select', { className: 'form-control', id: 'model-select' },
                        createElement('option', { value: 'direction' }, 'Direction Prediction'),
                        createElement('option', { value: 'profit_prob' }, 'Profit Probability'),
                        createElement('option', { value: 'reversal' }, 'Reversal Detection'),
                        createElement('option', { value: 'trading_signal' }, 'Trading Signal'),
                        createElement('option', { value: 'trend_sideways' }, 'Trend Analysis')
                    )
                ),
                createElement('button', { 
                    className: 'btn',
                    onclick: loadPredictions
                }, 'Load Predictions'),
                createElement('div', { id: 'predictions-chart' })
            )
        );
    }

    // Component: Database Page
    function createDatabasePage() {
        return createElement('div', { className: 'container' },
            createElement('div', { className: 'card' },
                createElement('h2', {}, 'Database Management'),
                createElement('button', { 
                    className: 'btn btn-danger',
                    onclick: clearDatabase
                }, 'Clear All Data'),
                createElement('div', { id: 'database-info' })
            )
        );
    }

    // Navigation
    function navigateTo(page) {
        currentPage = page;
        render();
    }

    // Event Handlers
    function handleFileUpload(event) {
        appState.selectedFile = event.target.files[0];
    }

    async function uploadData() {
        if (!appState.selectedFile) {
            alert('Please select a file first');
            return;
        }

        const formData = new FormData();
        formData.append('file', appState.selectedFile);

        try {
            const response = await fetch(`${API_BASE}/api/upload-data`, {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            if (result.success) {
                alert('Data uploaded successfully!');
            } else {
                alert('Upload failed: ' + result.error);
            }
        } catch (error) {
            alert('Upload error: ' + error.message);
        }
    }

    async function trainModels() {
        try {
            const response = await fetch(`${API_BASE}/api/train-models`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ models: [] })
            });
            const result = await response.json();
            if (result.success) {
                alert('Model training completed!');
                loadModelsStatus();
            } else {
                alert('Training failed: ' + result.error);
            }
        } catch (error) {
            alert('Training error: ' + error.message);
        }
    }

    async function loadPredictions() {
        const modelSelect = document.getElementById('model-select');
        const modelName = modelSelect.value;
        
        try {
            const result = await fetchAPI(`/predictions/${modelName}`);
            if (result.success) {
                displayPredictions(result);
            } else {
                alert('Failed to load predictions: ' + result.error);
            }
        } catch (error) {
            alert('Prediction error: ' + error.message);
        }
    }

    async function clearDatabase() {
        if (confirm('Are you sure you want to clear all data?')) {
            try {
                const response = await fetch(`${API_BASE}/api/database/clear-all`, {
                    method: 'DELETE'
                });
                const result = await response.json();
                if (result.success) {
                    alert('Database cleared successfully!');
                } else {
                    alert('Clear failed: ' + result.error);
                }
            } catch (error) {
                alert('Clear error: ' + error.message);
            }
        }
    }

    async function loadModelsStatus() {
        try {
            const result = await fetchAPI('/models/status');
            if (result.success) {
                displayModelsStatus(result.data);
            }
        } catch (error) {
            console.error('Error loading models status:', error);
        }
    }

    function displayModelsStatus(data) {
        const container = document.getElementById('models-status');
        if (!container) return;

        container.innerHTML = '';
        
        if (data.status === 'no_models') {
            container.appendChild(createElement('div', { className: 'alert alert-info' },
                'No trained models found. Upload data and train models to get started.'
            ));
            return;
        }

        Object.keys(data.trained_models).forEach(modelName => {
            const model = data.trained_models[modelName];
            const modelCard = createElement('div', { className: 'model-card' },
                createElement('h4', {}, model.name),
                createElement('div', { className: 'model-accuracy' }, 
                    `Accuracy: ${(model.accuracy * 100).toFixed(1)}%`
                ),
                createElement('p', {}, `Type: ${model.task_type}`)
            );
            container.appendChild(modelCard);
        });
    }

    function displayPredictions(data) {
        const container = document.getElementById('predictions-chart');
        if (!container) return;

        container.innerHTML = '';
        
        const info = createElement('div', { className: 'alert alert-info' },
            `Loaded ${data.total_predictions} predictions for ${data.model_name}. `,
            `Up: ${data.up_predictions}, Down: ${data.down_predictions}`
        );
        
        container.appendChild(info);
        
        // Simple chart representation
        const chartDiv = createElement('div', { className: 'predictions-chart' },
            createElement('h4', {}, 'Recent Predictions'),
            createElement('div', {}, data.predictions.slice(0, 10).map(pred => 
                createElement('div', { 
                    style: `padding: 5px; margin: 2px; background: ${pred.prediction ? '#e8f5e8' : '#ffeaea'}` 
                }, 
                `${pred.date}: ${pred.prediction ? 'UP' : 'DOWN'} (${(pred.confidence * 100).toFixed(1)}% confidence)`)
            ))
        );
        
        container.appendChild(chartDiv);
    }

    // Main Render Function
    function render() {
        const app = document.getElementById('root');
        if (!app) return;

        app.innerHTML = '';
        
        app.appendChild(createNavbar());
        
        switch (currentPage) {
            case 'upload':
                app.appendChild(createUploadPage());
                break;
            case 'models':
                app.appendChild(createModelsPage());
                setTimeout(loadModelsStatus, 100);
                break;
            case 'predictions':
                app.appendChild(createPredictionsPage());
                break;
            case 'database':
                app.appendChild(createDatabasePage());
                break;
            default:
                app.appendChild(createHomePage());
        }
    }

    // Initialize App
    document.addEventListener('DOMContentLoaded', function() {
        console.log('✅ Root element found in DOM');
        const root = document.getElementById('root');
        if (root) {
            console.log('✅ Root element found, mounting React app...');
            render();
            console.log('✅ React app mounted successfully');
        } else {
            console.error('❌ Root element not found');
        }
    });

    // Global navigation for links
    window.navigateTo = navigateTo;
})();