#!/bin/bash
# Frontend startup script for TribexAlpha
# Starts the React development server

cd frontend

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

# Start development server
echo "Starting React development server..."
npm run dev