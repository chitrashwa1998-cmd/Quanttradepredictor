module.exports = {
  devServer: {
    port: 3000,
    host: '0.0.0.0',
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
        secure: false,
      },
    },
  },
  webpack: {
    configure: (webpackConfig) => {
      // Ignore source map warnings
      webpackConfig.ignoreWarnings = [
        /Failed to parse source map/,
        {
          module: /node_modules/,
          message: /Failed to parse source map/,
        },
      ];

      // Exclude source map loader for plotly.js
      webpackConfig.module.rules.forEach((rule) => {
        if (rule.enforce === 'pre' && rule.use) {
          rule.use.forEach((loader) => {
            if (loader.loader && loader.loader.includes('source-map-loader')) {
              loader.exclude = [
                /node_modules\/plotly\.js/,
                /plotly\.js/,
                /maplibre-gl/
              ];
            }
          });
        }
      });

      return webpackConfig;
    },
  },
};