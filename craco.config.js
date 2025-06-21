module.exports = {
  devServer: {
    port: 3000,
    host: '0.0.0.0',
  },
  webpack: {
    configure: (webpackConfig) => {
      // Ignore source map warnings for plotly.js
      webpackConfig.ignoreWarnings = [
        {
          module: /plotly\.js/,
          message: /Failed to parse source map/,
        },
      ];

      // Alternative approach - disable source map loader for plotly.js
      webpackConfig.module.rules.forEach((rule) => {
        if (rule.enforce === 'pre' && rule.use) {
          rule.use.forEach((loader) => {
            if (loader.loader && loader.loader.includes('source-map-loader')) {
              loader.exclude = [/plotly\.js/];
            }
          });
        }
      });

      return webpackConfig;
    },
  },
};