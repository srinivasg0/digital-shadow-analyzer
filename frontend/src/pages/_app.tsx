// frontend/src/pages/_app.tsx

import * as React from 'react';
import type { AppProps } from 'next/app';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';

// This is our global dark theme
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#90caf9',
    },
    secondary: {
      main: '#f48fb1',
    },
    background: {
      default: '#121212',
      paper: '#1e1e1e',
    },
  },
});

function MyApp({ Component, pageProps }: AppProps) {
  // We apply the theme to the entire application here
  return (
    <ThemeProvider theme={darkTheme}>
      {/* CssBaseline kicks off an elegant, consistent cross-browser baseline */}
      <CssBaseline />
      <Component {...pageProps} />
    </ThemeProvider>
  );
}

export default MyApp;
