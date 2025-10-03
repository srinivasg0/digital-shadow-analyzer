// frontend/src/pages/index.tsx

import Head from 'next/head';
import AnalysisTabs from '../components/AnalysisTabs';
import { CssBaseline } from '@mui/material';

export default function HomePage() {
  return (
    <>
      <Head>
        <title>Digital Shadow Analyzer</title>
        <meta name="description" content="Analyze your digital footprint for PII exposure." />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>
      <CssBaseline /> {/* This provides a consistent baseline style */}
      <main>
        <AnalysisTabs />
      </main>
    </>
  );
}
