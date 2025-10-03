// frontend/src/components/AnalysisTabs.tsx (Corrected Version)

import React, { useState } from 'react';
import {
  Box,
  Tabs,
  Tab,
  TextField,
  Button,
  Typography,
  Paper,
  Container,
  LinearProgress,
  styled,
  List,
  Grid,
  ListItem,
  ListItemText,
} from '@mui/material';
import {
  UploadFile as UploadFileIcon,
  Description as DescriptionIcon,
} from '@mui/icons-material';
import axios from 'axios';
import FileUploader from './FileUploader';

// Define the structures of our API responses
interface TextAnalysisResponse {
  inputtext: string;
  sentiment: { label: string; score: number };
  evidence: { label?: string; type?: string; text: string }[];
  score: { rawscore: number; exposurescore: number };
}

interface FileAnalysisResponse {
  analysis: TextAnalysisResponse | null;
  ocr_text?: string;
  transcribed_text?: string;
}

const StyledTabPanel = styled('div')({ padding: 24 });

export default function AnalysisTabs() {
  const [currentTab, setCurrentTab] = useState(0);
  const [textInput, setTextInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<
    TextAnalysisResponse | FileAnalysisResponse | null
  >(null);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setCurrentTab(newValue);
    setAnalysisResult(null);
    setError(null);
    setTextInput('');
  };

  const handleTextFileRead = (file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target?.result as string;
      setTextInput(text);
    };
    reader.readAsText(file);
  };

  const handleAnalysisStart = () => {
    setIsLoading(true);
    setError(null);
    setAnalysisResult(null);
  };
  const handleAnalysisComplete = (data: any) => {
    setAnalysisResult(data);
    setIsLoading(false);
  };
  const handleAnalysisError = (message: string) => {
    setError(message);
    setIsLoading(false);
  };

  const handleAnalyzeText = async () => {
    handleAnalysisStart();
    try {
      const response = await axios.post('http://127.0.0.1:8000/analyze-text/', {
        text: textInput,
      });
      handleAnalysisComplete(response.data);
    } catch (err: any) {
      handleAnalysisError(
        err.response?.data?.detail || 'An unknown error occurred.'
      );
    }
  };

  // RENDER FUNCTION
  const renderResults = (result: TextAnalysisResponse | FileAnalysisResponse) => {
    const analysisData = 'analysis' in result ? result.analysis : result;

    // First, display the transcribed or OCR text if it exists
    const extraText =
      'transcribed_text' in result && result.transcribed_text
        ? `Transcribed Text: ${result.transcribed_text}`
        : 'ocr_text' in result && result.ocr_text
        ? `OCR Text: ${result.ocr_text}`
        : null;

    if (!analysisData) {
      return (
        <>
          {extraText && (
            <Typography
              variant="body2"
              sx={{ mt: 2, fontStyle: 'italic' }}
            >
              {extraText}
            </Typography>
          )}
          <Typography sx={{ mt: 2 }}>
            Analysis could not be performed on the text content.
          </Typography>
        </>
      );
    }

    return (
      <>
        {extraText && (
          <Typography
            variant="body2"
            sx={{ mt: 2, mb: 2, fontStyle: 'italic', color: 'text.secondary' }}
          >
            {extraText}
          </Typography>
        )}
        <Grid container spacing={2}>
          <Grid item xs={12} md={4}>
            <Paper variant="outlined" sx={{ p: 2, textAlign: 'center' }}>
              <Typography variant="overline">Exposure Score</Typography>
              <Typography variant="h3">
                {analysisData.score.exposurescore}
              </Typography>
              <LinearProgress
                variant="determinate"
                value={analysisData.score.exposurescore}
                sx={{ height: 10, borderRadius: 5, mt: 1 }}
              />
            </Paper>
          </Grid>
          <Grid item xs={12} md={8}>
            <Paper variant="outlined" sx={{ p: 2 }}>
              <Typography variant="overline">Details</Typography>
              <Typography>
                Sentiment: <strong>{analysisData.sentiment.label}</strong> (
                {(analysisData.sentiment.score * 100).toFixed(1)}% confidence)
              </Typography>
              <Box sx={{ mt: 2 }}>
                <Typography variant="h6">Evidence Found:</Typography>
                <List dense>
                  {analysisData.evidence.length > 0 ? (
                    analysisData.evidence.map((item, index) => (
                      <ListItem key={index} disablePadding>
                        <ListItemText
                          primary={item.text}
                          secondary={item.label || item.type}
                        />
                      </ListItem>
                    ))
                  ) : (
                    <ListItem>
                      <ListItemText primary="No PII found." />
                    </ListItem>
                  )}
                </List>
              </Box>
            </Paper>
          </Grid>
        </Grid>
      </>
    );
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography
        variant="h3"
        component="h1"
        gutterBottom
        align="center"
        sx={{ fontWeight: 'bold' }}
      >
        Digital Shadow Analyzer
      </Typography>
      <Paper elevation={6}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={currentTab} onChange={handleTabChange} centered>
            <Tab label="Text" />
            <Tab label="Image" />
            <Tab label="Video" />
            <Tab label="Audio" />
          </Tabs>
        </Box>

        <StyledTabPanel hidden={currentTab !== 0}>
          <TextField
            label="Paste your text here..."
            multiline
            fullWidth
            rows={10}
            value={textInput}
            onChange={(e) => setTextInput(e.target.value)}
          />
          <Box
            sx={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              mt: 2,
            }}
          >
            <Button
              variant="contained"
              onClick={handleAnalyzeText}
              disabled={isLoading || !textInput.trim()}
            >
              {isLoading ? 'Analyzing...' : 'Analyze Pasted Text'}
            </Button>
            <Button component="label" startIcon={<DescriptionIcon />}>
              Or Upload .txt File
              <input
                type="file"
                accept=".txt"
                hidden
                onChange={(e) =>
                  e.target.files && handleTextFileRead(e.target.files[0])
                }
              />
            </Button>
          </Box>
        </StyledTabPanel>

        <StyledTabPanel hidden={currentTab !== 1}>
          <FileUploader
            apiEndpoint="http://127.0.0.1:8000/analyze-image/"
            acceptedFileTypes="image/*"
            onAnalysisStart={handleAnalysisStart}
            onAnalysisComplete={handleAnalysisComplete}
            onAnalysisError={handleAnalysisError}
          />
        </StyledTabPanel>

        <StyledTabPanel hidden={currentTab !== 2}>
          <FileUploader
            apiEndpoint="http://127.0.0.1:8000/analyze-video/"
            acceptedFileTypes="video/*"
            onAnalysisStart={handleAnalysisStart}
            onAnalysisComplete={handleAnalysisComplete}
            onAnalysisError={handleAnalysisError}
          />
        </StyledTabPanel>

        <StyledTabPanel hidden={currentTab !== 3}>
          <FileUploader
            apiEndpoint="http://127.0.0.1:8000/analyze-audio/"
            acceptedFileTypes="audio/*"
            onAnalysisStart={handleAnalysisStart}
            onAnalysisComplete={handleAnalysisComplete}
            onAnalysisError={handleAnalysisError}
          />
        </StyledTabPanel>
      </Paper>

      {isLoading && <LinearProgress sx={{ mt: 4 }} />}
      {error && (
        <Typography
          color="error"
          sx={{
            mt: 2,
            p: 2,
            backgroundColor: 'rgba(255, 0, 0, 0.1)',
            borderRadius: 1,
          }}
          align="center"
        >
          {error}
        </Typography>
      )}
      {analysisResult && !isLoading && (
        <Box sx={{ mt: 4 }}>
          <Typography variant="h5" gutterBottom>
            Analysis Results
          </Typography>
          {renderResults(analysisResult)}
        </Box>
      )}
    </Container>
  );
}
