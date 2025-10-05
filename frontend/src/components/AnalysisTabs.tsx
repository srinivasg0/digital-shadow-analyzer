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

interface FusionAnalysisResponse {
  fusion_score: number;
  modality_contributions: {
    visual: number;
    text: number;
    audio: number;
  };
  primary_modality?: string;
  visual_embeddings_count: number;
  text_embeddings_count: number;
  audio_embeddings_count: number;
  explanation?: string;
  contributing_factors?: Array<{
    modality: 'visual' | 'text' | 'audio';
    index: number;
    importance: number;
    region?: { bbox?: number[]; class_name?: string; confidence?: number; frame_timestamp?: number };
    text_span?: [number, number];
    source?: string;
    time_range?: [number, number];
  }>;
  token_importance?: { visual?: number[]; text?: number[]; audio?: number[] };
  error?: string;
}

interface FileAnalysisResponse {
  analysis: TextAnalysisResponse | null;
  ocr_text?: string;
  transcribed_text?: string;
  detected_documents?: Array<{
    class_id: number;
    class_name: string;
    confidence: number;
    bbox: number[];
    ocr_text: string;
    frame_timestamp?: number;
  }>;
  fusion_analysis?: FusionAnalysisResponse;
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

    // Check for detected documents
    const hasDetectedDocuments = 'detected_documents' in result && 
      result.detected_documents && 
      result.detected_documents.length > 0;

    // Check for fusion analysis
    const hasFusionAnalysis = 'fusion_analysis' in result && 
      result.fusion_analysis && 
      !result.fusion_analysis.error;

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
          {hasDetectedDocuments && (
            <>
              <Typography variant="h6" sx={{ mt: 2 }}>Detected Documents:</Typography>
              <List dense>
                {result.detected_documents!.map((doc, idx) => (
                  <ListItem key={idx}>
                    <ListItemText
                      primary={`Type: ${doc.class_name} | Confidence: ${(doc.confidence * 100).toFixed(1)}%`}
                      secondary={
                        <Box>
                          <Typography variant="body2">
                            OCR: {doc.ocr_text || 'No text detected'}
                          </Typography>
                          <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                            Box: [{doc.bbox.map(x => x.toFixed(0)).join(', ')}]
                            {doc.frame_timestamp !== undefined && ` | Time: ${doc.frame_timestamp.toFixed(1)}s`}
                          </Typography>
                        </Box>
                      }
                    />
                  </ListItem>
                ))}
              </List>
            </>
          )}
          {hasFusionAnalysis && (
            <>
              <Typography variant="h6" sx={{ mt: 2 }}>Multimodal Fusion Analysis:</Typography>
              <Paper variant="outlined" sx={{ p: 2, mt: 1 }}>
                <Typography variant="h4" color="primary" sx={{ textAlign: 'center', mb: 2 }}>
                  Fusion Score: {(result.fusion_analysis!.fusion_score * 100).toFixed(1)}%
                </Typography>
                <Typography variant="body2" sx={{ mb: 2, textAlign: 'center' }}>
                  Primary Modality: <strong>{result.fusion_analysis!.primary_modality}</strong>
                </Typography>
                <Box sx={{ display: 'flex', justifyContent: 'space-around', mb: 2 }}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography variant="h6" color="secondary">
                      {(result.fusion_analysis!.modality_contributions.visual * 100).toFixed(1)}%
                    </Typography>
                    <Typography variant="body2">Visual</Typography>
                    <Typography variant="caption">
                      ({result.fusion_analysis!.visual_embeddings_count} regions)
                    </Typography>
                  </Box>
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography variant="h6" color="secondary">
                      {(result.fusion_analysis!.modality_contributions.text * 100).toFixed(1)}%
                    </Typography>
                    <Typography variant="body2">Text</Typography>
                    <Typography variant="caption">
                      ({result.fusion_analysis!.text_embeddings_count} texts)
                    </Typography>
                  </Box>
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography variant="h6" color="secondary">
                      {(result.fusion_analysis!.modality_contributions.audio * 100).toFixed(1)}%
                    </Typography>
                    <Typography variant="body2">Audio</Typography>
                    <Typography variant="caption">
                      ({result.fusion_analysis!.audio_embeddings_count} segments)
                    </Typography>
                  </Box>
                </Box>
                {result.fusion_analysis!.explanation && (
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    <strong>Explanation:</strong> {result.fusion_analysis!.explanation}
                  </Typography>
                )}
                {result.fusion_analysis!.contributing_factors && result.fusion_analysis!.contributing_factors.length > 0 && (
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="subtitle1">Top Contributing Evidence</Typography>
                    <List dense>
                      {result.fusion_analysis!.contributing_factors.slice(0, 5).map((f, i) => (
                        <ListItem key={i}>
                          <ListItemText
                            primary={`${f.modality.toUpperCase()} • importance ${(f.importance * 100).toFixed(1)}%`}
                            secondary={
                              f.modality === 'visual'
                                ? `Box: [${f.region?.bbox?.map(x => x.toFixed(0)).join(', ')}]` + (f.region?.frame_timestamp !== undefined ? ` | Time: ${f.region?.frame_timestamp?.toFixed(1)}s` : '')
                                : f.modality === 'text'
                                ? `${f.source || 'text'} span: [${f.text_span ? `${f.text_span[0]}, ${f.text_span[1]}` : 'n/a'}]`
                                : `Time: ${f.time_range ? `${f.time_range[0].toFixed(1)}s - ${f.time_range[1].toFixed(1)}s` : 'n/a'}`
                            }
                          />
                        </ListItem>
                      ))}
                    </List>
                  </Box>
                )}
              </Paper>
            </>
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
        
        {hasDetectedDocuments && (
          <>
            <Typography variant="h6" sx={{ mt: 2 }}>Detected Documents:</Typography>
            <List dense>
              {result.detected_documents!.map((doc, idx) => (
                <ListItem key={idx}>
                  <ListItemText
                    primary={`Type: ${doc.class_name} | Confidence: ${(doc.confidence * 100).toFixed(1)}%`}
                    secondary={
                      <Box>
                        <Typography variant="body2">
                          OCR: {doc.ocr_text || 'No text detected'}
                        </Typography>
                        <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                          Box: [{doc.bbox.map(x => x.toFixed(0)).join(', ')}]
                          {doc.frame_timestamp !== undefined && ` | Time: ${doc.frame_timestamp.toFixed(1)}s`}
                        </Typography>
                      </Box>
                    }
                  />
                </ListItem>
              ))}
            </List>
          </>
        )}
        
        {hasFusionAnalysis && (
          <>
            <Typography variant="h6" sx={{ mt: 2 }}>Multimodal Fusion Analysis:</Typography>
            <Paper variant="outlined" sx={{ p: 2, mt: 1 }}>
              <Typography variant="h4" color="primary" sx={{ textAlign: 'center', mb: 2 }}>
                Fusion Score: {(result.fusion_analysis!.fusion_score * 100).toFixed(1)}%
              </Typography>
              <Typography variant="body2" sx={{ mb: 2, textAlign: 'center' }}>
                Primary Modality: <strong>{result.fusion_analysis!.primary_modality}</strong>
              </Typography>
              <Box sx={{ display: 'flex', justifyContent: 'space-around', mb: 2 }}>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="h6" color="secondary">
                    {(result.fusion_analysis!.modality_contributions.visual * 100).toFixed(1)}%
                  </Typography>
                  <Typography variant="body2">Visual</Typography>
                  <Typography variant="caption">
                    ({result.fusion_analysis!.visual_embeddings_count} regions)
                  </Typography>
                </Box>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="h6" color="secondary">
                    {(result.fusion_analysis!.modality_contributions.text * 100).toFixed(1)}%
                  </Typography>
                  <Typography variant="body2">Text</Typography>
                  <Typography variant="caption">
                    ({result.fusion_analysis!.text_embeddings_count} texts)
                  </Typography>
                </Box>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="h6" color="secondary">
                    {(result.fusion_analysis!.modality_contributions.audio * 100).toFixed(1)}%
                  </Typography>
                  <Typography variant="body2">Audio</Typography>
                  <Typography variant="caption">
                    ({result.fusion_analysis!.audio_embeddings_count} segments)
                  </Typography>
                </Box>
              </Box>
            </Paper>
          </>
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
