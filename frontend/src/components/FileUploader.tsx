// frontend/src/components/FileUploader.tsx

import React, { useState, ChangeEvent } from 'react';
import { Box, Button, Typography, Paper, CircularProgress, LinearProgress } from '@mui/material';
import { UploadFile as UploadFileIcon } from '@mui/icons-material';
import axios from 'axios';

// Define the component's props
interface FileUploaderProps {
  apiEndpoint: string;
  acceptedFileTypes: string; // e.g., "image/*,video/mp4"
  onAnalysisComplete: (data: any) => void;
  onAnalysisError: (message: string) => void;
  onAnalysisStart: () => void;
}

const FileUploader: React.FC<FileUploaderProps> = ({ apiEndpoint, acceptedFileTypes, onAnalysisComplete, onAnalysisError, onAnalysisStart }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setSelectedFile(event.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    onAnalysisStart(); // Tell the parent component we are starting
    setIsUploading(true);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(apiEndpoint, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      onAnalysisComplete(response.data);
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || `Upload failed for ${selectedFile.name}.`;
      onAnalysisError(errorMessage);
    } finally {
      setIsUploading(false);
      setSelectedFile(null); // Reset file input after upload
    }
  };

  return (
    <Paper sx={{ p: 3, textAlign: 'center' }} variant="outlined">
      <UploadFileIcon sx={{ fontSize: 48, color: 'action.active', mb: 2 }} />
      <Typography variant="h6" gutterBottom>
        Select a file to analyze
      </Typography>
      <Button variant="contained" component="label">
        Choose File
        <input
          type="file"
          accept={acceptedFileTypes}
          hidden
          onChange={handleFileChange}
        />
      </Button>
      {selectedFile && (
        <Typography sx={{ mt: 2 }} variant="body2">
          Selected: {selectedFile.name}
        </Typography>
      )}
      <Button
        variant="outlined"
        onClick={handleUpload}
        disabled={!selectedFile || isUploading}
        sx={{ mt: 2, width: '100%' }}
      >
        {isUploading ? 'Analyzing...' : 'Upload and Analyze'}
      </Button>
      {isUploading && <LinearProgress sx={{ mt: 2 }} />}
    </Paper>
  );
};

export default FileUploader;
