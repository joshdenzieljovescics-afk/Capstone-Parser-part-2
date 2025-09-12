import React, { useState, useRef } from 'react';
import axios from 'axios';
import { Document, Page, pdfjs } from 'react-pdf';
import ReactMarkdown from 'react-markdown';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import 'react-pdf/dist/Page/TextLayer.css';
import './App.css';

// Setup for the PDF.js worker (required by react-pdf)
pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [jsonOutput, setJsonOutput] = useState(null);
  const [chunkedOutput, setChunkedOutput] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [pdfPreviewFile, setPdfPreviewFile] = useState(null);
  const [hoveredChunkIndex, setHoveredChunkIndex] = useState(null);
  const [numPages, setNumPages] = useState(null);
  const [pageDimensions, setPageDimensions] = useState({});
  const pageRefs = useRef([]);
  const [isViewerOpen, setIsViewerOpen] = useState(false);
  const [viewerTab, setViewerTab] = useState('parsed'); 
  const [chunkDisplayMode, setChunkDisplayMode] = useState('regular');

  // Handle file selection
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file && file.type === 'application/pdf') {
      setSelectedFile(file);
      setPdfPreviewFile(file);
      setJsonOutput(null);
      setChunkedOutput(null);
      setError('');
    } else {
      setError('Please select a valid PDF file.');
    }
  };

  // Handle document parsing
  const handleParse = async () => {
    if (!selectedFile) {
      setError('Please select a PDF file first.');
      return;
    }
    const formData = new FormData();
    formData.append('file', selectedFile);

    setIsLoading(true);
    setError('');
    setJsonOutput(null);

    try {
      const response = await axios.post('http://127.0.0.1:5000/parse-pdf', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setJsonOutput(response.data);
    } catch (err) {
      if (err.response) {
        setError(`Error ${err.response.status}: ${err.response.data.error || 'Server error'}`);
      } else if (err.request) {
        setError('Network Error: Could not connect to the server. Is it running?');
      } else {
        setError(`Unexpected error: ${err.message}`);
      }
    } finally {
      setIsLoading(false);
    }
  };

  // Handle chunk processing
  
  const handleChunk = async () => {
    if (!jsonOutput) {
      setError('Please parse the PDF first.');
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      // Extract filename from selectedFile
      const filename = selectedFile ? selectedFile.name : 'unknown.pdf';
      
      const response = await axios.post('http://127.0.0.1:5000/test-anchoring', {
        simplified_view: jsonOutput.simplified,
        structured: jsonOutput.structured,
        source_filename: filename  // Add this
      });

      setChunkedOutput(response.data);
    } catch (err) {
      if (err.response) {
        setError(`Error ${err.response.status}: ${err.response.data.error || 'Server error'}`);
      } else if (err.request) {
        setError('Network Error: Could not connect to the server.');
      } else {
        setError(`Unexpected error: ${err.message}`);
      }
    } finally {
      setIsLoading(false);
    }
  };

  // Handle PDF document load success
  const onDocumentLoadSuccess = ({ numPages }) => {
    setNumPages(numPages);
  };

  // Handle page render success
  const onPageRenderSuccess = (page) => {
    const pageNumber = page.pageNumber;
    const viewport = page.getViewport({ scale: 1.0 });
    setPageDimensions(prev => ({
      ...prev,
      [pageNumber]: {
        width: page.width,
        height: page.height,
        originalWidth: viewport.width,
        originalHeight: viewport.height,
      }
    }));
  };

  // Handle mouse enter on chunk
  const handleMouseEnter = (chunk, index) => {
    setHoveredChunkIndex(index);
    
    // Scroll to the page if chunk has page info
    if (chunk.metadata?.page && pageRefs.current[chunk.metadata.page - 1]) {
      pageRefs.current[chunk.metadata.page - 1].scrollIntoView({
        behavior: 'smooth',
        block: 'center'
      });
    }
  };

  return (
    <div className="container">
      <h1>PDF Grounding Tool</h1>
      <p>Upload a PDF to see its content split into "chunks" with their locations highlighted.</p>

      <div className="upload-section">
        <input type="file" accept=".pdf" onChange={handleFileChange} />
        <button onClick={handleParse} disabled={isLoading || !selectedFile}>
          {isLoading ? 'Parsing...' : 'Parse Document'}
        </button>
        <button 
          onClick={handleChunk} 
          disabled={isLoading || !jsonOutput}
        >
          {isLoading ? 'Processing...' : 'Process Chunks'}
        </button>
        <button
          onClick={() => {
            setViewerTab('parsed');
            setIsViewerOpen(true);
          }}
          disabled={!jsonOutput && !chunkedOutput}
          style={{ marginLeft: 8 }}
        >
          Open Chunk Viewer
        </button>
      </div>

      {error && <p className="error-message">{error}</p>}

      <div className="main-content-area">
        {/* PDF Preview Column */}
        <div className="pdf-preview-container">
          <h2>PDF Preview</h2>
          <div className="pdf-document-wrapper">
            {pdfPreviewFile ? (
              <Document file={pdfPreviewFile} onLoadSuccess={onDocumentLoadSuccess}>
                {Array(numPages)
                  .fill()
                  .map((_, index) => (
                  <div
                    key={`page_container_${index + 1}`}
                    ref={(el) => (pageRefs.current[index] = el)}
                    className="pdf-page-container"
                  >
                    <Page pageNumber={index + 1} width={600} onRenderSuccess={onPageRenderSuccess} />

                    {/* Overlay bounding boxes from structured */}
                    {jsonOutput?.structured
                      ?.filter((el) => el.page === index + 1 && el.box)
                      .map((el, elIndex) => {
                        const { box, page_width, page_height } = el;
                        const currentPageDimensions = pageDimensions[index + 1];
                        if (!currentPageDimensions) return null;

                        const scaleX = currentPageDimensions.width / page_width;
                        const scaleY = currentPageDimensions.height / page_height;

                        return (
                          <div
                            key={`box_${elIndex}`}
                            className={`bounding-box`}
                            style={{
                              left: box.l * scaleX + 'px',
                              top: box.t * scaleY + 'px',
                              width: (box.r - box.l) * scaleX + 'px',
                              height: (box.b - box.t) * scaleY + 'px',
                            }}
                          />
                        );
                      })}
                  </div>
                ))}
              </Document>
            ) : (
              <div className="placeholder">Select a PDF file to preview</div>
            )}
          </div>
        </div>

        {/* Parsed Output Column (with toggle) */}
        <div className="parsed-output-container">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
            <h2>Parsed Chunks</h2>
            {/* Toggle buttons for chunk display mode */}
            {chunkedOutput?.chunks?.length > 0 && (
              <div style={{ display: 'flex', gap: 8 }}>
                <button
                  onClick={() => setChunkDisplayMode('regular')}
                  style={{
                    padding: '6px 12px',
                    background: chunkDisplayMode === 'regular' ? '#007acc' : '#333',
                    color: 'white',
                    border: '1px solid #444',
                    borderRadius: 4,
                    cursor: 'pointer',
                    fontSize: 12,
                  }}
                >
                  Regular
                </button>
                <button
                  onClick={() => setChunkDisplayMode('smart')}
                  style={{
                    padding: '6px 12px',
                    background: chunkDisplayMode === 'smart' ? '#007acc' : '#333',
                    color: 'white',
                    border: '1px solid #444',
                    borderRadius: 4,
                    cursor: 'pointer',
                    fontSize: 12,
                  }}
                >
                  Smart
                </button>
              </div>
            )}
          </div>

          <div className="content-container">
            {isLoading && <p className="placeholder">Processing...</p>}
            {!isLoading && !jsonOutput && !chunkedOutput && (
              <p className="placeholder">Parsed content will appear here.</p>
            )}

            {chunkedOutput?.chunks?.map((chunk, index) => (
              <div
                key={`chunk_text_${index}`}
                onMouseEnter={() => handleMouseEnter(chunk, index)}
                onMouseLeave={() => setHoveredChunkIndex(null)}
                className={`markdown-chunk ${hoveredChunkIndex === index ? 'hovered' : ''}`}
              >
                <div className="chunk-header">
                  <strong>Page {chunk.metadata?.page || 'N/A'}</strong>
                  <span className="chunk-type">{chunk.metadata?.type}</span>
                  {chunk.metadata?.level && (
                    <span className="heading-level">H{chunk.metadata.level}</span>
                  )}
                  {/* Show additional metadata for smart mode */}
                  {chunkDisplayMode === 'smart' && chunk.metadata?.section && (
                    <span className="chunk-section" style={{ fontSize: 12, color: '#666', marginLeft: 8 }}>
                      {chunk.metadata.section}
                    </span>
                  )}
                  {chunkDisplayMode === 'smart' && Array.isArray(chunk.metadata?.tags) && chunk.metadata.tags.length > 0 && (
                    <span style={{ fontSize: 12, color: '#6aa84f', marginLeft: 8 }}>
                      {chunk.metadata.tags.join(', ')}
                    </span>
                  )}
                </div>
                
                {/* Render content based on display mode */}
                {chunkDisplayMode === 'smart' ? (
                  <ReactMarkdown>{chunk.text}</ReactMarkdown>
                ) : (
                  <p>{chunk.text}</p>
                )}

                {chunk.metadata?.style?.length > 0 && (
                  <small className="style-info">
                    Styles: {chunk.metadata.style.join(', ')}
                  </small>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Modal Viewer */}
      {isViewerOpen && (
        <div className="modal-backdrop" onClick={() => setIsViewerOpen(false)}>
          <div
            className="modal"
            onClick={(e) => e.stopPropagation()}
            style={{
              width: '80vw',
              maxWidth: 1000,
              height: '80vh',
              background: '#1e1e1e',
              color: '#ddd',
              borderRadius: 8,
              overflow: 'hidden',
              display: 'flex',
              flexDirection: 'column',
              boxShadow: '0 10px 30px rgba(0,0,0,0.5)',
            }}
          >
            <div
              className="modal-header"
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '10px 16px',
                borderBottom: '1px solid #333',
              }}
            >
              <div>
                <button
                  onClick={() => setViewerTab('parsed')}
                  className={viewerTab === 'parsed' ? 'active-tab' : ''}
                  style={{
                    marginRight: 8,
                    padding: '6px 10px',
                    background: viewerTab === 'parsed' ? '#2d2d2d' : '#1e1e1e',
                    border: '1px solid #444',
                    color: '#ddd',
                    borderRadius: 4,
                    cursor: 'pointer',
                  }}
                >
                  Parsed
                </button>
                <button
                  onClick={() => setViewerTab('smart')}
                  className={viewerTab === 'smart' ? 'active-tab' : ''}
                  style={{
                    padding: '6px 10px',
                    background: viewerTab === 'smart' ? '#2d2d2d' : '#1e1e1e',
                    border: '1px solid #444',
                    color: '#ddd',
                    borderRadius: 4,
                    cursor: 'pointer',
                  }}
                >
                  Smart
                </button>
              </div>
              <button
                onClick={() => setIsViewerOpen(false)}
                style={{
                  padding: '6px 10px',
                  background: '#1e1e1e',
                  border: '1px solid #444',
                  color: '#ddd',
                  borderRadius: 4,
                  cursor: 'pointer',
                }}
              >
                Close
              </button>
            </div>

            <div className="modal-body" style={{ flex: 1, overflow: 'auto', padding: 16 }}>
              {viewerTab === 'parsed' && (
                <>
                  {jsonOutput?.simplified ? (
                    <pre
                      style={{
                        whiteSpace: 'pre-wrap',
                        wordWrap: 'break-word',
                        background: '#111',
                        padding: 12,
                        borderRadius: 6,
                        border: '1px solid #333',
                      }}
                    >
                      {jsonOutput.simplified}
                    </pre>
                  ) : (
                    <p className="placeholder">No parsed output yet. Parse a PDF first.</p>
                  )}
                </>
              )}

              {viewerTab === 'smart' && (
                <>
                  {chunkedOutput?.chunks?.length ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                      {chunkedOutput.chunks.map((chunk, idx) => (
                        <div
                          key={`smart_chunk_${idx}`}
                          style={{
                            padding: 12,
                            background: '#111',
                            borderRadius: 6,
                            border: '1px solid #333',
                          }}
                        >
                          <div style={{ display: 'flex', gap: 8, marginBottom: 8, alignItems: 'center' }}>
                            <strong>Page {chunk.metadata?.page ?? 'N/A'}</strong>
                            <span
                              style={{
                                fontSize: 12,
                                padding: '2px 6px',
                                border: '1px solid #444',
                                borderRadius: 4,
                                background: '#222',
                              }}
                            >
                              {chunk.metadata?.type}
                            </span>
                            {chunk.metadata?.section && (
                              <span style={{ fontSize: 12, color: '#aaa' }}>{chunk.metadata.section}</span>
                            )}
                            {Array.isArray(chunk.metadata?.tags) && chunk.metadata.tags.length > 0 && (
                              <span style={{ fontSize: 12, color: '#6aa84f' }}>
                                {chunk.metadata.tags.join(', ')}
                              </span>
                            )}
                          </div>

                          <ReactMarkdown>{chunk.text || ''}</ReactMarkdown>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="placeholder">No smart chunks yet. Click "Process Chunks".</p>
                  )}
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;