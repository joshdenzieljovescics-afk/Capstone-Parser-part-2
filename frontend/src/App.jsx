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
  const [hoveredChunk, setHoveredChunk] = useState(null);
  const [numPages, setNumPages] = useState(null);
  const [pageDimensions, setPageDimensions] = useState({});
  const pageRefs = useRef([]);
  const [isViewerOpen, setIsViewerOpen] = useState(false);
  const [viewerTab, setViewerTab] = useState('parsed');
  const chunkDisplayMode = 'smart';

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file && file.type === 'application/pdf') {
      setSelectedFile(file);
      setPdfPreviewFile(file);
      setJsonOutput(null);
      setChunkedOutput(null);
      setError('');
      setHoveredChunk(null);
    } else {
      setError('Please select a valid PDF file.');
    }
  };

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
    setHoveredChunk(null);

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

  const handleChunk = async () => {
    if (!jsonOutput) {
      setError('Please parse the PDF first.');
      return;
    }

    setIsLoading(true);
    setError('');
    setHoveredChunk(null);
    try {
      const filename = selectedFile ? selectedFile.name : 'unknown.pdf';

      const response = await axios.post('http://127.0.0.1:5000/test-anchoring', {
        simplified_view: jsonOutput.simplified,
        structured: jsonOutput.structured,
        source_filename: filename
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

  const onDocumentLoadSuccess = ({ numPages }) => {
    setNumPages(numPages);
  };

  // Improved page dimension handling
  const onPageRenderSuccess = (page) => {
    const pageNumber = page.pageNumber;
    setPageDimensions(prev => ({
      ...prev,
      [pageNumber]: {
        width: page.width,
        height: page.height,
      }
    }));
  };

  const handleMouseEnter = (chunk, index) => {
    setHoveredChunk(chunk);
  };

  const handleMouseLeave = () => {
    setHoveredChunk(null);
  };

  // Helper function to check if element is inside chunk
  const isElementInChunk = (element, chunk, scaleX, scaleY) => {
    if (!element.box || !chunk.metadata?.box) return false;
    
    const elBox = element.box;
    const chunkBox = chunk.metadata.box;
    
    // Apply scaling to element coordinates
    const scaledElLeft = elBox.l * scaleX;
    const scaledElRight = elBox.r * scaleX;
    const scaledElTop = elBox.t * scaleY;
    const scaledElBottom = elBox.b * scaleY;
    
    // Apply scaling to chunk coordinates
    const scaledChunkLeft = chunkBox.l * scaleX;
    const scaledChunkRight = chunkBox.r * scaleX;
    const scaledChunkTop = chunkBox.t * scaleY;
    const scaledChunkBottom = chunkBox.b * scaleY;
    
    // Check for overlap with tolerance
    const tolerance = 5; // pixels
    return !(
      scaledElRight < scaledChunkLeft - tolerance ||
      scaledElLeft > scaledChunkRight + tolerance ||
      scaledElBottom < scaledChunkTop - tolerance ||
      scaledElTop > scaledChunkBottom + tolerance
    );
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
                      <Page 
                        pageNumber={index + 1} 
                        width={600} 
                        onRenderSuccess={onPageRenderSuccess} 
                      />

                      {/* Improved overlay bounding boxes */}
                      {jsonOutput?.structured
                        ?.filter((el) => el.page === index + 1 && el.box)
                        .map((el, elIndex) => {
                          const { box, page_width, page_height } = el;
                          const currentPageDimensions = pageDimensions[index + 1];
                          if (!currentPageDimensions) return null;

                          // Calculate scaling factors based on rendered page size
                          const scaleX = currentPageDimensions.width / page_width;
                          const scaleY = currentPageDimensions.height / page_height;

                          // Determine if this element should be highlighted
                          let isHovered = false;
                          if (hoveredChunk?.metadata?.page === el.page) {
                            isHovered = isElementInChunk(el, hoveredChunk, scaleX, scaleY);
                          }

                          return (
                            <div
                              key={`box_${elIndex}`}
                              className={`bounding-box ${isHovered ? 'hovered' : ''}`}
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

        <div className="parsed-output-container">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
            <h2>Parsed Chunks</h2>
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
                onMouseLeave={handleMouseLeave}
                className={`markdown-chunk ${hoveredChunk === chunk ? 'hovered' : ''}`}
              >
                <div className="chunk-header">
                  <strong>Page {chunk.metadata?.page || 'N/A'}</strong>
                  <span className="chunk-type">{chunk.metadata?.type}</span>
                  {chunk.metadata?.level && (
                    <span className="heading-level">H{chunk.metadata.level}</span>
                  )}
                  {chunk.metadata?.section && (
                    <span className="chunk-section">
                      {chunk.metadata.section}
                    </span>
                  )}
                  {Array.isArray(chunk.metadata?.tags) && chunk.metadata.tags.length > 0 && (
                    <span className="chunk-tags">
                      {chunk.metadata.tags.join(', ')}
                    </span>
                  )}
                </div>

                <div style={{ whiteSpace: 'pre-line' }}>
                  <ReactMarkdown>
                    {chunk.text}
                  </ReactMarkdown>
                </div>

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

      {isViewerOpen && (
        <div className="modal-backdrop" onClick={() => setIsViewerOpen(false)}>
          <div
            className="modal"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="modal-header">
              <div>
                <button
                  onClick={() => setViewerTab('parsed')}
                  className={`modal-tab-button ${viewerTab === 'parsed' ? 'active' : ''}`}
                >
                  Parsed
                </button>
                <button
                  onClick={() => setViewerTab('smart')}
                  className={`modal-tab-button ${viewerTab === 'smart' ? 'active' : ''}`}
                >
                  Smart Chunks
                </button>
              </div>
              <button
                onClick={() => setIsViewerOpen(false)}
                className="modal-close-button"
              >
                Close
              </button>
            </div>

            <div className="modal-body">
              {viewerTab === 'parsed' && (
                <>
                  {jsonOutput?.simplified ? (
                    <pre className="modal-content-pre">
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
                    <div className="smart-chunks-container">
                      {chunkedOutput.chunks.map((chunk, idx) => (
                        <div
                          key={`smart_chunk_${idx}`}
                          className="smart-chunk-card"
                          style={{ whiteSpace: 'pre-line' }}
                        >
                          <div className="smart-chunk-header">
                            <strong>Page {chunk.metadata?.page ?? 'N/A'}</strong>
                            <span className="smart-chunk-type">
                              {chunk.metadata?.type}
                            </span>
                            {chunk.metadata?.section && (
                              <span className="smart-chunk-section">
                                {chunk.metadata.section}
                              </span>
                            )}
                            {Array.isArray(chunk.metadata?.tags) && chunk.metadata.tags.length > 0 && (
                              <span className="smart-chunk-tags">
                                {chunk.metadata.tags.join(', ')}
                              </span>
                            )}
                          </div>

                          <div style={{ whiteSpace: 'pre-line' }}>
                            <ReactMarkdown className="smart-chunk-content">{chunk.text || ''}</ReactMarkdown>
                          </div>
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