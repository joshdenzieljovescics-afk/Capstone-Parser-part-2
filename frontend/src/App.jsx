import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Document, Page, pdfjs } from 'react-pdf';
import ReactMarkdown from 'react-markdown';
import rehypeRaw from 'rehype-raw';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import 'react-pdf/dist/Page/TextLayer.css';
import UploadBox from './UploadBox';
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
  // const [hoveredChunkIndex, setHoveredChunkIndex] = useState(null);
  // const [hoveredChunk, setHoveredChunk] = useState(null);
  const [selectedChunkId, setSelectedChunkId] = useState(null);
  const [numPages, setNumPages] = useState(null);
  const [pageDimensions, setPageDimensions] = useState({});
  const pageRefs = useRef([]);
  const [isViewerOpen, setIsViewerOpen] = useState(false);
  const [viewerTab, setViewerTab] = useState('parsed'); 
  // const [chunkDisplayMode, setChunkDisplayMode] = useState('regular'); //removed
  const [chunkView, setChunkView] = useState('markdown'); 
  const [dragActive, setDragActive] = useState(false);
  const [editingChunkIndex, setEditingChunkIndex] = useState(null); // Tracks which chunk is being edited
  const [editText, setEditText] = useState(""); // Holds the text while editing
  const [highlightBox, setHighlightBox] = useState(null);
  
  const getAllBoxesForPage = (pageNumber) => {
  if (!chunkedOutput?.chunks) return [];
  const currentPageInfo = pageDimensions[pageNumber];
  if (!currentPageInfo) return [];

  const scaleX = currentPageInfo.width / currentPageInfo.originalWidth;
  const scaleY = currentPageInfo.height / currentPageInfo.originalHeight;

  return chunkedOutput.chunks
    .filter(c => c.metadata?.page === pageNumber)
    .flatMap(c => {
      if (Array.isArray(c.metadata?.boxes)) {
        return c.metadata.boxes.map(b => ({
          left: b.l * scaleX,
          top: b.t * scaleY,
          width: (b.r - b.l) * scaleX,
          height: (b.b - b.t) * scaleY,
          chunkId: c.id,
        }));
      } else if (c.metadata?.box) {
        const b = c.metadata.box;
        return [{
          left: b.l * scaleX,
          top: b.t * scaleY,
          width: (b.r - b.l) * scaleX,
          height: (b.b - b.t) * scaleY,
          chunkId: c.id,
        }];
      }
      return [];
    });
};



  // --- Drag & Drop Handlers ---
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    const dtFiles = e.dataTransfer?.files;
    if (dtFiles && dtFiles[0]) {
      handleFileChange({ target: { files: dtFiles } });
    }
  };


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

  const handleChunkClick = (chunk) => {
    // If clicking the already selected chunk, deselect it. Otherwise, select the new one.
    if (chunk.id === selectedChunkId) {
      setSelectedChunkId(null);
    } else {
      setSelectedChunkId(chunk.id);
    }
  };


  // ✅ ADD this useEffect hook to react to changes in the selected chunk
  useEffect(() => {
    // If no chunk is selected, clear the highlight
    if (!selectedChunkId) {
      setHighlightBox(null);
      return;
    }

    // Find the full chunk object from the ID
    const chunk = chunkedOutput?.chunks.find(c => c.id === selectedChunkId);
    if (!chunk) return;

    const pageNumber = chunk.metadata?.page;
    const currentPageInfo = pageDimensions[pageNumber];

    // Calculate the highlight box coordinates (logic moved from old handleMouseEnter)
    if (pageNumber && currentPageInfo) {
      const scaleX = currentPageInfo.width / currentPageInfo.originalWidth;
      const scaleY = currentPageInfo.height / currentPageInfo.originalHeight;
      let scaledBoxes = [];

      if (Array.isArray(chunk.metadata?.boxes)) {
        scaledBoxes = chunk.metadata.boxes.map(b => ({
          left: b.l * scaleX,
          top: b.t * scaleY,
          width: (b.r - b.l) * scaleX,
          height: (b.b - b.t) * scaleY,
        }));
      } else if (chunk.metadata?.box) {
        const b = chunk.metadata.box;
        scaledBoxes.push({
          left: b.l * scaleX,
          top: b.t * scaleY,
          width: (b.r - b.l) * scaleX,
          height: (b.b - b.t) * scaleY,
        });
      }
      setHighlightBox({ page: pageNumber, boxes: scaledBoxes });
    }

    // Scroll the PDF page into view
    if (pageNumber && pageRefs.current[pageNumber - 1]) {
      pageRefs.current[pageNumber - 1].scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [selectedChunkId, chunkedOutput, pageDimensions]); // Dependencies for the effect


  // When a user clicks "Edit"
  const handleEdit = (chunkText, index) => {
    setEditingChunkIndex(index);
    setEditText(chunkText);
  };

  // When a user clicks "Save"
  const handleSave = (index) => {
    // Create a new chunks array to avoid direct state mutation
    const updatedChunks = [...chunkedOutput.chunks];
    // Update the text of the specific chunk
    updatedChunks[index].text = editText;
    
    // Update the main state
    setChunkedOutput({ ...chunkedOutput, chunks: updatedChunks });
    
    // Exit editing mode
    setEditingChunkIndex(null);
    setEditText("");
  };

  // When a user clicks "Cancel"
  const handleCancel = () => {
    setEditingChunkIndex(null);
    setEditText("");
  };

  return (
    <div className="container">
      <h1>PDF Grounding Tool</h1>
      <p>Upload a PDF to see its content split into "chunks" with their locations highlighted.</p>

      <div className="upload-section">
        <div
          className={`upload-box ${dragActive ? "drag-active" : ""}`}
          onDragEnter={handleDrag}
          onDragOver={handleDrag}
          onDragLeave={handleDrag}
          onDrop={handleDrop}
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" fill="#7da4f7" viewBox="0 0 24 24">
            <path d="M12 16v-8m0 0l-4 4m4-4l4 4m6 4v2a2 2 0 01-2 2H4a2 2 0 01-2-2v-2"/>
          </svg>
          <p>select your file or drag and drop</p>
          <small>pdf files accepted</small>

          <label className="browse-btn">
            Browse
            <input type="file" accept=".pdf" hidden onChange={handleFileChange} />
          </label>

        
        </div>

        {/* Show these only AFTER a file is uploaded */}
        {selectedFile && (
          <>
            <button
              onClick={handleParse}
              disabled={isLoading}
              style={{ marginTop: "12px" }}
            >
              {isLoading ? "Uploading..." : "Upload"}
            </button>
            <button 
              onClick={handleChunk} 
              disabled={isLoading || !jsonOutput}
            >
              {isLoading ? 'Processing...' : 'Process Chunks'}
            </button>
          </>
        )}
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

                  {/* Always draw all boxes for this page */}
                  {getAllBoxesForPage(index + 1).map((b, i) => {
                    const isHovered = highlightBox && highlightBox.page === index + 1 && 
                                      highlightBox.boxes?.some(hb =>
                                        Math.abs(hb.left - b.left) < 1 &&
                                        Math.abs(hb.top - b.top) < 1
                                      );

                    return (
                      <div
                        key={`highlight_${index}_${i}`}
                        className={`highlight-box ${isHovered ? 'hovered' : ''}`}
                        style={{
                          left: `${b.left}px`,
                          top: `${b.top}px`,
                          width: `${b.width}px`,
                          height: `${b.height}px`,
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
            <h2>Parse</h2>
            {chunkedOutput?.chunks?.length > 0 && (
              <div style={{ display: 'flex', gap: 8 }}>
                <button
                  onClick={() => setChunkView('markdown')}
                  className={chunkView === 'markdown' ? 'active-tab' : ''}
                >
                  Markdown
                </button>
                <button
                  onClick={() => setChunkView('json')}
                  className={chunkView === 'json' ? 'active-tab' : ''}
                >
                  JSON
                </button>
              </div>
            )}
          </div>


          <div className="content-container">
            {isLoading && <p className="placeholder">Processing...</p>}
            {!isLoading && !jsonOutput && !chunkedOutput && (
              <p className="placeholder">Parsed content will appear here.</p>
            )}

            {chunkView === 'markdown' && chunkedOutput?.chunks?.map((chunk, index) => (
            <div
              key={`chunk_text_${index}`}
              onClick={() => handleChunkClick(chunk)}
              className={`markdown-chunk ${chunk.id === selectedChunkId ? 'selected' : ''}`}
            >
                {editingChunkIndex === index ? (
                  // --- EDITING VIEW (No changes here) ---
                  <div className="chunk-editor">
                    <textarea
                      value={editText}
                      onChange={(e) => setEditText(e.target.value)}
                      className="edit-textarea"
                    />
                    <div className="edit-controls">
                      <button className="edit-button save" onClick={() => handleSave(index)}>Save</button>
                      <button className="edit-button cancel" onClick={handleCancel}>Cancel</button>
                    </div>
                  </div>
                ) : (
                  // --- DISPLAY VIEW (Corrected) ---
                  <>
                    <div className="chunk-header">
                      <strong>Page {chunk.metadata?.page || 'N/A'}</strong>
                      <span className="chunk-type">{chunk.metadata?.type}</span>
                      
                      {/* Restored Metadata */}
                      {chunk.metadata?.level && (
                        <span className="heading-level">H{chunk.metadata.level}</span>
                      )}
                      {chunk.metadata?.section && (
                        <span className="chunk-section">{chunk.metadata.section}</span>
                      )}

                      {Array.isArray(chunk.metadata?.tags) && chunk.metadata.tags.length > 0 && (
                        <span style={{ fontSize: 12, color: '#6aa84f' }}>
                          {chunk.metadata.tags.join(', ')}
                        </span>
                      )}

                      <div className="chunk-actions">
                        <button className="edit-button" onClick={() => handleEdit(chunk.text, index)}>Edit</button>
                      </div>
                    </div>
                    <ReactMarkdown rehypePlugins={[rehypeRaw]}>{chunk.text}</ReactMarkdown>
                  </>
                )}
              </div>
            ))}

            {chunkView === 'json' && chunkedOutput?.chunks?.map((chunk, index) => {
            const safeString = JSON.stringify(chunk, null, 2);
            return (
              <pre
                key={`json_chunk_${index}`}
                onClick={() => handleChunkClick(chunk)}
                className={`json-chunk ${chunk.id === selectedChunkId ? 'selected' : ''}`}
              >
                  <code>{safeString}</code>
                </pre>
              );
            })}




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
                    {JSON.stringify(jsonOutput.simplified, null, 2)}
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

                          <ReactMarkdown rehypePlugins={[rehypeRaw]}>{chunk.text || ''}</ReactMarkdown>
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