import React, { useState } from "react";
import { CloudUpload } from "lucide-react"; // icon library (already included if you use shadcn/lucide-react)

const UploadBox = ({ onFileSelect }) => {
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      onFileSelect(e.target.files[0]);
    }
  };

  return (
    <div
      className={`upload-box ${dragActive ? "drag-active" : ""}`}
      onDragEnter={handleDrag}
      onDragOver={handleDrag}
      onDragLeave={handleDrag}
      onDrop={handleDrop}
    >
      <CloudUpload size={40} color="#7da4f7" />
      <p>select your file or drag and drop</p>
      <small>pdf files accepted</small>
      <label className="browse-btn">
        Browse
        <input type="file" accept=".pdf" hidden onChange={handleChange} />
      </label>
    </div>
  );
};

export default UploadBox;
