import React, { useState, useRef } from 'react';
import './tokkatot-design.css';
import './assets/material-symbols.css';
import './assets/mi-sans.css';
import logo from './assets/tokkatot logo-02.png';

const API_URL = '/api/v1/verify';

// Custom SVG Glyph Component
const Glyph = ({ type, className = "", size = 40, color = "currentColor" }) => {
  const glyphs = {
    chicken: (
      <svg width={size} height={size} viewBox="0 0 40 40" fill="none">
        <circle cx="20" cy="20" r="18" fill="#FFBA49" fillOpacity="0.2" stroke="#FFBA49" strokeWidth="2"/>
        <path d="M20 12C18 12 16 14 16 16C16 18 18 20 20 20C22 20 24 18 24 16C24 14 22 12 20 12Z" fill="#FFBA49"/>
        <path d="M12 28C12 24 15 22 20 22C25 22 28 24 28 28" stroke="#FFBA49" strokeWidth="2" strokeLinecap="round"/>
      </svg>
    ),
    ai: (
      <svg width={size} height={size} viewBox="0 0 40 40" fill="none">
        <rect x="10" y="10" width="20" height="20" rx="6" stroke="#20A39E" strokeWidth="2"/>
        <circle cx="20" cy="20" r="4" fill="#20A39E"/>
        <path d="M20 6V10M20 30V34M6 20H10M30 20H34" stroke="#20A39E" strokeWidth="2" strokeLinecap="round"/>
      </svg>
    ),
    cloud: (
      <svg width={size} height={size} viewBox="0 0 40 40" fill="none">
        <path d="M30 28C33.3137 28 36 25.3137 36 22C36 18.6863 33.3137 16 30 16C29.65 16 29.31 16.03 28.98 16.09C27.93 12.56 24.64 10 20.73 10C16.14 10 12.33 13.43 11.59 17.92C8.42 18.42 6 21.14 6 24.43C6 28.06 8.94 31 12.57 31H30" stroke="#20A39E" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      </svg>
    ),
    shield: (
      <svg width={size} height={size} viewBox="0 0 40 40" fill="none">
        <path d="M20 6L8 11V19C8 26.06 13.12 32.66 20 34C26.88 32.66 32 26.06 32 19V11L20 6Z" fill="#20A39E" fillOpacity="0.1" stroke="#20A39E" strokeWidth="2" strokeLinejoin="round"/>
        <path d="M15 20L18.5 23.5L25 17" stroke="#20A39E" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      </svg>
    )
  };
  return <div className={`interactive-glyph ${className}`}>{glyphs[type] || null}</div>;
};

const VisualMetric = ({ label, value, color }) => (
  <div className="visual-metric">
    <div className="metric-label">
      <span>{label}</span>
      <span>{(value * 100).toFixed(1)}%</span>
    </div>
    <div className="progress-bar-bg">
      <div 
        className="progress-bar-fill" 
        style={{ width: `${value * 100}%`, backgroundColor: color }}
      ></div>
    </div>
  </div>
);

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
      setError('');
    }
  };

  const handleSubmit = async (e) => {
    if (e) e.preventDefault();
    if (!file) {
      setError('Please select an image first.');
      return;
    }
    setLoading(true);
    setError('');
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await fetch(API_URL, {
        method: 'POST',
        body: formData
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError('Analysis failed: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  // Handler to reset for new upload
  const handleNewUpload = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError('');
    fileInputRef.current.value = '';
  };

  return (
    <>
      {/* Floating Background */}
      <div className="tokkatot-floating-bg">
        {[...Array(6)].map((_, i) => (
          <div key={i} className="floating-item" style={{
            top: `${Math.random() * 80}%`,
            left: `${Math.random() * 80}%`,
            animationDelay: `${i * 2}s`,
            animationDuration: `${10 + Math.random() * 10}s`
          }}>
            <Glyph type={['chicken', 'ai', 'cloud', 'shield'][i % 4]} size={60 + Math.random() * 40} />
          </div>
        ))}
      </div>

      <nav className="tokkatot-navbar">
        <div className="tokkatot-nav-brand">
          <img src={logo} alt="Tokkatot AI" className="tokkatot-logo" />
          <span>Tokkatot AI</span>
        </div>
        <div style={{fontSize: '0.7em', opacity: 0.7, fontWeight: 1200}}>Edge Intelligence for Poultry Health</div>
      </nav>

      <main className="tokkatot-container">
        <div className="tokkatot-main-card">
          <h1 className="tokkatot-title">Tokkatot Poultry Disease Detection System</h1>
          <p className="tokkatot-subtitle">Upload fecal images for instant AI-powered chicken fecal assessment</p>

          {/* If result exists, show result and new upload button */}
          {result ? (
            <>
              <div className="tokkatot-result-card">
                <div className="result-header">
                  <div>
                    <h3 style={{margin: 0, fontSize: '1.4em'}}>{result.classification}</h3>
                    <p style={{margin: '4px 0 0 0', color: '#666', fontSize: '0.9em'}}>Primary Diagnosis</p>
                  </div>
                  <div className={`status-badge ${result.classification === 'Healthy' ? 'status-healthy' : 'status-disease'}`}> 
                    <span className="material-symbols">{result.classification === 'Healthy' ? 'check_circle' : 'warning'}</span>
                    {result.risk_level} Risk
                  </div>
                </div>

                <div style={{marginBottom: '25px', padding: '15px', background: '#f8fafc', borderRadius: '12px'}}>
                  <strong style={{display:'block', marginBottom: '5px', color: '#1e293b'}}>Recommended Action:</strong>
                  <span style={{color: '#475569'}}>{result.action}</span>
                </div>

                <h4 style={{marginBottom: '15px', color: '#1e293b'}}>Model Confidence Breakdown</h4>
                <VisualMetric 
                  label="EfficientNetB0 (Texture Analysis)" 
                  value={result.models.efficientnet.confidence} 
                  color="#20a39e" 
                />
                <VisualMetric 
                  label="DenseNet121 (Pattern Recognition)" 
                  value={result.models.densenet.confidence} 
                  color="#ffba49" 
                />
              </div>
              <div style={{marginTop: '30px', textAlign: 'center'}}>
                <button className="tokkatot-btn" onClick={handleNewUpload}>
                  Upload Another Image
                </button>
              </div>
            </>
          ) : (
            <>
              <div 
                className="tokkatot-upload-zone"
                onClick={() => fileInputRef.current.click()}
              >
                {preview ? (
                  <img src={preview} alt="Preview" style={{maxHeight: '200px', borderRadius: '12px'}} />
                ) : (
                  <>
                    <div style={{fontSize: '3em', marginBottom: '10px'}}>📸</div>
                    <div style={{fontWeight: 600, color: '#4a5568'}}>Click to browse or drag & drop</div>
                    <div style={{fontSize: '0.85em', color: '#a0aec0', marginTop: '5px'}}>Supports JPG, PNG</div>
                  </>
                )}
                <input 
                  type="file" 
                  ref={fileInputRef}
                  onChange={handleFileChange} 
                  className="tokkatot-input-hidden" 
                  accept="image/*"
                />
              </div>

              <button 
                className="tokkatot-btn" 
                onClick={handleSubmit} 
                disabled={loading || !file}
              >
                {loading ? 'Analyzing Neural Networks...' : 'Start Health Check'}
              </button>

              {error && <div className="tokkatot-error" style={{marginTop:'20px'}}>{error}</div>}
            </>
          )}
        </div>

        <div className="info-grid">
          <div className="info-card">
            <Glyph type="ai" size={32} />
            <h4 style={{margin: '10px 0 5px'}}>Ensemble AI</h4>
            <p style={{fontSize: '0.85em', color: '#64748b'}}>Dual-model verification for 98.4% accuracy</p>
          </div>
          <div className="info-card">
            <Glyph type="cloud" size={32} />
            <h4 style={{margin: '10px 0 5px'}}>Real-time Sync</h4>
            <p style={{fontSize: '0.85em', color: '#64748b'}}>Instant synchronization with farm cloud</p>
          </div>
          <div className="info-card">
            <Glyph type="shield" size={32} />
            <h4 style={{margin: '10px 0 5px'}}>Biosecurity</h4>
            <p style={{fontSize: '0.85em', color: '#64748b'}}>Early warning system for disease control</p>
          </div>
        </div>

        <footer className="tokkatot-footer">
          &copy; 2026 Tokkatot • Sustainable Poultry Solutions | <a href="https://tokkatot.aztrolabe.com/" rel="noopener noreferrer" target="_blank">tokkatot.aztrolabe.com</a> | <a href="https://github.com/SirOsborn/tokkatot_ai/" target="_blank" rel="noopener noreferrer">GitHub</a>
        </footer>
      </main>
    </>
  );
}

export default App;
