import React, { useState, useRef, useCallback } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';
import { 
  Camera, 
  Scan, 
  ClipboardList, 
  CheckCircle2, 
  AlertTriangle, 
  Info, 
  ArrowRight,
  RefreshCcw,
  Sparkles,
  Upload,
  Brain,
  ShieldCheck,
  Search
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000/api";

const App = () => {
  const [step, setStep] = useState(0); // 0: Hero, 1: Camera, 2: Questionnaire, 3: Result, 4: OCR, 5: OCR Result
  const [loading, setLoading] = useState(false);
  const [capturedImage, setCapturedImage] = useState(null);
  const [skinResult, setSkinResult] = useState(null);
  const [ocrResult, setOcrResult] = useState(null);
  const [answers, setAnswers] = useState({ q1: '', q2: '', q3: '', q4: '', q5: '' });
  
  const resetAnalysis = () => {
    setStep(0);
    setCapturedImage(null);
    setSkinResult(null);
    setOcrResult(null);
    setAnswers({ q1: '', q2: '', q3: '', q4: '', q5: '' });
  };
  
  const webcamRef = useRef(null);

  const capture = useCallback(() => {
    const imageSrc = webcamRef.current.getScreenshot();
    setCapturedImage(imageSrc);
    setStep(2);
  }, [webcamRef]);

  const handleNext = async () => {
    if (step === 2) {
      // Analyze Skin
      setLoading(true);
      try {
        const response = await axios.post(`${API_BASE}/predict-skin`, {
          image_b64: capturedImage,
          questionnaire: answers
        });
        setSkinResult(response.data);
        setStep(3);
      } catch (err) {
        alert("Extraction failed. Using mock results for demo.");
        setSkinResult({ skin_type: 'Combination', confidence: 0.85, breakdown: {oily: 0.6, dry: 0.3, normal: 0.1} });
        setStep(3);
      }
      setLoading(false);
    }
  };

  // Compress image to reduce payload size and speed up OCR
  const compressImage = (file, maxWidth = 1200, quality = 0.7) => {
    return new Promise((resolve) => {
      const img = new Image();
      const url = URL.createObjectURL(file);
      img.onload = () => {
        const canvas = document.createElement('canvas');
        const scale = Math.min(1, maxWidth / img.width);
        canvas.width = img.width * scale;
        canvas.height = img.height * scale;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        const b64 = canvas.toDataURL('image/jpeg', quality);
        URL.revokeObjectURL(url);
        resolve(b64);
      };
      img.src = url;
    });
  };

  const onFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    setLoading(true);
    try {
      const b64 = await compressImage(file);
      const response = await axios.post(`${API_BASE}/analyze-ingredients`, {
        image_b64: b64,
        skin_type: skinResult?.skin_type?.toLowerCase() || 'normal'
      }, { timeout: 60000 });

      // Backend may return 200 with an error field
      if (response.data.error) {
        alert(`Analysis issue: ${response.data.error}`);
      } else {
        setOcrResult(response.data);
        setStep(5);
      }
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'Unknown error';
      alert(`OCR Analysis failed: ${msg}`);
    }
    setLoading(false);
    // Reset file input so same file can be re-uploaded
    e.target.value = '';
  };

  return (
    <div className="container">
      {/* Step Indicator */}
      {step > 0 && (
        <div className="step-indicator">
          {[1, 2, 3, 4].map(s => (
            <div key={s} className={`step-dot ${Math.ceil(step / 1.5) === s ? 'active' : ''}`} />
          ))}
        </div>
      )}

      {/* Hero Section */}
      {step === 0 && (
        <section className="glass-card" style={{ padding: '60px', textAlign: 'center' }}>
          <Sparkles size={64} color="#8b5cf6" style={{ marginBottom: '20px' }} />
          <h1>Skincare AI <span style={{ color: '#8b5cf6' }}>Pro</span></h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '1.2rem', marginBottom: '40px' }}>
            Personalized skin analysis and product suitability scanner using Deep Learning.
          </p>
          <div style={{ display: 'flex', justifyContent: 'center', gap: '20px' }}>
            <button className="btn-primary" onClick={() => setStep(1)}>
              Start Analysis <ArrowRight size={20} />
            </button>
          </div>

          <div className="about-grid" style={{ marginTop: '80px', display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '30px', textAlign: 'left' }}>
            <div className="glass-card" style={{ padding: '30px' }}>
              <Brain size={32} color="var(--primary)" style={{ marginBottom: '15px' }} />
              <h3 style={{ marginBottom: '10px' }}>AI Skin Analysis</h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem', lineHeight: '1.6' }}>
                Advanced deep learning models analyze your skin type from a single selfie with medical-grade precision.
              </p>
            </div>
            <div className="glass-card" style={{ padding: '30px' }}>
              <Search size={32} color="var(--accent-blue)" style={{ marginBottom: '15px' }} />
              <h3 style={{ marginBottom: '10px' }}>Ingredient Scanner</h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem', lineHeight: '1.6' }}>
                Instant OCR technology identifies every ingredient in your product label and cross-references our database.
              </p>
            </div>
            <div className="glass-card" style={{ padding: '30px' }}>
              <ShieldCheck size={32} color="var(--accent-green)" style={{ marginBottom: '15px' }} />
              <h3 style={{ marginBottom: '10px' }}>Safety Verdict</h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem', lineHeight: '1.6' }}>
                Get a personalized safety score based on your unique skin profile to avoid irritation and breakouts.
              </p>
            </div>
          </div>
        </section>
      )}

      {/* Camera Capture */}
      {step === 1 && (
        <section className="glass-card" style={{ padding: '40px', textAlign: 'center' }}>
          <h2 style={{ marginBottom: '20px' }}>Face Analysis</h2>
          <p style={{ color: 'var(--text-muted)', marginBottom: '30px' }}>Position your face within the frame in a well-lit environment.</p>
          
          <div className="webcam-container">
            <Webcam
              audio={false}
              ref={webcamRef}
              screenshotFormat="image/jpeg"
              style={{ width: '100%', height: '100%', objectFit: 'cover' }}
            />
            <div className="webcam-overlay">
              <div className="face-guide"></div>
            </div>
          </div>
          
          <button className="btn-primary" onClick={capture} style={{ margin: '30px auto' }}>
            <Camera size={20} /> Capture Face
          </button>
        </section>
      )}

      {/* Questionnaire */}
      {step === 2 && (
        <section>
          <h2 style={{ marginBottom: '30px', textAlign: 'center' }}>Personalized Skin Quiz</h2>
          <div className="glass-card question-card">
            <h3>1. Is your T-zone often shiny?</h3>
            <div style={{ display: 'flex', gap: '10px', marginTop: '15px' }}>
              <button className={`btn-option ${answers.q1 === 'yes' ? 'selected' : ''}`} onClick={() => setAnswers({...answers, q1: 'yes'})}>Yes</button>
              <button className={`btn-option ${answers.q1 === 'no' ? 'selected' : ''}`} onClick={() => setAnswers({...answers, q1: 'no'})}>No</button>
            </div>
          </div>

          <div className="glass-card question-card">
            <h3>2. Does your skin feel tight after washing?</h3>
            <div style={{ display: 'flex', gap: '10px', marginTop: '15px' }}>
              <button className={`btn-option ${answers.q2 === 'yes' ? 'selected' : ''}`} onClick={() => setAnswers({...answers, q2: 'yes'})}>Yes</button>
              <button className={`btn-option ${answers.q2 === 'no' ? 'selected' : ''}`} onClick={() => setAnswers({...answers, q2: 'no'})}>No</button>
            </div>
          </div>

          <div className="glass-card question-card">
            <h3>3. Do you have frequent breakouts?</h3>
            <div style={{ display: 'flex', gap: '10px', marginTop: '15px' }}>
              <button className={`btn-option ${answers.q3 === 'yes' ? 'selected' : ''}`} onClick={() => setAnswers({...answers, q3: 'yes'})}>Yes</button>
              <button className={`btn-option ${answers.q3 === 'no' ? 'selected' : ''}`} onClick={() => setAnswers({...answers, q3: 'no'})}>No</button>
            </div>
          </div>

          <div className="glass-card question-card">
            <h3>4. Is your skin easily irritated or sensitive?</h3>
            <div style={{ display: 'flex', gap: '10px', marginTop: '15px' }}>
              <button className={`btn-option ${answers.q4 === 'yes' ? 'selected' : ''}`} onClick={() => setAnswers({...answers, q4: 'yes'})}>Yes</button>
              <button className={`btn-option ${answers.q4 === 'no' ? 'selected' : ''}`} onClick={() => setAnswers({...answers, q4: 'no'})}>No</button>
            </div>
          </div>

          <div className="glass-card question-card">
            <h3>5. Is your skin oily in some areas and dry in others?</h3>
            <div style={{ display: 'flex', gap: '10px', marginTop: '15px' }}>
              <button className={`btn-option ${answers.q5 === 'yes' ? 'selected' : ''}`} onClick={() => setAnswers({...answers, q5: 'yes'})}>Yes</button>
              <button className={`btn-option ${answers.q5 === 'no' ? 'selected' : ''}`} onClick={() => setAnswers({...answers, q5: 'no'})}>No</button>
            </div>
          </div>

          <button className="btn-primary" onClick={handleNext} style={{ width: '100%', justifyContent: 'center' }}>
            {loading ? <div className="loader" style={{ width: '20px', height: '20px', margin: '0' }} /> : "Analyze My Skin"}
          </button>
        </section>
      )}

      {/* Skin Result */}
      {step === 3 && (
        <section className="glass-card" style={{ padding: '60px', textAlign: 'center' }}>
          <CheckCircle2 size={64} color="var(--accent-green)" style={{ marginBottom: '20px' }} />
          <h2 style={{ marginBottom: '10px' }}>Your Skin Type: <span style={{ color: 'var(--primary)' }}>{skinResult?.skin_type}</span></h2>
          <p style={{ color: 'var(--text-muted)', marginBottom: '40px' }}>Based on our deep learning model and your questionnaire.</p>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '15px', marginBottom: '40px' }}>
            {Object.entries(skinResult?.breakdown || {}).map(([key, val]) => (
              <div key={key} className="glass-card" style={{ padding: '15px' }}>
                <div style={{ fontSize: '0.8rem', opacity: 0.6, textTransform: 'uppercase' }}>{key}</div>
                <div style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>{Math.round(val * 100)}%</div>
              </div>
            ))}
          </div>

          <button className="btn-primary" onClick={() => setStep(4)} style={{ width: '100%', justifyContent: 'center' }}>
            Scan Product Ingredients <Scan size={20} />
          </button>
        </section>
      )}

      {/* OCR Scanner */}
      {step === 4 && (
        <section className="glass-card" style={{ padding: '60px', textAlign: 'center' }}>
          <Upload size={64} color="var(--primary)" style={{ marginBottom: '20px' }} />
          <h2>Upload Product Label</h2>
          <p style={{ color: 'var(--text-muted)', marginBottom: '40px' }}>Upload a clear photo of the ingredients list for analysis.</p>
          
          <input 
            type="file" 
            id="file-upload" 
            style={{ display: 'none' }} 
            onChange={onFileUpload}
            accept="image/*"
          />
          <label htmlFor="file-upload" className="btn-primary" style={{ display: 'inline-flex', cursor: 'pointer' }}>
            {loading ? <div className="loader" style={{ width: '20px', height: '20px', margin: '0' }} /> : <><Upload size={20} /> Choose File</>}
          </label>
        </section>
      )}

      {/* OCR Result */}
      {step === 5 && (
        <section className="glass-card" style={{ padding: '40px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '30px' }}>
            <h2>Ingredient Report</h2>
            <div className={`result-badge ${ocrResult.verdict === 'Generally Suitable' ? 'badge-green' : 'badge-orange'}`}>
              {ocrResult?.verdict}
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.5fr', gap: '30px' }}>
            <div>
              <div style={{ marginBottom: '20px' }}>
                <h4 style={{ color: 'var(--accent-green)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <CheckCircle2 size={16} /> Suitable ({ocrResult?.suitable?.length})
                </h4>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginTop: '10px' }}>
                  {ocrResult?.suitable?.map(ing => <span key={ing} className="btn-option" style={{ padding: '5px 10px', fontSize: '0.8rem' }}>{ing}</span>)}
                </div>
              </div>
              <div>
                <h4 style={{ color: 'var(--accent-red)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <AlertTriangle size={16} /> Avoid ({ocrResult?.harmful?.length})
                </h4>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginTop: '10px' }}>
                  {ocrResult?.harmful?.map(ing => <span key={ing} className="btn-option" style={{ padding: '5px 10px', fontSize: '0.8rem', borderColor: 'rgba(239, 68, 68, 0.3)' }}>{ing}</span>)}
                </div>
              </div>
              <div style={{ marginTop: '20px' }}>
                <h4 style={{ color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Info size={16} /> Neutral ({ocrResult?.neutral?.length})
                </h4>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginTop: '10px' }}>
                  {ocrResult?.neutral?.map(ing => <span key={ing} className="btn-option" style={{ padding: '5px 10px', fontSize: '0.8rem', opacity: 0.7 }}>{ing}</span>)}
                </div>
              </div>
            </div>
            
            <div className="glass-card" style={{ padding: '20px' }}>
              <h4 style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '15px' }}>
                <Info size={16} /> AI Verdict
              </h4>
              <p style={{ color: 'var(--text-muted)', lineHeight: '1.6' }}>
                We detected {ocrResult?.detected_count} ingredients in this product. Based on your <strong>{skinResult?.skin_type}</strong> skin type, 
                this product is {ocrResult?.verdict?.toLowerCase()}. 
                {ocrResult?.harmful?.length > 0 ? " Watch out for potential irritants." : " It looks safe to use!"}
              </p>
              <div style={{ display: 'flex', gap: '10px', marginTop: '20px' }}>
                <button className="btn-secondary" onClick={() => setStep(4)} style={{ flex: 1 }}>
                  <RefreshCcw size={16} /> Scan Another
                </button>
                <button className="btn-primary" onClick={resetAnalysis} style={{ flex: 1 }}>
                  <RefreshCcw size={16} /> New Analysis
                </button>
              </div>
            </div>
          </div>
        </section>
      )}
    </div>
  );
};

export default App;
