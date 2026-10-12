import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Waves, Upload, ScanSearch, Image as ImageIcon, Activity,
  Download, ShieldCheck, Github, Cpu, LoaderCircle
} from "lucide-react";
import "./style.css";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  function chooseFile(f) {
    if (!f) return;
    if (!f.type.startsWith("image/")) {
      setError("Please select an image file.");
      return;
    }
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError("");
  }

  async function process() {
    if (!file) return;
    setBusy(true);
    setError("");
    try {
      const body = new FormData();
      body.append("file", file);
      const response = await fetch(`${API}/api/process`, {
        method: "POST", body
      });
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.detail || `API error ${response.status}`);
      }
      setResult(await response.json());
    } catch (e) {
      setError(`${e.message}. Check that the backend is running and VITE_API_URL is correct.`);
    } finally {
      setBusy(false);
    }
  }

  function downloadBase64(base64, name) {
    const a = document.createElement("a");
    a.href = `data:image/png;base64,${base64}`;
    a.download = name;
    a.click();
  }

  const imageUrl = (data) => `data:image/png;base64,${data}`;

  return <div className="app">
    <header className="topbar">
      <div className="brand"><span className="logo"><Waves /></span>
        <div><strong>EURS <span>AI STUDIO</span></strong>
          <small>UNDERWATER VISION LAB</small></div>
      </div>
      <nav><a href="#workspace">Workspace</a><a href="#metrics">Analysis</a>
        <a href="https://www.macvi.org/workshop/macvi27/challenges/eurs" target="_blank" rel="noreferrer">Challenge ↗</a>
      </nav>
      <div className="live"><i/> SYSTEM READY</div>
    </header>

    <main>
      <section className="hero">
        <div className="eyebrow"><Waves size={15}/> MARITIME COMPUTER VISION · WACV 2027</div>
        <h1>See beneath<br/><em>the blue.</em></h1>
        <p className="lead">An underwater vision workspace for image restoration, marine-class segmentation, and embedded AI research.</p>
        <div className="pills"><span><ShieldCheck/> EURS workflow</span><span><Cpu/> Track 1 + Track 2 ready structure</span></div>
      </section>

      <section className="stats" id="metrics">
        <div><small>PIPELINE</small><strong>02 <span>STAGES</span></strong><p>Restore → Segment</p></div>
        <div><small>SEMANTIC LABELS</small><strong>05 <span>CLASSES</span></strong><p>Marine-scene categories</p></div>
        <div><small>INFERENCE</small><strong>{result ? `${result.processing_ms} ms` : "—"}</strong><p>Local processing time</p></div>
        <div><small>MODEL STATUS</small><strong className="baseline">BASELINE</strong><p>Training required for AI scoring</p></div>
      </section>

      <section className="workspace" id="workspace">
        <div className="section-head"><div><div className="eyebrow">01 / IMAGE LAB</div><h2>Restoration & segmentation</h2></div>
          <span className="tag">CLASSICAL CV BASELINE</span></div>

        <div className="workgrid">
          <div className="panel upload-panel">
            <div className="panel-title"><Upload size={17}/> INPUT IMAGE <span>01</span></div>
            <label className="drop" onDragOver={e => e.preventDefault()}
              onDrop={e => { e.preventDefault(); chooseFile(e.dataTransfer.files?.[0]); }}>
              {preview ? <img src={preview} alt="Selected underwater input"/> :
                <div className="empty"><ImageIcon size={36}/><strong>Drop an underwater image</strong>
                  <span>or click to browse · PNG, JPG, WEBP</span></div>}
              <input type="file" accept="image/*" onChange={e => chooseFile(e.target.files?.[0])}/>
            </label>
            <div className="fileline"><span>{file ? file.name : "No image selected"}</span>
              <span>{file ? `${(file.size / 1024).toFixed(0)} KB` : "MAX 12 MB"}</span></div>
            <button className="primary" disabled={!file || busy} onClick={process}>
              {busy ? <LoaderCircle className="spin"/> : <ScanSearch size={17}/>}
              {busy ? "Processing..." : "Restore & analyze"} <span>→</span>
            </button>
            {error && <p className="error">{error}</p>}
          </div>

          <div className="panel output-panel">
            <div className="panel-title"><Activity size={17}/> PIPELINE OUTPUT <span>02</span></div>
            {!result ? <div className="output-empty"><div className="pulse"><Waves size={34}/></div>
              <strong>Awaiting input</strong><p>Upload an image to run the processing pipeline.</p>
              <div className="steps"><span>01 RESTORE</span><i>→</i><span>02 SEGMENT</span></div>
            </div> : <>
              <div className="outputs">
                <div className="output-card"><img src={imageUrl(result.restored_png)} alt="Restored baseline"/>
                  <div><b>Restored image</b><button onClick={() => downloadBase64(result.restored_png, "restored.png")}><Download size={14}/></button></div></div>
                <div className="output-card"><img src={imageUrl(result.overlay_png)} alt="Segmentation heuristic overlay"/>
                  <div><b>Segmentation overlay</b><button onClick={() => downloadBase64(result.mask_png, "mask-visualization.png")}><Download size={14}/></button></div></div>
              </div>
              <div className="warning">{result.warning}</div>
              <div className="counts">{Object.entries(result.class_pixel_counts).map(([name, count]) =>
                <div key={name}><span>{name}</span><b>{count.toLocaleString()} px</b></div>)}</div>
            </>}
          </div>
        </div>
      </section>

      <section className="classes">
        <div className="section-head"><div><div className="eyebrow">02 / SEMANTIC MAP</div><h2>Five-class taxonomy</h2></div></div>
        <div className="classgrid">
          {[["00","Background","#202020"],["01","Coral","#ff9114"],["02","Fish","#00dcff"],["03","Elasmobranch","#dc28b4"],["04","Other","#50c864"]].map(c =>
            <div className="classitem" key={c[0]}><i style={{background:c[2]}}/><div><small>CLASS {c[0]}</small><strong>{c[1]}</strong></div></div>)}
        </div>
      </section>

      <section className="next">
        <div><div className="eyebrow">03 / RESEARCH ROADMAP</div><h2>From baseline to embedded AI.</h2>
          <p>Next: inspect the official dataset, train with ground-truth masks, evaluate per-class IoU and restoration quality, then export and validate the embedded ONNX model.</p></div>
        <a className="secondary" href="https://drive.google.com/drive/folders/10nWoxDEdWjKoIzYQp3--KSnWca3Vn4C5" target="_blank" rel="noreferrer">Open EURS dataset ↗</a>
      </section>
    </main>
    <footer><div className="brand"><Waves/><strong>EURS AI STUDIO</strong></div>
      <span>Research prototype · Validate before challenge submission</span>
      <a href="https://www.macvi.org/workshop/macvi27/challenges/eurs" target="_blank" rel="noreferrer"><Github size={15}/> OFFICIAL CHALLENGE</a>
    </footer>
  </div>;
}

createRoot(document.getElementById("root")).render(<App/>);
