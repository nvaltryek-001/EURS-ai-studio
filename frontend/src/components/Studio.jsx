import React, { useEffect, useRef, useState } from "react";
import { CLASSES, png, processImage } from "../api";
import { Download, Spinner, Upload } from "../icons";
import Compare from "./Compare";

const SAMPLES = ["/samples/sample1.jpg", "/samples/sample2.jpg", "/samples/sample3.jpg"];

function download(b64, name) {
  const a = document.createElement("a");
  a.href = png(b64);
  a.download = name;
  a.click();
}

export default function Studio({ api }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [model, setModel] = useState("auto");
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [view, setView] = useState("compare");
  const [opacity, setOpacity] = useState(0.6);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  useEffect(() => () => preview && URL.revokeObjectURL(preview), [preview]);

  function choose(f) {
    if (!f) return;
    if (!f.type.startsWith("image/")) return setError("That file is not an image. Choose a PNG, JPG or WEBP.");
    if (f.size > 12 * 1024 * 1024) return setError("That image is larger than 12 MB. Choose a smaller file.");
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError("");
  }

  async function useSample(url) {
    const blob = await fetch(url).then((r) => r.blob());
    choose(new File([blob], url.split("/").pop(), { type: blob.type || "image/jpeg" }));
  }

  async function run() {
    if (!file) return;
    setBusy(true);
    setError("");
    try {
      setResult(await processImage(file, model));
      setView("compare");
    } catch (e) {
      setError(api.online === false
        ? "The analysis server is not reachable. Start the backend, or check VITE_API_URL."
        : e.message);
    } finally {
      setBusy(false);
    }
  }

  const pct = result?.class_percentages || {};
  const stem = (file?.name || "image").replace(/\.[^.]+$/, "");

  return (
    <section className="band band-shallow" id="studio" data-depth="8" data-zone="Studio">
      <div className="wrap">
        <header className="band-head">
          <h2>Studio</h2>
          <p>Upload an underwater photo. The server restores its colour and contrast, then labels every pixel as background, coral, fish, elasmobranch or other.</p>
        </header>

        <div className="studio">
          <div className="studio-input">
            <label className={`drop ${dragging ? "is-drag" : ""}`}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => { e.preventDefault(); setDragging(false); choose(e.dataTransfer.files?.[0]); }}>
              {preview
                ? <img src={preview} alt="Selected input" />
                : <span className="drop-empty"><Upload width={28} height={28} />
                    <strong>Drop an image here</strong><span>or click to choose a file (PNG, JPG or WEBP, up to 12 MB)</span></span>}
              <input ref={inputRef} type="file" accept="image/*" onChange={(e) => choose(e.target.files?.[0])} />
            </label>

            <div className="samples">
              <span>No underwater photo to hand? Try a synthetic scene:</span>
              <div>{SAMPLES.map((s, i) => (
                <button key={s} className="sample" onClick={() => useSample(s)} aria-label={`Use synthetic sample ${i + 1}`}>
                  <img src={s} alt="" />
                </button>))}
              </div>
            </div>

            <div className="field">
              <label htmlFor="model">Model</label>
              <select id="model" value={model} onChange={(e) => setModel(e.target.value)}>
                <option value="auto">Best available{api.default ? `: ${api.label(api.default)}` : ""}</option>
                {(api.models || []).map((m) => <option key={m.name} value={m.name}>{m.display_name}</option>)}
              </select>
            </div>

            <button className="btn btn-primary" disabled={!file || busy} onClick={run}>
              {busy ? <><Spinner /> Analysing</> : "Analyse image"}
            </button>
            {error && <p className="error" role="alert">{error}</p>}
          </div>

          <div className="studio-output" aria-live="polite">
            {!result ? (
              <div className="output-empty">
                <p>{busy ? "Restoring and segmenting your image." : "Your restored image and segmentation will appear here."}</p>
              </div>
            ) : (
              <>
                <div className="tabs" role="tablist">
                  {[["compare", "Before and after"], ["overlay", "Segmentation"], ["mask", "Label map"]].map(([k, l]) => (
                    <button key={k} role="tab" aria-selected={view === k} className={view === k ? "on" : ""} onClick={() => setView(k)}>{l}</button>
                  ))}
                </div>

                <div className="stage" style={{ aspectRatio: `${result.width} / ${result.height}` }}>
                  {view === "compare" && <Compare before={png(result.input_png)} after={png(result.restored_png)} ratio={`${result.width} / ${result.height}`} />}
                  {view === "overlay" && <div className="stack">
                    <img src={png(result.restored_png)} alt="Restored image" />
                    <img src={png(result.mask_png)} alt="Predicted classes" style={{ opacity }} />
                  </div>}
                  {view === "mask" && <img className="pixel" src={png(result.mask_png)} alt="Predicted label map" />}
                </div>

                {view === "overlay" && <div className="field inline">
                  <label htmlFor="op">Label opacity</label>
                  <input id="op" type="range" min="0" max="1" step="0.05" value={opacity} onChange={(e) => setOpacity(+e.target.value)} />
                </div>}

                <div className="composition">
                  <div className="comp-bar" role="img" aria-label="Share of pixels per class">
                    {CLASSES.map((c) => pct[c.name] > 0 && (
                      <span key={c.name} style={{ flexGrow: pct[c.name], background: c.color }} title={`${c.name}: ${pct[c.name].toFixed(1)}%`} />
                    ))}
                  </div>
                  <table className="comp-table">
                    <thead><tr><th>Class</th><th>Pixels</th><th>Share</th></tr></thead>
                    <tbody>{CLASSES.map((c) => (
                      <tr key={c.name}><td><i style={{ background: c.color }} />{c.name}</td>
                        <td>{(result.class_pixel_counts[c.name] || 0).toLocaleString()}</td>
                        <td>{(pct[c.name] || 0).toFixed(1)}%</td></tr>))}
                    </tbody>
                  </table>
                </div>

                <dl className="run-meta">
                  <div><dt>Model</dt><dd>{result.model_display_name}</dd></div>
                  <div><dt>Time</dt><dd>{result.processing_ms} ms</dd></div>
                  <div><dt>Size</dt><dd>{result.width} × {result.height}</dd></div>
                </dl>
                {result.warning && <p className="note">{result.warning}</p>}

                <div className="downloads">
                  <button className="btn btn-quiet" onClick={() => download(result.restored_png, `${stem}_restored.png`)}><Download /> Restored image</button>
                  <button className="btn btn-quiet" onClick={() => download(result.mask_png, `${stem}_labels_colour.png`)}><Download /> Colour label map</button>
                  <button className="btn btn-quiet" onClick={() => download(result.mask_index_png, `${stem}_labels.png`)}><Download /> Class-index mask</button>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
