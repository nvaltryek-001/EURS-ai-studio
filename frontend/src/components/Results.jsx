import React, { useEffect, useState } from "react";
import { CLASSES } from "../api";

// Categorical slots 1-3 of the validated reference palette (dark-surface steps); colour follows the model.
const MODEL_COLOR = { classical: "#3987e5", eurs_lite: "#d95926", eurs_net: "#199e70" };
const short = (s) => s.split(" (")[0];
const fmt = (v, d = 2, pct = false) => (v == null || Number.isNaN(v) ? "–" : pct ? (100 * v).toFixed(1) : v.toFixed(d));

function Bars({ models }) {
  const [hover, setHover] = useState(null);
  return (
    <figure className="iou-chart">
      <figcaption>
        <strong>Per-class IoU on the test split</strong>
        <span className="legend">{models.map((m) => (
          <span key={m.name}><i style={{ background: MODEL_COLOR[m.name] || "#9085e9" }} />{short(m.display_name)}</span>))}
        </span>
      </figcaption>
      <div className="iou-rows">
        {CLASSES.map((c) => (
          <div className="iou-row" key={c.name}>
            <span className="iou-class">{c.name}</span>
            <div className="iou-bars">
              {models.map((m) => {
                const v = m.per_class_iou?.[c.name];
                const key = `${c.name}-${m.name}`;
                return (
                  <div className="iou-track" key={m.name} onMouseEnter={() => setHover(key)} onMouseLeave={() => setHover(null)}>
                    <span className="iou-bar" style={{ width: `${Math.max(0.5, 100 * (v || 0))}%`, background: MODEL_COLOR[m.name] || "#9085e9" }} />
                    <span className="iou-val">{fmt(v, 1, true)}</span>
                    {hover === key && <span className="tip">{short(m.display_name)}, {c.name}: {fmt(v, 1, true)}% IoU</span>}
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </figure>
  );
}

export default function Results() {
  const [data, setData] = useState(null);
  const [tab, setTab] = useState("main");

  useEffect(() => {
    fetch("/results.json").then((r) => (r.ok ? r.json() : null)).then((d) => {
      setData(d);
      if (d && !(d.main?.models?.length) && d.synthetic?.models?.length) setTab("synthetic");
    }).catch(() => setData(null));
  }, []);

  const block = data?.[tab];
  const models = block?.models || [];
  const base = models[0];

  return (
    <section className="band band-mid" id="results" data-depth="18" data-zone="Results">
      <div className="wrap">
        <header className="band-head">
          <h2>Results</h2>
          <p>Every number below is read from the evaluation files the training pipeline writes. Models are selected on the validation split and scored once on the held-out test split.</p>
        </header>

        <div className="seg" role="tablist" aria-label="Benchmark">
          <button role="tab" aria-selected={tab === "main"} className={tab === "main" ? "on" : ""} onClick={() => setTab("main")}>Official EURS data</button>
          <button role="tab" aria-selected={tab === "synthetic"} className={tab === "synthetic" ? "on" : ""} onClick={() => setTab("synthetic")}>Synthetic smoke test</button>
        </div>

        {tab === "synthetic" && models.length > 0 && (
          <p className="banner">These scores come from procedurally generated scenes. They show that training and evaluation run end to end; they are not EURS benchmark results.</p>
        )}

        {models.length === 0 ? (
          <div className="pending">
            <h3>No EURS results yet</h3>
            <p>Run the pipeline on the official training archive and this section fills in automatically:</p>
            <pre><code>bash ml/run_pipeline.sh data/eurs_train eurs</code></pre>
            <p>Or train on a free GPU with the Colab notebook in <code>ml/notebooks/</code>.</p>
          </div>
        ) : (
          <>
            {block.split && (
              <dl className="split">
                <div><dt>Train</dt><dd>{block.split.counts.train}</dd></div>
                <div><dt>Validation</dt><dd>{block.split.counts.val}</dd></div>
                <div><dt>Test</dt><dd>{block.split.counts.test}</dd></div>
                <div><dt>Split</dt><dd>{block.split.group_aware ? "grouped by dive" : "per image"}, seed {block.split.seed}</dd></div>
              </dl>
            )}
            <div className="table-scroll">
              <table className="bench">
                <thead><tr><th>Method</th><th>PSNR (dB)</th><th>SSIM</th><th>mIoU (%)</th><th>Mean F1 (%)</th><th>Time per image</th></tr></thead>
                <tbody>
                  {base && <tr className="muted"><td>Degraded input, unchanged</td><td>{fmt(base.input_psnr)}</td><td>{fmt(base.input_ssim, 3)}</td><td>–</td><td>–</td><td>–</td></tr>}
                  {models.map((m) => (
                    <tr key={m.name}>
                      <td><i className="swatch" style={{ background: MODEL_COLOR[m.name] || "#9085e9" }} />{m.display_name}</td>
                      <td>{fmt(m.psnr)}{m.ci95?.PSNR && <small> [{m.ci95.PSNR[0].toFixed(1)}, {m.ci95.PSNR[1].toFixed(1)}]</small>}</td>
                      <td>{fmt(m.ssim, 3)}</td>
                      <td>{fmt(m.miou, 1, true)}{m.ci95?.mIoU && <small> [{(100 * m.ci95.mIoU[0]).toFixed(1)}, {(100 * m.ci95.mIoU[1]).toFixed(1)}]</small>}</td>
                      <td>{fmt(m.mf1, 1, true)}</td>
                      <td>{m.latency_ms ? `${m.latency_ms.toFixed(0)} ms` : "–"}</td>
                    </tr>))}
                </tbody>
              </table>
            </div>
            <p className="fine">Brackets give 95% bootstrap intervals over test images. Times are median CPU latency for the full pipeline.</p>
            <Bars models={models.filter((m) => m.per_class_iou)} />
            {tab === "synthetic" && <img className="qual" src="/synthetic_qualitative.png" alt="Test-split examples: degraded input, restored output, clean reference, predicted labels and ground truth" />}
          </>
        )}
        {data?.generated_utc && <p className="fine">Generated {data.generated_utc}.</p>}
      </div>
    </section>
  );
}
