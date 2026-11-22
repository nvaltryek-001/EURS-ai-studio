import React, { useState } from "react";
import { Check, Copy, Doc, Github } from "../icons";

const REPO = "https://github.com/nvaltryek-001/EURS-ai-studio";
const BIBTEX = `@misc{eurs_ai_studio_2026,
  title  = {Restoration-Aware Segmentation of Underwater Imagery
            for the EURS Challenge},
  author = {EURS AI Studio contributors},
  year   = {2026},
  howpublished = {\\url{${REPO}}}
}`;

export default function Paper() {
  const [copied, setCopied] = useState(false);
  async function copy() {
    try {
      await navigator.clipboard.writeText(BIBTEX);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch { /* clipboard blocked: the text stays selectable below */ }
  }
  return (
    <section className="band band-abyss" id="paper" data-depth="38" data-zone="Paper">
      <div className="wrap paper">
        <header className="band-head">
          <h2>Paper</h2>
        </header>
        <article className="abstract">
          <h3>Restoration-Aware Segmentation of Underwater Imagery for the EURS Challenge</h3>
          <p>Light underwater loses red first, and scatter adds a blue-green veil, so both colour restoration and scene parsing degrade together. We study the two tasks jointly on the EURS benchmark. EURS-Net shares one encoder between a restoration decoder and a five-class segmentation decoder, and feeds the restoration features into the segmentation decoder at every scale. EURS-Lite, a CPU-only baseline built from two small per-pixel networks, gives a reproducible lower bound that trains in minutes without a GPU. All models follow one leakage-aware protocol: an 80/10/10 split grouped by dive, validation-only model selection, and a single scored pass on the held-out test split with bootstrap confidence intervals.</p>
          <div className="paper-links">
            <a className="btn btn-sun" href={`${REPO}/blob/main/paper/main.pdf`}><Doc /> Read the PDF</a>
            <a className="btn btn-ghost" href={`${REPO}/tree/main/paper`}><Github /> LaTeX source</a>
          </div>
        </article>
        <div className="bib">
          <div className="bib-head"><span>Cite this work</span>
            <button className="btn btn-ghost small" onClick={copy}>{copied ? <><Check /> Copied</> : <><Copy /> Copy BibTeX</>}</button></div>
          <pre><code>{BIBTEX}</code></pre>
        </div>
      </div>
    </section>
  );
}
