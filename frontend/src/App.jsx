import React, { useEffect, useState } from "react";
import { getJSON } from "./api";
import Compare from "./components/Compare";
import Method from "./components/Method";
import Paper from "./components/Paper";
import Results from "./components/Results";
import Studio from "./components/Studio";
import { Github, Wave } from "./icons";

const REPO = "https://github.com/nvaltryek-001/EURS-ai-studio";
const CHALLENGE = "https://www.macvi.org/workshop/macvi27/challenges/eurs";

/** Fixed ruler on the left edge: scroll position read as depth, as light fades with it. */
function DepthGauge() {
  const [state, setState] = useState({ depth: 0, zone: "Surface", frac: 0 });
  useEffect(() => {
    const onScroll = () => {
      const max = document.documentElement.scrollHeight - window.innerHeight;
      const frac = max > 0 ? Math.min(1, window.scrollY / max) : 0;
      let zone = "Surface";
      document.querySelectorAll("[data-zone]").forEach((el) => {
        if (el.getBoundingClientRect().top < window.innerHeight * 0.45) zone = el.dataset.zone;
      });
      setState({ depth: Math.round(frac * 40), zone, frac });
    };
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => { window.removeEventListener("scroll", onScroll); window.removeEventListener("resize", onScroll); };
  }, []);
  return (
    <div className="gauge" aria-hidden="true">
      <div className="gauge-scale">
        {[0, 10, 20, 30, 40].map((m) => <span key={m} style={{ top: `${m * 2.5}%` }}>{m} m</span>)}
        <div className="gauge-mark" style={{ top: `${state.frac * 100}%` }}>
          <span><b>{state.depth} m</b><em>{state.zone}</em></span>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [api, setApi] = useState({ online: null, models: [], default: null });

  useEffect(() => {
    let alive = true;
    getJSON("/api/models")
      .then((d) => alive && setApi({ online: true, models: d.models, default: d.default }))
      .catch(() => alive && setApi({ online: false, models: [], default: null }));
    return () => { alive = false; };
  }, []);

  const label = (name) => api.models.find((m) => m.name === name)?.display_name || name;
  const status = api.online === null ? "Connecting to server" : api.online ? `Server online, ${label(api.default)}` : "Server offline";

  return (
    <>
      <a className="skip" href="#studio">Skip to the studio</a>
      <DepthGauge />
      <header className="nav">
        <a className="brand" href="#top"><Wave width={22} height={22} /> EURS AI Studio</a>
        <nav aria-label="Sections">
          <a href="#studio">Studio</a><a href="#results">Results</a><a href="#method">Method</a><a href="#paper">Paper</a>
        </nav>
        <span className={`status ${api.online ? "ok" : api.online === false ? "down" : ""}`}><i />{status}</span>
      </header>

      <main id="top">
        <section className="band band-surface hero" data-depth="0" data-zone="Surface">
          <div className="wrap hero-grid">
            <div className="hero-copy">
              <h1>Bring back the colour the water took.</h1>
              <p className="lead">EURS AI Studio restores underwater photographs and maps every pixel to coral, fish, sharks and rays, other objects, or open water. Built for the EURS challenge at MaCVi, WACV 2027.</p>
              <div className="cta">
                <a className="btn btn-primary" href="#studio">Analyse an image</a>
                <a className="btn btn-quiet" href="#results">See the results</a>
              </div>
            </div>
            <figure className="hero-figure">
              <Compare before="/samples/hero_input.jpg" after="/samples/hero_restored.jpg" />
              <figcaption>Drag across a synthetic test scene to compare the degraded input with EURS-Lite's restoration.</figcaption>
            </figure>
          </div>
        </section>

        <Studio api={{ ...api, label }} />
        <Results />
        <Method />
        <Paper />
      </main>

      <footer className="foot">
        <div className="wrap foot-grid">
          <span className="brand"><Wave width={20} height={20} /> EURS AI Studio</span>
          <p>Research software for underwater image restoration and segmentation. Outputs are model predictions; check them before using them in a study.</p>
          <nav aria-label="External">
            <a href={REPO}><Github /> Source code</a>
            <a href={CHALLENGE}>EURS challenge</a>
          </nav>
        </div>
      </footer>
    </>
  );
}
