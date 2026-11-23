import React from "react";

const STEPS = [
  ["Index", "Pair every degraded image with its clean reference and label mask, whatever the archive's folder layout."],
  ["Split", "Divide the training archive 80 / 10 / 10 by dive, so frames from one dive never appear in two splits. Rare classes are stratified."],
  ["Train", "Fit on the training split only. Pick checkpoints and hyper-parameters on the validation split."],
  ["Test", "Score once on the untouched test split: PSNR and SSIM for restoration, IoU, F1, precision and recall per class for segmentation."],
  ["Report", "Write the paper's tables and figures and this page's results from the same evaluation files."],
];

export default function Method() {
  return (
    <section className="band band-deep" id="method" data-depth="28" data-zone="Method">
      <div className="wrap">
        <header className="band-head">
          <h2>Method</h2>
          <p>Two models share one protocol. EURS-Lite trains on a laptop CPU in minutes. EURS-Net is a deep network for GPU training, exported to ONNX for the server and for embedded targets.</p>
        </header>

        <ol className="protocol">
          {STEPS.map(([t, d]) => <li key={t}><strong>{t}</strong><p>{d}</p></li>)}
        </ol>

        <figure className="arch">
          <svg viewBox="0 0 800 250" role="img" aria-labelledby="arch-title">
            <title id="arch-title">EURS-Net: a shared encoder feeds a restoration decoder and a segmentation decoder; restoration features are fused into the segmentation decoder at every scale.</title>
            <defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="currentColor" /></marker></defs>
            <g className="arch-box"><rect x="2" y="100" width="140" height="50" rx="6" /><text x="72" y="130">Degraded image</text></g>
            <g className="arch-box strong"><rect x="180" y="85" width="170" height="80" rx="6" /><text x="265" y="120">Shared encoder</text><text x="265" y="140" className="sub">ResNet or MobileNetV3</text></g>
            <g className="arch-box"><rect x="395" y="25" width="185" height="60" rx="6" /><text x="487" y="52">Restoration decoder</text><text x="487" y="70" className="sub">U-Net, residual output</text></g>
            <g className="arch-box"><rect x="395" y="165" width="185" height="60" rx="6" /><text x="487" y="192">Segmentation decoder</text><text x="487" y="210" className="sub">5 classes</text></g>
            <g className="arch-box out"><rect x="625" y="25" width="170" height="60" rx="6" /><text x="710" y="60">Restored image</text></g>
            <g className="arch-box out"><rect x="625" y="165" width="170" height="60" rx="6" /><text x="710" y="200">Label map</text></g>
            <g className="arch-line">
              <path d="M142 125H178" markerEnd="url(#ah)" />
              <path d="M350 110C372 110 372 55 393 55" markerEnd="url(#ah)" />
              <path d="M350 140C372 140 372 195 393 195" markerEnd="url(#ah)" />
              <path d="M580 55H623" markerEnd="url(#ah)" />
              <path d="M580 195H623" markerEnd="url(#ah)" />
              <path className="fuse" d="M487 85V163" markerEnd="url(#ah)" />
            </g>
            <text x="497" y="128" className="fuse-label">feature fusion at every scale</text>
          </svg>
        </figure>
      </div>
    </section>
  );
}
