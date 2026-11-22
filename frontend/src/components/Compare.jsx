import React, { useState } from "react";

/** Before/after slider. The range input gives keyboard + screen-reader control. */
export default function Compare({ before, after, beforeLabel = "Degraded", afterLabel = "Restored", ratio = "4 / 3" }) {
  const [pos, setPos] = useState(50);
  return (
    <div className="compare" style={{ aspectRatio: ratio }}>
      <img src={before} alt={beforeLabel} draggable="false" />
      <img src={after} alt={afterLabel} draggable="false" style={{ clipPath: `inset(0 0 0 ${pos}%)` }} />
      <div className="compare-handle" style={{ left: `${pos}%` }} aria-hidden="true"><span /></div>
      <span className="compare-tag left">{beforeLabel}</span>
      <span className="compare-tag right">{afterLabel}</span>
      <input type="range" min="0" max="100" value={pos} onChange={(e) => setPos(+e.target.value)}
        aria-label={`Drag to compare ${beforeLabel.toLowerCase()} and ${afterLabel.toLowerCase()}`} />
    </div>
  );
}
