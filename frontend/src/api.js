const env = import.meta.env || {};
export const API = (env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");

export async function getJSON(path, opts = {}) {
  const res = await fetch(path.startsWith("http") ? path : `${API}${path}`, opts);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed with status ${res.status}`);
  }
  return res.json();
}

export async function processImage(file, model) {
  const body = new FormData();
  body.append("file", file);
  return getJSON(`/api/process?model=${encodeURIComponent(model || "auto")}`, { method: "POST", body });
}

export const CLASSES = [
  { name: "Background", color: "#18222f" },
  { name: "Coral", color: "#ff8a4c" },
  { name: "Fish", color: "#38d6ff" },
  { name: "Elasmobranch", color: "#e848c4" },
  { name: "Other", color: "#78d678" },
];

export const png = (b64) => `data:image/png;base64,${b64}`;
