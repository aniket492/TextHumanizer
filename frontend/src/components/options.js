export const TONES = [
  { value: "natural", label: "Natural" },
  { value: "professional", label: "Professional" },
  { value: "academic", label: "Academic" },
  { value: "casual", label: "Casual" },
  { value: "simple", label: "Simple" },
  { value: "concise", label: "Concise" },
  { value: "detailed", label: "Detailed" },
];

export const INTENSITIES = [
  { value: "light", label: "Light", hint: "Smooth awkward phrasing only" },
  { value: "medium", label: "Medium", hint: "Rephrase freely, keep structure" },
  { value: "strong", label: "Strong", hint: "Restructure sentences where it helps" },
];

export const MAX_WORDS = 1500;

export function countWords(text) {
  const m = text.match(/[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*/g);
  return m ? m.length : 0;
}

export function formatDate(iso) {
  try {
    return new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(new Date(iso));
  } catch {
    return iso;
  }
}

export const label = (list, value) => list.find((o) => o.value === value)?.label ?? value;
