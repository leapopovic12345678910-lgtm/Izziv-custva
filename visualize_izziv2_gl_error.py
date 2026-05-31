from pathlib import Path
import csv
import html
import json
from statistics import mean


BASE_DIR = Path(__file__).resolve().parent
INPUT = BASE_DIR / "data/processed/izziv2_GL_euclidean_error.csv"
OUTPUT = BASE_DIR / "data/processed/izziv2_GL_euclidean_error_visualization.html"

CLIP_ORDER = ["ples", "hodnik", "restavracija"]
METHOD_ORDER = ["yellow", "blue", "red", "green"]
METHOD_LABELS = {
    "yellow": "Yellow",
    "blue": "Blue",
    "red": "Red",
    "green": "Green",
}
GT_LABELS = {
    "ples": "happiness",
    "hodnik": "fear",
    "restavracija": "amusement",
}


def clean_row(row):
    return {key.strip(): (value or "").strip() for key, value in row.items()}


def load_rows():
    with INPUT.open(newline="", encoding="utf-8") as file:
        return [clean_row(row) for row in csv.DictReader(file)]


def summarize(rows):
    summary = []
    for method in METHOD_ORDER:
        for clip in CLIP_ORDER:
            values = [
                float(row["emotion_error"])
                for row in rows
                if row["original method"] == method and row["klip"] == clip
            ]
            if not values:
                continue
            summary.append(
                {
                    "method": method,
                    "clip": clip,
                    "mean": mean(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }
            )
    return summary


def build_html(rows, summary):
    max_error = max(float(row["emotion_error"]) for row in rows)
    min_error = min(float(row["emotion_error"]) for row in rows)

    payload = {
        "rows": rows,
        "summary": summary,
        "clipOrder": CLIP_ORDER,
        "methodOrder": METHOD_ORDER,
        "methodLabels": METHOD_LABELS,
        "gtLabels": GT_LABELS,
        "minError": min_error,
        "maxError": max_error,
    }

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Izziv 2 GL Error Visualization</title>
  <style>
    :root {{
      --ink: #1f2933;
      --muted: #697586;
      --line: #d9dee8;
      --paper: #ffffff;
      --bg: #f5f7fb;
      --yellow: #e5b931;
      --blue: #4c7fd9;
      --red: #d65f5f;
      --green: #61a856;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Arial, Helvetica, sans-serif;
      line-height: 1.35;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 28px;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 0 0 14px;
      font-size: 18px;
      letter-spacing: 0;
    }}
    p {{
      margin: 0;
      color: var(--muted);
      max-width: 860px;
    }}
    .panel {{
      margin-top: 22px;
      padding: 18px;
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-top: 18px;
    }}
    .stat {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }}
    .stat b {{
      display: block;
      font-size: 22px;
      margin-bottom: 2px;
    }}
    .stat span {{
      color: var(--muted);
      font-size: 13px;
    }}
    .legend {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
    }}
    .ramp {{
      width: 180px;
      height: 12px;
      border-radius: 999px;
      background: linear-gradient(90deg, #edf7ee, #f4d06f, #c83e4d);
      border: 1px solid var(--line);
    }}
    svg {{
      display: block;
      max-width: 100%;
      overflow: visible;
    }}
    .axis, .small {{
      fill: var(--muted);
      font-size: 12px;
    }}
    .label {{
      fill: var(--ink);
      font-size: 13px;
      font-weight: 700;
    }}
    .method-label {{
      fill: var(--ink);
      font-size: 13px;
      font-weight: 700;
      text-transform: capitalize;
    }}
    .cell-label {{
      fill: #111827;
      font-size: 11px;
      font-weight: 700;
      text-anchor: middle;
      dominant-baseline: central;
    }}
    .bar-label {{
      fill: var(--ink);
      font-size: 11px;
      text-anchor: middle;
    }}
    .note {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
    }}
    @media (max-width: 760px) {{
      main {{ padding: 18px; }}
      .stats {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>Izziv 2 GL Distance From Ground Truth</h1>
    <p>Lower error means the subject's normalized GL emotion profile is closer to the expected target emotion for that clip. Ground truth: ples = happiness, hodnik = fear, restavracija = amusement.</p>

    <section class="stats" id="stats"></section>

    <section class="panel">
      <h2>Individual Subject Error</h2>
      <div id="heatmap"></div>
      <div class="legend"><span>Closer</span><div class="ramp"></div><span>Further from GT</span></div>
      <div class="note">Each square is one participant in one clip. Numbers inside cells are weighted squared errors.</div>
    </section>

    <section class="panel">
      <h2>Average Error By Clip And Method</h2>
      <div id="bars"></div>
      <div class="note">Bars show average error; whiskers show min-max range inside each method and clip.</div>
    </section>
  </main>

  <script>
    const data = {json.dumps(payload, ensure_ascii=False)};

    const methodColors = {{
      yellow: getCss("--yellow"),
      blue: getCss("--blue"),
      red: getCss("--red"),
      green: getCss("--green"),
    }};

    function getCss(name) {{
      return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    }}

    function fmt(value) {{
      return Number(value).toFixed(3).replace(/0+$/, "").replace(/\\.$/, "");
    }}

    function colorFor(value) {{
      const t = data.maxError === data.minError ? 0 : (Number(value) - data.minError) / (data.maxError - data.minError);
      const stops = [
        [237, 247, 238],
        [244, 208, 111],
        [200, 62, 77],
      ];
      const scaled = t * 2;
      const i = Math.min(1, Math.floor(scaled));
      const local = scaled - i;
      const a = stops[i];
      const b = stops[i + 1];
      const rgb = a.map((start, idx) => Math.round(start + (b[idx] - start) * local));
      return `rgb(${{rgb[0]}}, ${{rgb[1]}}, ${{rgb[2]}})`;
    }}

    function renderStats() {{
      const values = data.rows.map(row => Number(row.emotion_error));
      const best = data.rows.reduce((a, b) => Number(a.emotion_error) <= Number(b.emotion_error) ? a : b);
      const worst = data.rows.reduce((a, b) => Number(a.emotion_error) >= Number(b.emotion_error) ? a : b);
      const avg = values.reduce((sum, value) => sum + value, 0) / values.length;
      const cards = [
        [`${{data.rows.length}}`, "subject-clip observations"],
        [fmt(avg), "overall average error"],
        [fmt(best.emotion_error), `closest: ${{best.original_method || best["original method"]}} / ${{best.klip}} / ${{best["šifra"]}}`],
        [fmt(worst.emotion_error), `furthest: ${{worst.original_method || worst["original method"]}} / ${{worst.klip}} / ${{worst["šifra"]}}`],
      ];
      document.getElementById("stats").innerHTML = cards
        .map(([big, label]) => `<div class="stat"><b>${{big}}</b><span>${{label}}</span></div>`)
        .join("");
    }}

    function renderHeatmap() {{
      const margin = {{ top: 70, right: 20, bottom: 34, left: 90 }};
      const cell = 34;
      const gap = 4;
      const clipGap = 20;
      const methodGap = 22;
      const maxRows = Math.max(...data.methodOrder.map(method => {{
        return Math.max(...data.clipOrder.map(clip => data.rows.filter(row => row["original method"] === method && row.klip === clip).length));
      }}));
      const width = margin.left + data.clipOrder.length * (maxRows * (cell + gap) + clipGap) - clipGap + margin.right;
      const methodHeight = cell + 10;
      const height = margin.top + data.methodOrder.length * methodHeight + (data.methodOrder.length - 1) * methodGap + margin.bottom;
      const svg = el("svg", {{ viewBox: `0 0 ${{width}} ${{height}}`, role: "img" }});

      data.clipOrder.forEach((clip, clipIdx) => {{
        const x = margin.left + clipIdx * (maxRows * (cell + gap) + clipGap);
        svg.appendChild(text(x + (maxRows * (cell + gap) - gap) / 2, 24, `${{clip}}`, "label", "middle"));
        svg.appendChild(text(x + (maxRows * (cell + gap) - gap) / 2, 42, `GT: ${{data.gtLabels[clip]}}`, "small", "middle"));
      }});

      data.methodOrder.forEach((method, methodIdx) => {{
        const y = margin.top + methodIdx * (methodHeight + methodGap);
        svg.appendChild(text(8, y + cell / 2, data.methodLabels[method], "method-label", "start"));
        data.clipOrder.forEach((clip, clipIdx) => {{
          const items = data.rows
            .filter(row => row["original method"] === method && row.klip === clip)
            .sort((a, b) => Number(a["šifra"]) - Number(b["šifra"]));
          const x0 = margin.left + clipIdx * (maxRows * (cell + gap) + clipGap);
          items.forEach((row, itemIdx) => {{
            const x = x0 + itemIdx * (cell + gap);
            const value = Number(row.emotion_error);
            const rect = el("rect", {{
              x, y, width: cell, height: cell, rx: 4,
              fill: colorFor(value),
              stroke: "#9aa4b2",
              "stroke-width": 1,
            }});
            rect.appendChild(el("title", {{}}, `${{data.methodLabels[method]}} / ${{clip}} / subject ${{row["šifra"]}}: error ${{fmt(value)}}`));
            svg.appendChild(rect);
            svg.appendChild(text(x + cell / 2, y + cell / 2, fmt(value), "cell-label"));
            svg.appendChild(text(x + cell / 2, y + cell + 14, row["šifra"], "small", "middle"));
          }});
        }});
      }});

      document.getElementById("heatmap").appendChild(svg);
    }}

    function renderBars() {{
      const margin = {{ top: 26, right: 20, bottom: 54, left: 54 }};
      const width = 1060;
      const height = 340;
      const plotW = width - margin.left - margin.right;
      const plotH = height - margin.top - margin.bottom;
      const max = Math.max(...data.summary.map(row => row.max));
      const svg = el("svg", {{ viewBox: `0 0 ${{width}} ${{height}}`, role: "img" }});
      const groupW = plotW / data.clipOrder.length;
      const barW = 34;
      const barGap = 12;

      [0, 0.1, 0.2, 0.3].forEach(tick => {{
        const y = margin.top + plotH - (tick / max) * plotH;
        svg.appendChild(el("line", {{ x1: margin.left, y1: y, x2: width - margin.right, y2: y, stroke: "#e4e8f0" }}));
        svg.appendChild(text(margin.left - 10, y + 4, tick.toFixed(1), "axis", "end"));
      }});

      data.clipOrder.forEach((clip, clipIdx) => {{
        const groupX = margin.left + clipIdx * groupW;
        svg.appendChild(text(groupX + groupW / 2, height - 18, clip, "label", "middle"));
        data.methodOrder.forEach((method, methodIdx) => {{
          const item = data.summary.find(row => row.clip === clip && row.method === method);
          if (!item) return;
          const x = groupX + groupW / 2 - (data.methodOrder.length * barW + (data.methodOrder.length - 1) * barGap) / 2 + methodIdx * (barW + barGap);
          const barH = (item.mean / max) * plotH;
          const y = margin.top + plotH - barH;
          const minY = margin.top + plotH - (item.min / max) * plotH;
          const maxY = margin.top + plotH - (item.max / max) * plotH;
          const visibleMinY = Math.min(minY, y - 8);
          const rect = el("rect", {{ x, y, width: barW, height: barH, rx: 4, fill: methodColors[method] }});
          rect.appendChild(el("title", {{}}, `${{data.methodLabels[method]}} / ${{clip}}: mean ${{fmt(item.mean)}}, range ${{fmt(item.min)}}-${{fmt(item.max)}}`));
          svg.appendChild(rect);
          svg.appendChild(el("line", {{ x1: x + barW / 2, y1: maxY, x2: x + barW / 2, y2: visibleMinY, stroke: "#2f3a4a", "stroke-width": 1.5 }}));
          svg.appendChild(el("line", {{ x1: x + 8, y1: maxY, x2: x + barW - 8, y2: maxY, stroke: "#2f3a4a", "stroke-width": 1.5 }}));
          svg.appendChild(el("line", {{ x1: x + 8, y1: visibleMinY, x2: x + barW - 8, y2: visibleMinY, stroke: "#2f3a4a", "stroke-width": 1.5 }}));
          svg.appendChild(text(x + barW / 2, y - 5, fmt(item.mean), "bar-label"));
          svg.appendChild(text(x + barW / 2, margin.top + plotH + 15, data.methodLabels[method][0], "axis", "middle"));
        }});
      }});

      svg.appendChild(text(14, margin.top + plotH / 2, "mean error", "axis", "middle", -90));
      document.getElementById("bars").appendChild(svg);
    }}

    function el(name, attrs = {{}}, textContent = null) {{
      const node = document.createElementNS("http://www.w3.org/2000/svg", name);
      for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
      if (textContent !== null) node.textContent = textContent;
      return node;
    }}

    function text(x, y, content, className, anchor = null, rotate = null) {{
      const attrs = {{ x, y, class: className }};
      if (anchor) attrs["text-anchor"] = anchor;
      if (rotate !== null) attrs.transform = `rotate(${{rotate}} ${{x}} ${{y}})`;
      return el("text", attrs, content);
    }}

    renderStats();
    renderHeatmap();
    renderBars();
  </script>
</body>
</html>
"""


def main():
    rows = load_rows()
    summary = summarize(rows)
    OUTPUT.write_text(build_html(rows, summary), encoding="utf-8")
    print(f"Wrote visualization to {OUTPUT}")


if __name__ == "__main__":
    main()
