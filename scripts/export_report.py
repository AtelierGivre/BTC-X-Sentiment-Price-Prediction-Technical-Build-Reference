"""Export the report to PDF and the charts to PNG (dev-only; not part of the pipeline).

    python scripts/export_report.py

Produces:
  reports/evaluation_report.pdf      A4, with all 8 charts in an appendix
  reports/figures/png/NN_*.png       2x screenshots of each interactive chart

Needs Playwright + Chromium (`pip install playwright && playwright install chromium`).
Nothing in the model, evaluation or reporting path imports this module.
"""
from __future__ import annotations

import datetime as dt
import functools
import http.server
import socketserver
import threading
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "reports" / "figures"
PNG_DIR = FIGS / "png"
MD = ROOT / "reports" / "evaluation_report.md"
PDF = ROOT / "reports" / "evaluation_report.pdf"

CSS = """
@page { size: A4; margin: 15mm 13mm 15mm 13mm; }
* { box-sizing: border-box; }
body { font-family: "Segoe UI", -apple-system, Arial, sans-serif; font-size: 10.5pt;
       line-height: 1.45; color: #1f2328; }
h1 { font-size: 18pt; margin: 0 0 4pt; }
h2 { font-size: 13pt; margin: 16pt 0 6pt; padding-bottom: 3pt;
     border-bottom: 1px solid #d0d7de; page-break-after: avoid; }
h3 { font-size: 11pt; margin: 12pt 0 4pt; page-break-after: avoid; }
p, li { orphans: 3; widows: 3; }
a { color: #0969da; text-decoration: none; }
code { font-family: Consolas, monospace; font-size: 8.8pt; background: #f6f8fa;
       padding: 1px 3px; border-radius: 3px; }
pre { background: #f6f8fa; padding: 7pt; border-radius: 4px; font-size: 8.5pt;
      page-break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; table-layout: fixed; font-size: 8.2pt;
        margin: 6pt 0 9pt; page-break-inside: avoid; }
th, td { border: 1px solid #d0d7de; padding: 2.5pt 3.5pt; text-align: right;
         overflow-wrap: anywhere; }
th:first-child, td:first-child { width: 25%; }
th:first-child, td:first-child { text-align: left; }
th { background: #f6f8fa; font-weight: 600; }
tr:nth-child(even) td { background: #fafbfc; }
blockquote { margin: 6pt 0; padding: 4pt 8pt; border-left: 3px solid #d0d7de; color: #57606a; }
ul, ol { margin: 4pt 0 8pt; padding-left: 16pt; }
hr { border: none; border-top: 1px solid #d0d7de; margin: 12pt 0; }
.cover { border: 1px solid #d0d7de; border-radius: 5px; padding: 10pt 12pt; margin-bottom: 12pt;
         background: #f6f8fa; }
.cover .meta { font-size: 9pt; color: #57606a; line-height: 1.6; }
figure { margin: 0 0 12pt; page-break-inside: avoid; }
figure img { width: 100%; border: 1px solid #d0d7de; border-radius: 4px; }
figcaption { font-size: 8.5pt; color: #57606a; margin-top: 3pt; }
.appendix { page-break-before: always; }
"""

COVER = """
<div class="cover">
  <h1>Evaluation report — BTCUSDT X-sentiment price model</h1>
  <div class="meta">
    CrowdWisdomTrading &middot; X Sentiment Price Prediction assessment<br>
    Asset BTCUSDT &middot; 1-hour bars &middot; UTC &middot; horizons n+0 … n+5<br>
    Repository: https://github.com/AtelierGivre/BTC-X-Sentiment-Price-Prediction-Technical-Build-Reference<br>
    {generated}
  </div>
</div>
"""


def _generated_line() -> str:
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    return f"Generated {today}"


def serve() -> tuple[socketserver.TCPServer, int]:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


def export_pngs(page, port: int) -> list[Path]:
    PNG_DIR.mkdir(exist_ok=True)
    out = []
    charts = sorted(p for p in FIGS.glob("*.html") if p.name[0].isdigit())
    for html in charts:
        page.goto(f"http://127.0.0.1:{port}/reports/figures/{html.name}", wait_until="load")
        page.wait_for_selector(".plotly-graph-div", state="visible", timeout=30_000)
        page.wait_for_timeout(900)
        target = page.query_selector(".plotly-graph-div")
        png = PNG_DIR / f"{html.stem}.png"
        target.screenshot(path=str(png))
        out.append(png)
        print(f"png  {png.name}  {png.stat().st_size / 1024:.0f} KB")
    return out


def build_html(pngs: list[Path]) -> str:
    source = MD.read_text(encoding="utf-8")
    # the cover already carries the title; drop the markdown duplicate
    lines = source.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    body = markdown.markdown(
        "\n".join(lines),
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
    )
    figures = "\n".join(
        f'<figure><img src="{p.as_uri()}"><figcaption>{p.stem}</figcaption></figure>'
        for p in pngs
    )
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Evaluation report — BTCUSDT X-sentiment</title><style>{CSS}</style></head>
<body>{COVER.format(generated=_generated_line())}{body}
<div class="appendix"><h2>Appendix — charts</h2>
<p>Static exports of the eight interactive charts in <code>reports/figures/</code>.
</p>{figures}</div>
</body></html>"""


def export_pdf(page, pngs: list[Path]) -> None:
    tmp = ROOT / "reports" / "_export.html"
    tmp.write_text(build_html(pngs), encoding="utf-8")
    page.goto(tmp.as_uri(), wait_until="load")
    page.emulate_media(media="print")
    page.pdf(
        path=str(PDF), format="A4", print_background=True,
        margin={"top": "15mm", "bottom": "15mm", "left": "13mm", "right": "13mm"},
        display_header_footer=True,
        header_template="<span></span>",
        footer_template=(
            '<div style="width:100%;font-size:7pt;color:#8b949e;text-align:center;'
            'font-family:Segoe UI,Arial,sans-serif;">'
            'BTCUSDT X-sentiment evaluation &middot; page '
            '<span class="pageNumber"></span> / <span class="totalPages"></span></div>'
        ),
    )
    tmp.unlink()
    pages = len(PdfReader(str(PDF)).pages)
    print(f"pdf  {PDF.name}  {PDF.stat().st_size / 1024:.0f} KB  {pages} pages")


def main() -> None:
    httpd, port = serve()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1400, "height": 900},
                                    device_scale_factor=2)
            pngs = export_pngs(page, port)
            export_pdf(page, pngs)
            browser.close()
    finally:
        httpd.shutdown()

    text = "\n".join((pg.extract_text() or "") for pg in PdfReader(str(PDF)).pages)
    for needle in ("Sharpe", "Sortino", "Max Drawdown", "max_drawdown", "Summary", "n+5"):
        assert needle in text, f"missing from PDF: {needle}"
    for png in pngs:
        assert png.stem in text, f"chart missing from PDF appendix: {png.stem}"
    print(f"pdf  text check: ok ({len(pngs)} charts in appendix)")


if __name__ == "__main__":
    main()
