"""Run pytest and save a self-contained HTML test report."""

import html
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "test_report.html"

result = subprocess.run(
    [sys.executable, "-m", "pytest", "-q"],
    cwd=ROOT,
    text=True,
    capture_output=True,
)
output = (result.stdout + "\n" + result.stderr).strip()
status = "PASS" if result.returncode == 0 else "FAIL"

page = f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<title>Pytest Test Report</title>
<style>
body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; color: #111; }}
h1 {{ border-bottom: 2px solid #111; padding-bottom: 8px; }}
pre {{ border: 1px solid #999; padding: 16px; white-space: pre-wrap; background: #f7f7f7; }}
</style>
</head>
<body>
<h1>Pytest Test Report</h1>
<p><strong>Status:</strong> {status}</p>
<p><strong>Generated:</strong> {datetime.now().isoformat(timespec='seconds')}</p>
<pre>{html.escape(output)}</pre>
</body>
</html>"""
REPORT.write_text(page, encoding="utf-8")
print(output)
print("HTML test report:", REPORT)
raise SystemExit(result.returncode)
