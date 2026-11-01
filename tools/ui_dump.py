import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "ui.xml"
s = open(path, encoding="utf-8", errors="replace").read()
for node in re.findall(r"<node\b[^>]*>", s):
    text = re.search(r'text="([^"]*)"', node)
    bounds = re.search(r'bounds="([^"]*)"', node)
    rid = re.search(r'resource-id="([^"]*)"', node)
    cls = re.search(r'class="([^"]*)"', node)
    clickable = re.search(r'clickable="([^"]*)"', node)
    label = text.group(1) if text else ""
    ident = rid.group(1) if rid else ""
    klass = cls.group(1) if cls else ""
    if not label.strip() and not ident.strip():
        continue
    print(f"{bounds.group(1) if bounds else '':24} click={clickable.group(1) if clickable else '?':5} {klass.split('.')[-1]:18} {ident:28} {label}")
