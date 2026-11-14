"""What The Data? - clean a geothermal station export.

Reads test-input.csv (same directory, bare filename) and writes test-output.csv:
    reading_id,station_name,temperature_c,reading_date
Fixes, generically:
  * preamble of '#' comment lines and blank lines before the real header (any length)
  * duplicated readings (satellite retransmissions): one row per reading_id
  * decimal commas (54,6) and stray whitespace / degree signs in temperatures
  * Fahrenheit readings (unit F) converted to Celsius, one decimal place
  * mis-decoded station names (UTF-8 bytes read as Latin-1 / Windows-1252, even twice)
Standard library only.
"""
import csv
import io
import re
import unicodedata
from decimal import Decimal, ROUND_HALF_UP

IN_FILE = "test-input.csv"
OUT_FILE = "test-output.csv"
FIELDS = ["reading_id", "station_name", "temperature_raw", "unit", "reading_date"]


# ---------- reading ----------
def read_text(path):
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def data_lines(text):
    """Drop the preamble: '#' comments and blank lines before (and after) the header."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return [l for l in lines if l.strip() and not l.lstrip().startswith("#")]


# ---------- station names ----------
_SUSPECT = re.compile("[\u0080-ÿŒœŠšŸŽžƒˆ˜–—‘-„†-•…‰‹›€™]")


def _to_bytes(s):
    """Inverse of decoding bytes as Windows-1252, with Latin-1 for the 5 bytes cp1252 leaves undefined."""
    out = bytearray()
    for ch in s:
        try:
            out += ch.encode("cp1252")
        except UnicodeEncodeError:
            if ord(ch) < 256:
                out.append(ord(ch))
            else:
                raise
    return bytes(out)


def fix_name(s):
    s = s.strip().replace("﻿", "")
    for _ in range(3):  # handles text that was mis-decoded more than once
        if not _SUSPECT.search(s):
            break
        try:
            fixed = _to_bytes(s).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            break  # already correct text (e.g. 'Mývatn' is not valid as UTF-8 bytes)
        if fixed == s:
            break
        s = fixed
    return unicodedata.normalize("NFC", s)


# ---------- temperatures ----------
_NUM = re.compile(r"[-+]?\d+(?:[.,]\d+)?")


def to_celsius(raw, unit):
    txt = raw.strip().replace("−", "-").replace(" ", "")
    m = _NUM.search(txt)
    if not m:
        raise ValueError(f"no number in temperature {raw!r}")
    val = Decimal(m.group(0).replace(",", "."))
    u = (unit or "").strip().upper().replace("°", "").replace("º", "")
    if not u:  # unit sometimes glued to the value, e.g. '182.7F'
        u = re.sub(r"[^A-Z]", "", txt.upper().replace("°", ""))[:1]
    if u.startswith("F"):
        val = (val - 32) * 5 / 9
    elif u.startswith("K"):
        val = val - Decimal("273.15")
    return val.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)


# ---------- main ----------
def main():
    lines = data_lines(read_text(IN_FILE))
    header = [h.strip().lower().lstrip("﻿") for h in lines[0].split(";")]
    col = {name: header.index(name) if name in header else i for i, name in enumerate(FIELDS)}

    rows = {}
    for rec in csv.reader(lines[1:], delimiter=";"):
        if not rec or all(not x.strip() for x in rec):
            continue
        get = lambda k: rec[col[k]] if col[k] < len(rec) else ""
        rid_txt = re.sub(r"[^\d-]", "", get("reading_id"))
        if not rid_txt:
            continue
        rid = int(rid_txt)
        if rid in rows:  # retransmission: keep the first copy
            continue
        rows[rid] = (
            fix_name(get("station_name")),
            to_celsius(get("temperature_raw"), get("unit")),
            get("reading_date").strip(),
        )

    out = io.StringIO()
    w = csv.writer(out, lineterminator="\r\n", quoting=csv.QUOTE_NONE, escapechar="\\")
    w.writerow(["reading_id", "station_name", "temperature_c", "reading_date"])
    for rid in sorted(rows):
        name, temp, date = rows[rid]
        w.writerow([rid, name, f"{temp:.1f}", date])
    with open(OUT_FILE, "w", encoding="utf-8", newline="") as f:
        f.write(out.getvalue())


if __name__ == "__main__":
    main()
