"""
STEP 1 — PREPROCESSOR (Enhanced DOCX Support)
Extract raw text from any uploaded file.
Supports: PDF, DOCX, TXT, JPG/PNG (OCR), CSV, XLSX

DOCX now extracts:
  - Paragraphs (with bold/italic metadata)
  - Tables (as structured rows + readable text)
  - Headers & Footers (all sections)
"""
import os, re

try:    import pdfplumber;         PDF   = True
except: PDF   = False

try:    from docx import Document; DOCX  = True
except: DOCX  = False

try:
    from PIL import Image
    import pytesseract
    OCR = True
except: OCR = False

try:    import pandas as pd;       PANDAS = True
except: PANDAS = False

try:    import openpyxl;           XLSX   = True
except: XLSX   = False


class Preprocessor:

    def extract(self, file_path: str) -> dict:
        ext = os.path.splitext(file_path)[1].lower()
        out = {
            "file_path":  file_path,
            "file_name":  os.path.basename(file_path),
            "file_type":  ext,
            "text":       "",
            "page_count": 1,
            "raw_data":   None,
            "error":      None
        }
        try:
            if   ext == ".pdf":                            out.update(self._pdf(file_path))
            elif ext == ".docx":                           out.update(self._docx(file_path))
            elif ext in [".jpg",".jpeg",".png",".tiff"]:  out.update(self._image(file_path))
            elif ext == ".txt":                            out.update(self._txt(file_path))
            elif ext == ".csv":                            out.update(self._csv(file_path))
            elif ext in [".xlsx", ".xls"]:                out.update(self._xlsx(file_path))
            else: out["error"] = f"Unsupported: {ext}"
        except Exception as e:
            out["error"] = str(e)

        out["text"] = self._clean(out["text"])
        return out

    # ── ENHANCED DOCX ─────────────────────────────────
    def _docx(self, p):
        """
        Extracts from a .docx file:
          1. Paragraphs → plain text (joined with newlines)
          2. Tables     → each row as pipe-separated text + structured raw_data
          3. Headers & Footers → prepended to text block

        raw_data structure:
          {
            "format": "docx",
            "paragraphs": [
                {"text": "...", "bold": False, "italic": False}
            ],
            "tables": [
                {
                    "index": 0,
                    "headers": ["#", "Date", "Subject", ...],
                    "rows": [
                        {"#": "1", "Date": "2026-03-10", "Subject": "English", ...}
                    ]
                }
            ],
            "headers": ["SUNRISE PUBLIC SCHOOL — Annual Exam Timetable"],
            "footers": ["Confidential — For Student Use Only   |   Page"]
          }
        """
        if not DOCX:
            raise ImportError("pip install python-docx")

        doc = Document(p)
        text_parts = []
        para_meta  = []
        table_data = []

        # ── 1. Headers & Footers ──────────────────────
        header_texts = []
        footer_texts = []

        for section in doc.sections:
            for para in section.header.paragraphs:
                t = para.text.strip()
                if t:
                    header_texts.append(t)
            for para in section.footer.paragraphs:
                t = para.text.strip()
                if t:
                    footer_texts.append(t)

        # Deduplicate while preserving order
        seen = set()
        unique_headers = []
        for h in header_texts:
            if h not in seen:
                seen.add(h)
                unique_headers.append(h)

        seen = set()
        unique_footers = []
        for f in footer_texts:
            if f not in seen:
                seen.add(f)
                unique_footers.append(f)

        if unique_headers:
            text_parts.append("[HEADER]\n" + "\n".join(unique_headers))
        if unique_footers:
            text_parts.append("[FOOTER]\n" + "\n".join(unique_footers))

        # ── 2. Paragraphs ─────────────────────────────
        para_lines = []
        for para in doc.paragraphs:
            t = para.text.strip()

            # Collect metadata for raw_data
            is_bold   = any(run.bold   for run in para.runs if run.text.strip())
            is_italic = any(run.italic for run in para.runs if run.text.strip())

            para_meta.append({
                "text":   t,
                "bold":   is_bold,
                "italic": is_italic
            })

            if t:
                para_lines.append(t)

        if para_lines:
            text_parts.append("\n".join(para_lines))

        # ── 3. Tables ─────────────────────────────────
        for t_idx, table in enumerate(doc.tables):
            rows = []
            for row in table.rows:
                rows.append([cell.text.strip() for cell in row.cells])

            if not rows:
                continue

            # First row = header
            headers = rows[0]
            data_rows = []
            table_text_lines = [" | ".join(headers)]  # readable header line

            for row in rows[1:]:
                # Build dict row (zip handles unequal lengths safely)
                row_dict = dict(zip(headers, row))
                data_rows.append(row_dict)
                table_text_lines.append(" | ".join(row))

            table_data.append({
                "index":   t_idx,
                "headers": headers,
                "rows":    data_rows
            })

            text_parts.append(f"[TABLE {t_idx + 1}]\n" + "\n".join(table_text_lines))

        return {
            "text": "\n\n".join(text_parts),
            "raw_data": {
                "format":     "docx",
                "paragraphs": para_meta,
                "tables":     table_data,
                "headers":    unique_headers,
                "footers":    unique_footers
            }
        }

    # ── Existing formats (unchanged) ──────────────────
    def _pdf(self, p):
        if not PDF: raise ImportError("pip install pdfplumber")
        text, pages = "", 0
        with pdfplumber.open(p) as pdf:
            pages = len(pdf.pages)
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
        return {"text": text, "page_count": pages}

    def _image(self, p):
        if not OCR: raise ImportError("pip install pytesseract pillow")
        return {"text": pytesseract.image_to_string(Image.open(p))}

    def _txt(self, p):
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            return {"text": f.read()}

    def _csv(self, p):
        if not PANDAS: raise ImportError("pip install pandas")
        try:
            df = pd.read_csv(p, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(p, encoding="latin-1")
        df = df.dropna(how="all")
        df.columns = [str(c).strip() for c in df.columns]
        text = self._df_to_text(df, source=p)
        return {
            "text": text,
            "raw_data": {
                "format":  "csv",
                "columns": list(df.columns),
                "rows":    df.fillna("").to_dict(orient="records"),
                "shape":   list(df.shape)
            }
        }

    def _xlsx(self, p):
        if not PANDAS: raise ImportError("pip install pandas openpyxl")
        sheets   = pd.read_excel(p, sheet_name=None, engine="openpyxl")
        all_text = []
        all_data = {}
        for sheet_name, df in sheets.items():
            df = df.dropna(how="all")
            df.columns = [str(c).strip() for c in df.columns]
            sheet_text = f"\n[Sheet: {sheet_name}]\n"
            sheet_text += self._df_to_text(df, source=sheet_name)
            all_text.append(sheet_text)
            all_data[sheet_name] = {
                "columns": list(df.columns),
                "rows":    df.fillna("").to_dict(orient="records"),
                "shape":   list(df.shape)
            }
        return {
            "text":       "\n".join(all_text),
            "page_count": len(sheets),
            "raw_data":   {"format": "xlsx", "sheets": all_data}
        }

    def _df_to_text(self, df, source=""):
        lines = ["Columns: " + " | ".join(df.columns)]
        for _, row in df.iterrows():
            lines.append(" | ".join(str(v) for v in row.values))
        return "\n".join(lines)

    def _clean(self, text):
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+',  ' ',   text)
        return text.strip()


# ── Quick test ────────────────────────────────────────
if __name__ == "__main__":
    import json, sys

    path = sys.argv[1] if len(sys.argv) > 1 else "exam_timetable.docx"
    result = Preprocessor().extract(path)

    print("=== TEXT EXTRACTED ===")
    print(result["text"][:2000])   # first 2000 chars

    print("\n=== RAW DATA SUMMARY ===")
    rd = result["raw_data"]
    if rd:
        print(f"Format   : {rd['format']}")
        print(f"Paragraphs: {len(rd['paragraphs'])}")
        print(f"Tables   : {len(rd['tables'])}")
        for t in rd["tables"]:
            print(f"  Table {t['index']+1}: {len(t['rows'])} rows | headers: {t['headers']}")
        print(f"Headers  : {rd['headers']}")
        print(f"Footers  : {rd['footers']}")