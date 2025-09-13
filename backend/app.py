import io
import json
import os
import re
import statistics
from datetime import datetime
import pdfplumber
from flask import Flask, request, jsonify
from flask_cors import CORS
# from openai import OpenAI   # ❌ Commented out
from pprint import pprint
from dotenv import load_dotenv
import fitz
import base64
from openai import OpenAI

load_dotenv()

# --- Flask setup ---
app = Flask(__name__)
CORS(app)

# --- OpenAI setup (commented) ---
load_dotenv(override=True)  # make sure dotenv is loaded
openai_api_key = os.environ.get("OPENAI_API_KEY")

if not openai_api_key:
    print("⚠️ Could not find OPENAI_API_KEY in environment. Falling back to dotenv...")
    from dotenv import dotenv_values
    env_vars = dotenv_values(".env")
    openai_api_key = env_vars.get("OPENAI_API_KEY")

if not openai_api_key:
    raise RuntimeError("❌ OPENAI_API_KEY environment variable is not set!")

client = OpenAI(api_key=openai_api_key)


# --- STEP 1: Extract text, tables, figures/images, fonts ---
def lines_from_chars(page, line_tol=5, word_tol=0.05):
    """
    Group page.chars into lines; return list of line dicts with
    text, bbox, font_size, style, spacing metadata, and per-word font info.
    """
    chars = sorted(
        page.chars,
        key=lambda c: (round(c.get("top", 0), 1), round(c.get("x0", 0), 1))
    )
    if not chars:
        return []

    # --- group chars into lines
    lines = []
    current = [chars[0]]
    for ch in chars[1:]:
        if abs(ch.get("top", 0) - current[0].get("top", 0)) < line_tol:
            current.append(ch)
        else:
            lines.append(current)
            current = [ch]
    if current:
        lines.append(current)

    # --- build line objects
    line_objs = []
    prev_bottom = None
    for idx, ln in enumerate(lines):
        # group chars into words within the line
        words = []
        current_word = [ln[0]]
        for ch in ln[1:]:
            prev = current_word[-1]
            if abs(ch.get("x0", 0) - prev.get("x1", 0)) > word_tol:
                words.append(current_word)
                current_word = [ch]
            else:
                current_word.append(ch)
        if current_word:
            words.append(current_word)

        word_objs = []
        for w in words:
            text = "".join(c.get("text", "") for c in w).strip()
            # print(f"Word: '{text}' | Length: {len(w)} | Chars: {[c.get('text', '') for c in w]}")
            if not text:
                continue
            l = min(c.get("x0", 0) for c in w)
            t = min(c.get("top", 0) for c in w)
            r = max(c.get("x1", 0) for c in w)
            b = max(c.get("bottom", 0) for c in w)

            sizes = [float(c.get("size", 0)) for c in w if c.get("size") is not None]
            font_size = round(statistics.median(sizes), 2) if sizes else None

            fonts = [c.get("fontname", "") for c in w]
            bold = any("Bold" in f for f in fonts)
            italic = any("Italic" in f or "Oblique" in f for f in fonts)

            word_objs.append({
                "text": text,
                "box": {"l": l, "t": t, "r": r, "b": b},
                "font_size": font_size,
                "bold": bold,
                "italic": italic,
            })

        if not word_objs:
            continue

        l = min(w["box"]["l"] for w in word_objs)
        t = min(w["box"]["t"] for w in word_objs)
        r = max(w["box"]["r"] for w in word_objs)
        b = max(w["box"]["b"] for w in word_objs)

        # spacing metadata
        line_breaks_before = 0
        if prev_bottom is not None and (t - prev_bottom) > line_tol:
            line_breaks_before = 1
        prev_bottom = b

        line_objs.append({
            "id": f"ln-{idx}",
            "type": "text",
            "text": " ".join(w["text"] for w in word_objs),
            "box": {"l": l, "t": t, "r": r, "b": b},
            "indent": l,
            "line_breaks_before": line_breaks_before,
            "line_breaks_after": 0,  # to be filled later
            "words": word_objs,
        })
    # --- fill line_breaks_after
    for i in range(len(line_objs) - 1):
        gap = line_objs[i+1]["box"]["t"] - line_objs[i]["box"]["b"]
        if gap > line_tol:
            line_objs[i]["line_breaks_after"] = 1

    return line_objs




def extract_tables_with_bbox(page):
    """
    Use page.find_tables() to get table objects and their bbox.
    Returns list of dicts: { type: 'table', 'table': rows, 'box': {l,t,r,b} }
    """
    tables = []
    found = page.find_tables()
    for t in found:
        bbox = getattr(t, "bbox", None) or getattr(t, "_bbox", None)
        if bbox and len(bbox) == 4:
            l, ttop, r, btm = bbox
        else:
            # fallback: compute bbox from extracted table rows if possible, else skip
            rows = t.extract()
            if rows:
                # try to find words in table rows to estimate bbox (best-effort)
                # fallback to whole page if can't compute
                try:
                    # collect text tokens, find their bounding boxes via page.extract_words
                    words = page.extract_words()
                    # naive fallback -> whole page dims
                    l, ttop, r, btm = 0, 0, page.width, page.height
                except Exception:
                    l, ttop, r, btm = 0, 0, page.width, page.height
            else:
                l, ttop, r, btm = 0, 0, page.width, page.height

        table_rows = t.extract()
        tables.append({
            "type": "table",
            "table": table_rows,
            "box": {"l": l, "t": ttop, "r": r, "b": btm},
        })
    return tables


def line_intersects_bbox(line, bbox, margin=1.0):
    """
    Return True if line's vertical midpoint is inside bbox vertically and horizontally overlaps.
    margin: small tolerance
    """
    line_mid = (line["box"]["t"] + line["box"]["b"]) / 2.0
    tb_top, tb_bottom = bbox["t"] - margin, bbox["b"] + margin
    horiz_overlap = not (line["box"]["r"] < bbox["l"] or line["box"]["l"] > bbox["r"])
    return (tb_top <= line_mid <= tb_bottom) and horiz_overlap

# --- Extract images using PyMuPDF ---
def extract_images_with_bbox_pymupdf(file_bytes, page_number):
    # Uses xref placement rects to get true positions of images on the page


    images = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        page = doc[page_number]

        # Enumerate all image xrefs used on this page
        xref_rows = page.get_images(full=True)  # [(xref, smask, w, h, bpc, cs, name, ...), ...]
        if not xref_rows:
            print(f"[DEBUG] Page {page_number+1}: no image xrefs")
            return images

        for row in xref_rows:
            xref = row[0]
            rects = page.get_image_rects(xref)  # all placements (may be multiple)
            if not rects:
                # No visible placement for this xref on this page
                continue

            # Build pixmap once per xref
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n > 4:  # convert CMYK/others to RGB
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                img_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
            except Exception as e:
                print(f"[WARN] xref={xref} pixmap failed: {e}")
                continue

            for rect in rects:
                l, t, r, b = float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)
                images.append({
                    "type": "image",
                    "subtype": "embedded",
                    "box": {"l": l, "t": t, "r": r, "b": b},
                    "page": page_number + 1,
                    "image_b64": img_b64,
                })

    print(f"[DEBUG] Page {page_number+1}: Found {len(images)} images (from xref rects)")
    return images


    # # --- Method 2: Embedded images (from resources, may lack bbox)
    # for img in page.get_images(full=True):
    #     xref = img[0]
    #     try:
    #         pix = fitz.Pixmap(doc, xref)
    #         if pix.n > 4:
    #             pix = fitz.Pixmap(fitz.csRGB, pix)
    #         img_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
    #         images.append({
    #             "type": "image",
    #             "subtype": "embedded",
    #             # fallback bbox = full page (UI can still box it)
    #             "box": {"l": 0, "t": 0, "r": page.rect.width, "b": page.rect.height},
    #             "page": page_number + 1,
    #             "image_b64": img_b64,
    #         })
    #     except Exception as e:
    #         print(f"[WARN] Embedded image failed: {e}")
    #         continue

    # print(f"[DEBUG] Page {page_number+1}: Found {len(images)} images "
    #       f"(subtypes: {[im.get('subtype', im['type']) for im in images]})")

    # return images

def assemble_elements(file_bytes, page, page_number):
    """
    Build ordered elements for the page:
    - get text lines (with font/style/spacing metadata)
    - get tables (with bbox)
    - get images (with bbox/base64)
    - remove lines that overlap table bboxes
    - combine into one list sorted by vertical position
    """
    text_lines = lines_from_chars(page)  # enriched with style + spacing + word-level info
    pprint(text_lines)
    tables = extract_tables_with_bbox(page)
    images = extract_images_with_bbox_pymupdf(file_bytes, page_number)
    
    # --- Filter out text lines that overlap with any table bbox
    filtered_lines = []
    for ln in text_lines:
        in_any_table = any(line_intersects_bbox(ln, tb["box"]) for tb in tables)
        if not in_any_table:
            filtered_lines.append(ln)

    # --- Merge all elements into a unified list
    elements = []
    for ln in filtered_lines:
        elements.append({
            **ln,  # contains line-level text, box, spacing, and word-level list
            "page": page_number + 1,
            "top": ln["box"]["t"],
        })
    for tb in tables:
        elements.append({
            **tb,
            "page": page_number + 1,
            "top": tb["box"]["t"],
        })
    for im in images:
        elements.append({
            **im,
            "page": page_number + 1,
            "top": im["box"]["t"],
        })

    # --- Sort by top coordinate, fallback to left (chronological order)
    elements.sort(key=lambda e: (e["top"], e["box"].get("l", 0)))

    return elements

# def extract_images_with_bbox(page):
#     """
#     Extract images with their bounding boxes.
#     pdfplumber gives page.images as list of dicts.
#     We'll standardize to {type: 'image', box: {...}, attrs: {...}}
#     """
#     images = []
#     for im in page.images:
#         l = im.get("x0", 0)
#         t = im.get("top", 0)
#         r = im.get("x1", 0)
#         b = im.get("bottom", 0)
#         images.append({
#             "type": "image",
#             "box": {"l": l, "t": t, "r": r, "b": b},
#             "attrs": {
#                 "width": r - l,
#                 "height": b - t,
#                 "name": im.get("name"),
#                 "stream": im.get("stream"),
#                 "page_number": page.page_number
#             }
#         })
    # return images



def build_simplified_view_from_elements(elements, gap_multiplier=1.5):
    """
    Build a simplified string preserving structure:
    - Preserve explicit line breaks (line_breaks_before/after) from extraction
    - Fallback to gap-based blank lines when explicit counts aren't present
    - Include a page header once per page
    - Place images inline at their positions with bbox info
    - Use inline markers for font size, bold, italic
    """
    lines_out = []

    # Group elements by page
    pages = {}
    for el in elements:
        page_no = el.get("page", 1)
        pages.setdefault(page_no, []).append(el)

    for page_no in sorted(pages.keys()):
        page_elems = pages[page_no]
        # Sort by visual order
        page_elems.sort(key=lambda e: (e.get("top", e["box"]["t"]), e["box"].get("l", 0)))

        # Median line height per page (gap fallback)
        heights = [
            (el["box"]["b"] - el["box"]["t"])
            for el in page_elems
            if el.get("type") == "text" and "box" in el
        ]
        median_height = statistics.median(heights) if heights else 12.0
        threshold = median_height * gap_multiplier

        # Page header (once)
        lines_out.append(f"[PAGE={page_no}]")

        prev_bottom = None
        active_size = None  # track active font size block

        for el in page_elems:
            top = el["box"]["t"]
            bottom = el["box"]["b"]

            # Explicit breaks BEFORE, else gap fallback
            lb_before = int(el.get("line_breaks_before", 0) or 0)
            if lb_before > 0:
                lines_out.extend([""] * lb_before)
            else:
                if prev_bottom is not None:
                    gap = top - prev_bottom
                    if gap > threshold:
                        lines_out.append("")

            if el["type"] == "text":
                words_out = []

                # Compute line font size (median of words)
                word_sizes = [w.get("font_size") for w in el.get("words", []) if w.get("font_size")]
                line_size = statistics.median(word_sizes) if word_sizes else None

                # Emit size tag when size changes
                if line_size and line_size != active_size:
                    if active_size:
                        words_out.append("</s>")
                    words_out.append(f"<s={int(line_size)}>")
                    active_size = line_size

                # Render words with style markers
                for w in el.get("words", []):
                    text = w.get("text", "")
                    bold = w.get("bold", False)
                    italic = w.get("italic", False)

                    if bold and italic:
                        words_out.append(f"*_ {text} _*")
                    elif bold:
                        words_out.append(f"*{text}*")
                    elif italic:
                        words_out.append(f"_{text}_")
                    else:
                        words_out.append(text)

                # Do not strip to preserve trailing spaces if present
                line_str = " ".join(words_out)
                lines_out.append(line_str)
                prev_bottom = bottom

            elif el["type"] == "table":
                lines_out.append("[TABLE]")
                for row in el.get("table", []):
                    lines_out.append(" | ".join(str(cell) for cell in row))
                lines_out.append("[/TABLE]")
                prev_bottom = bottom

            elif el["type"] == "image":
                bx = el.get("box", {})
                lines_out.append(
                    f"[IMAGE page={page_no} l={bx.get('l', 0):.1f} t={bx.get('t', 0):.1f} "
                    f"r={bx.get('r', 0):.1f} b={bx.get('b', 0):.1f}]"
                )
                prev_bottom = bottom

            # Explicit breaks AFTER
            lb_after = int(el.get("line_breaks_after", 0) or 0)
            if lb_after > 0:
                lines_out.extend([""] * lb_after)

        # Close active size at end of page
        if active_size:
            lines_out.append("</s>")
            active_size = None

        # Page separator
        lines_out.append("")

    # Trim trailing blanks
    while lines_out and lines_out[-1] == "":
        lines_out.pop()

    return "\n".join(lines_out)

# --- STEP 3: Endpoint ---
@app.route('/parse-pdf', methods=['POST'])
def parse_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "File is not a PDF"}), 400

    try:
        file_bytes = file.read()
        structured = []

        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for i, page in enumerate(pdf.pages):
                page_elems = assemble_elements(file_bytes, page, i)  # ✅ pass both args
                                # annotate with page number and page dims
                for el in page_elems:
                    el["page"] = i + 1
                    el["page_width"] = page.width
                    el["page_height"] = page.height
                structured.extend(page_elems)

        simplified = build_simplified_view_from_elements(structured)

        # Debug preview
        print("\n[DEBUG] Simplified view (preview):\n")
        print(simplified[:5000])

        # Collect images from structured
        # images = [el for el in structured if el.get("type") == "image"]
        # import base64

        # for img in images:
        #     print(f"[IMAGE] page={img.get('page')}, box={img.get('box')}, subtype={img.get('subtype')}")
        #     if "image_b64" in img:
        #         print(f"Base64 preview: {img['image_b64'][:80]}...")  # Print first 80 chars for preview

        return jsonify({
            "simplified": simplified,
            "structured": structured
        }), 200  # ✅ explicit success code

    except Exception as e:
        import traceback
        # print("[ERROR]", traceback.format_exc())  # ✅ print full stacktrace for debugging
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


def _pdf_lines_for_match(structured):
    # Flatten text lines in reading order
    lines = [el for el in structured if el.get("type") == "text" and el.get("text")]
    lines.sort(key=lambda e: (e.get("page", 0), e.get("box", {}).get("t", 0), e.get("box", {}).get("l", 0)))
    out = []
    for el in lines:
        out.append({
            "text": el.get("text", ""),
            "page": el.get("page"),
            "box": el.get("box")
        })
    return out

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _match_chunk_to_lines(chunk_text, pdf_lines, start_idx=0):
    """
    Greedy sequential substring matching across consecutive lines.
    Returns: (page, bbox_union, next_idx) or (None, None, start_idx) if no match.
    """
    target = _normalize(chunk_text)
    if not target:
        return None, None, start_idx

    for i in range(start_idx, len(pdf_lines)):
        combined = _normalize(pdf_lines[i]["text"])
        j = i
        while True:
            # exact/substring in either direction
            if target == combined or target in combined or combined in target:
                l = min(line["box"]["l"] for line in pdf_lines[i:j+1])
                t = min(line["box"]["t"] for line in pdf_lines[i:j+1])
                r = max(line["box"]["r"] for line in pdf_lines[i:j+1])
                b = max(line["box"]["b"] for line in pdf_lines[i:j+1])
                page = pdf_lines[i]["page"]
                return page, {"l": l, "t": t, "r": r, "b": b}, j + 1
            if j + 1 >= len(pdf_lines):
                break
            j += 1
            combined = _normalize(combined + " " + pdf_lines[j]["text"])
    return None, None, start_idx

def _anchor_chunks_to_pdf(result_chunks, structured, image_bindings, source_filename):
    """
    - For image chunks: assign page/box from image_bindings (order preserved).
    - For text chunks: sequentially match to pdf lines and union bbox.
    - For table chunks: map to next table on same page, else next global.
    - Add general metadata: source_file, processed_date, etc.
    """
    from datetime import datetime
    
    anchored = []
    pdf_lines = _pdf_lines_for_match(structured)
    idx = 0  # pointer into pdf_lines for sequential matching

    # General metadata for all chunks
    processing_date = datetime.now().isoformat()
    
    # Prepare tables by page in reading order
    tables = [el for el in structured if el.get("type") == "table"]
    tables.sort(key=lambda e: (e.get("page", 0), e.get("box", {}).get("t", 0)))
    tables_by_page = {}
    for tb in tables:
        p = tb.get("page")
        tables_by_page.setdefault(p, []).append(tb)
    table_ptr = {p: 0 for p in tables_by_page.keys()}
    global_tb_ptr = 0

    img_ptr = 0  # pointer into image_bindings

    for ch in result_chunks:
        md = ch.get("metadata", {}) or {}
        ctype = (md.get("type") or "").lower()

        # Default copy-through
        out = dict(ch)
        out_md = dict(md)

        # Add general metadata to ALL chunks
        out_md["source_file"] = source_filename or "unknown.pdf"
        out_md["processed_date"] = processing_date
        
        # Keep all existing metadata from sample_output.json
        # The dict(md) above preserves: context, tags, continues, is_page_break, siblings, etc.

        if ctype == "image":
            if img_ptr < len(image_bindings):
                bind = image_bindings[img_ptr]
                img_ptr += 1
                out_md["page"] = bind["page"]
                out_md["box"] = bind["box"]
                # optional: carry image dims
                out_md["image_width"] = bind.get("box", {}).get("r", 0) - bind.get("box", {}).get("l", 0)
                out_md["image_height"] = bind.get("box", {}).get("b", 0) - bind.get("box", {}).get("t", 0)
            else:
                # leave page as-is if model provided, but note missing binding
                out_md.setdefault("page", None)
                out_md["box"] = None

        elif ctype == "table":
            # Prefer a table on declared page
            target_page = out_md.get("page")
            picked = None
            if target_page in tables_by_page:
                tp = table_ptr.get(target_page, 0)
                if tp < len(tables_by_page[target_page]):
                    picked = tables_by_page[target_page][tp]
                    table_ptr[target_page] = tp + 1
            # Fallback: next global
            if not picked and global_tb_ptr < len(tables):
                picked = tables[global_tb_ptr]
                global_tb_ptr += 1
            if picked:
                out_md["page"] = picked.get("page")
                out_md["box"] = picked.get("box")
            else:
                out_md.setdefault("page", None)
                out_md["box"] = None

        else:
            # Texty types: heading/paragraph/list etc.
            page, box, idx = _match_chunk_to_lines(ch.get("text", ""), pdf_lines, idx)
            if page is not None:
                out_md["page"] = page
                out_md["box"] = box
            else:
                out_md.setdefault("page", None)
                out_md["box"] = None

        out["metadata"] = out_md
        anchored.append(out)

    return anchored

@app.route('/test-anchoring', methods=['POST'])
def test_anchoring():
    """Test route to apply anchoring to existing output with real PDF data"""
    print("🧪 [DEBUG] /test-anchoring endpoint called!")
    
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    simplified_view = data.get('simplified_view', '')
    structured = data.get('structured', [])
    source_filename = data.get('source_filename', 'unknown.pdf')  # Add this
    
    print(f"[DEBUG] Received structured data: {len(structured)} elements")
    print(f"[DEBUG] Source filename: {source_filename}")
    
    try:
        # Load your existing chunked output
        with open("sample_output.json", "r") as f:
            result = json.load(f)
        
        print(f"[DEBUG] Loaded sample output with {len(result.get('chunks', []))} chunks")
        
        # Create empty image_bindings since you're not using OpenAI
        image_bindings = []
        
        # Apply anchoring with the real PDF structured data + filename
        anchored_chunks = _anchor_chunks_to_pdf(
            result.get("chunks", []), 
            structured, 
            image_bindings, 
            source_filename=source_filename
        )
        result["chunks"] = anchored_chunks
        
        # Add document-level metadata too
        result["document_metadata"] = {
            "source_file": source_filename,
            "processed_date": datetime.now().isoformat(),
            "total_chunks": len(anchored_chunks),
            "processing_version": "1.0"
        }
        
        # Save the anchored result
        # with open("anchored_output.json", "w") as f:
        #     json.dump(result, f, indent=2)
        # print("[DEBUG] anchored_output.json created successfully")
        
        print("\n[DEBUG] Anchored result (first chunk metadata):")
        if anchored_chunks:
            print(json.dumps(anchored_chunks[0]["metadata"], indent=2))
        
        return jsonify(result), 200
        
    except FileNotFoundError:
        return jsonify({"error": "sample_output.json not found. Please ensure the file exists."}), 404
    except Exception as e:
        print(f"[ERROR] Error in test anchoring: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# Also add a simple route to just load and return your sample output:
@app.route('/load-sample', methods=['GET'])
def load_sample():
    """Load and return the sample output without anchoring"""
    try:
        with open("sample_output.json", "r") as f:
            result = json.load(f)
        return jsonify(result), 200
    except FileNotFoundError:
        return jsonify({"error": "sample_output.json not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# @app.route('/chunk-pdf', methods=['POST'])
# def chunk_pdf_content():
#     print("[DEBUG] Starting chunk_pdf_content function")

#     if not request.is_json:
#         print("[DEBUG] Request is not JSON")
#         return jsonify({"error": "Request must be JSON"}), 400
    
#     print("[DEBUG] Request is JSON, getting data...")
#     data = request.get_json()
#     simplified_view = data.get('simplified_view', '')
#     structured = data.get('structured', [])

#     print(f"[DEBUG] Data received - simplified_view length: {len(simplified_view)}, structured length: {len(structured)}")

#     # Collect images with payload
#     images = [el for el in structured if el.get("type") == "image" and el.get("image_b64")]
#     print(f"[DEBUG] Found {len(images)} images with base64 data")

#     # Optional debug
#     print(f"[DEBUG] /chunk-pdf: structured={len(structured)}, images(with b64)={len(images)}")

#     # --- Interleave text and images in the exact order of [IMAGE ...] markers ---
#     import re
#     image_pat = re.compile(r"\[IMAGE\s+page=(\d+)\s+l=([\d.]+)\s+t=([\d.]+)\s+r=([\d.]+)\s+b=([\d.]+)\]")

#     MAX_CHARS = 20000
#     MAX_IMAGES = 24
#     used_chars = 0
#     used_images = 0
#     contents = []
#     image_bindings = []

#     def push_text(txt: str):
#         nonlocal used_chars
#         if not txt:
#             return
#         remain = MAX_CHARS - used_chars
#         if remain <= 0:
#             return
#         chunk = txt[:remain]
#         contents.append({"type": "text", "text": chunk})
#         used_chars += len(chunk)

#     print("[DEBUG] About to process image markers...")

#     # Group available images by page
#     images_by_page = {}
#     for im in images:
#         p = im.get("page")
#         images_by_page.setdefault(p, []).append(im)

#     print(f"[DEBUG] Images grouped by page: {list(images_by_page.keys())}")

#     # Determine marker order per page (asc/desc by t) and sort images accordingly
#     # If markers decrease by t, use desc; else asc.
#     markers_by_page = {}
#     for m in image_pat.finditer(simplified_view):
#         p = int(m.group(1))
#         t = float(m.group(3))
#         markers_by_page.setdefault(p, []).append(t)

#     print(f"[DEBUG] Markers found on pages: {list(markers_by_page.keys())}")

#     def is_desc(ts):
#         # decide by majority of adjacent deltas
#         if len(ts) < 2:
#             return False
#         dec = sum(1 for i in range(1, len(ts)) if ts[i] < ts[i-1])
#         inc = sum(1 for i in range(1, len(ts)) if ts[i] > ts[i-1])
#         return dec > inc

#     for p, lst in images_by_page.items():
#         desc = is_desc(markers_by_page.get(p, []))
#         lst.sort(key=lambda e: e["box"]["t"], reverse=desc)

#     # Pointers per page to consume images in order
#     pointers = {p: 0 for p in images_by_page.keys()}

#     idx = 0
#     for m in image_pat.finditer(simplified_view):
#         start, end = m.span()
#         # text before the marker
#         push_text(simplified_view[idx:start])

#         page = int(m.group(1))

#         # Keep the marker text so structure remains visible to the model
#         push_text(simplified_view[start:end] + "\n")

#         # Attach next image for this page (by order)
#         if used_images < MAX_IMAGES and page in images_by_page:
#             pos = pointers.get(page, 0)
#             if pos < len(images_by_page[page]):
#                 im = images_by_page[page][pos]
#                 pointers[page] = pos + 1
#                 contents.append({
#                     "type": "image_url",
#                     "image_url": {"url": f"data:image/png;base64,{im['image_b64']}"}
#                 })
#                 image_bindings.append({"page": im.get("page"), "box": im.get("box")})
#                 used_images += 1

#         idx = end

#     # Trailing text after last marker (or whole text if no markers)
#     push_text(simplified_view[idx:])

#     # Debug
#     for p, lst in images_by_page.items():
#         print(f"[DEBUG] Page {p}: markers_used={pointers.get(p,0)} of {len(lst)} images")
#     print(f"[DEBUG] Interleaved -> text_chars={used_chars}, images_attached={used_images}")


#     # Messages array
#     messages = [
#         {
#             "role": "system",
#             "content": f"""You are a PDF content analyzer that outputs structured JSON.
#             Your task is to:
#             1. Analyze the input content (text and attached images).
#             2. Split it into logical chunks based on:
#                - Section boundaries
#                - Topic changes
#                - Visual formatting (headings, paragraphs, lists, tables, images)
#             3. Output a JSON object with the following schema:

#             {JSON_SCHEMA}

#             Formatting notes for the input you will receive:
#             - Inline markers in the text are **for your analysis only**. Do NOT include them in your final JSON output.
#             - Markers you may see:
#               *word* → bold
#               _word_ → italic
#               *_word_* → bold+italic
#               <s=XX> ... </s> → font size start/end
#             - These markers indicate formatting styles of the original document.
#             - Preserve the original text and line breaks exactly, but strip out these markers in your JSON output.
#             - For images, analyze the attached image content and provide `caption` and put it in text field + `context` in metadata.
#             - Always fill metadata fields: type, section, context (≤1 sentence), tags, continues, is_page_break, siblings, and page number.
#             """
#         },
#         {
#             "role": "user",
#             "content": contents if contents else [{"type": "text", "text": simplified_view[:MAX_CHARS]}]
#         }
#     ]

#     print(f"[DEBUG] Messages created. Content items: {len(contents)}")
#     print("[DEBUG] About to call GPT API...")



#     try:
#             # Call GPT-4o multimodal
#         response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=messages,
#             response_format={ "type": "json_object" },
#             temperature=0.0
#         )

#         print("[DEBUG] GPT API call successful!")

#         print(f"\n[DEBUG] Raw GPT response content:")
#         print("=" * 50)
#         print(response.choices[0].message.content)
#         print("=" * 50)
        
#         result = json.loads(response.choices[0].message.content)

#         # Write sample output BEFORE any further processing
#         with open("sample_output.json", "w") as f:
#             json.dump(result, f, indent=2)
#         print("[DEBUG] sample_output.json created successfully")

#         # --- Map model chunks back to PDF positions (page + bbox) ---
#         anchored_chunks = _anchor_chunks_to_pdf(result.get("chunks", []), structured, image_bindings)
#         result["chunks"] = anchored_chunks
#         pprint(result)

#         # Debug log
#         print("\n[DEBUG] GPT Response:")
#         print("==================")
#         # print(json.dumps(result, indent=2))
#         print("==================\n")
#         return jsonify(result), 200
#     except json.JSONDecodeError as e:
#         print(f"\n[ERROR] Failed to parse JSON: {str(e)}")
#         print(f"Raw response: {response.choices[0].message.content}")
        
#         # Still try to save the raw response for debugging
#         try:
#             with open("raw_response.txt", "w") as f:
#                 f.write(response.choices[0].message.content)
#             print("[DEBUG] Saved raw response to raw_response.txt")
#         except Exception as save_err:
#             print(f"[ERROR] Could not save raw response: {save_err}")
            
#         return jsonify({"error": "Failed to parse model output as JSON"}), 500
#     except Exception as e:
#         print(f"\n[ERROR] Unexpected error: {str(e)}")
#         print(f"Error type: {type(e)}")
#         import traceback
#         traceback.print_exc()
#         return jsonify({"error": f"Internal server error: {str(e)}"}), 500


# You can also add a schema validator not a validator not sure yet
# Add date inside metadata later and the source file. 
JSON_SCHEMA = """{
  "chunks": [
    {
      "text": "string",
      "metadata": {
        "type": "heading|paragraph|list|table|image",
        "section": "string",
        "context": "string (maximum one sentence)",
        "tags": ["string"],
        "row_index": "integer (for tables only) or null",
        "continues": "boolean",
        "is_page_break": "boolean",
        "siblings": ["string"],  
        "page": "integer"
      }
    }
  ]
}"""

if __name__ == '__main__':
    app.run(debug=True)