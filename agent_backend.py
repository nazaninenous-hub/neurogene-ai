#!/usr/bin/env python3
"""
NeuroGene Agentic Backend
=========================
Claude reasons about your data, writes analysis code,
executes it, and returns structured results + interpretation.

Install:
    pip3 install fastapi uvicorn python-multipart anthropic \
                 nibabel pydicom mne biopython h5py scipy \
                 scikit-learn matplotlib pandas numpy h2o

Run:
    python3 agent_backend.py
"""

import io, os, sys, json, time, tempfile, traceback, warnings, base64, textwrap
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# ── Anthropic ────────────────────────────────────────────────────────
try:
    import anthropic
    ANTHROPIC_CLIENT = anthropic.Anthropic()
except Exception:
    ANTHROPIC_CLIENT = None

app = FastAPI(title="NeuroGene Agent", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ══════════════════════════════════════════════════════════════════════
#  FILE INSPECTION — raw metadata without full parsing
# ══════════════════════════════════════════════════════════════════════

def inspect_file(content: bytes, filename: str) -> dict:
    """Return lightweight metadata about any file so Claude can reason about it."""
    name = filename.lower()
    size_mb = len(content) / 1024 / 1024
    info = {"filename": filename, "size_mb": round(size_mb, 2), "format": "unknown", "details": {}}

    # ── HDF5 ──
    if name.endswith(('.h5','.hdf5','.he5','.h5ad')) or content[:4] == b'\x89HDF':
        info["format"] = "hdf5"
        try:
            import h5py, io as _io
            f = h5py.File(_io.BytesIO(content), 'r')
            datasets, groups = [], []
            def walk(node, path=''):
                for key in node.keys():
                    fp = f"{path}/{key}"
                    item = node[key]
                    if isinstance(item, h5py.Dataset):
                        datasets.append({"path": fp, "shape": list(item.shape), "dtype": str(item.dtype)})
                    elif isinstance(item, h5py.Group):
                        groups.append(fp)
                        walk(item, fp)
            walk(f)
            # Sample a small piece of the first numeric dataset
            sample_data = None
            for ds in datasets:
                if any(t in ds['dtype'] for t in ['float','int','uint']) and len(ds['shape']) >= 1:
                    try:
                        arr = np.array(f[ds['path']][:])
                        flat = arr.flatten()[:20]
                        sample_data = {"path": ds['path'], "values": [round(float(v),4) for v in flat]}
                        break
                    except: pass
            f.close()
            info["details"] = {"datasets": datasets[:15], "groups": groups[:10], "n_datasets": len(datasets), "sample": sample_data}
        except Exception as e:
            info["details"] = {"error": str(e)}

    # ── NIfTI ──
    elif name.endswith(('.nii','.nii.gz')):
        info["format"] = "nifti"
        try:
            import nibabel as nib
            with tempfile.NamedTemporaryFile(suffix='.nii.gz' if name.endswith('.gz') else '.nii', delete=False) as tmp:
                tmp.write(content); tmp_path = tmp.name
            img = nib.load(tmp_path)
            data = img.get_fdata()
            info["details"] = {
                "shape": list(data.shape), "voxel_size": [round(float(v),3) for v in img.header.get_zooms()[:3]],
                "dtype": str(data.dtype), "value_range": [round(float(data.min()),2), round(float(data.max()),2)],
                "n_nonzero": int(np.sum(data != 0)), "is_4d": data.ndim == 4
            }
            os.unlink(tmp_path)
        except Exception as e:
            info["details"] = {"error": str(e)}

    # ── DICOM ──
    elif name.endswith('.dcm'):
        info["format"] = "dicom"
        try:
            import pydicom, io as _io
            ds = pydicom.dcmread(_io.BytesIO(content))
            info["details"] = {
                "modality": str(ds.get("Modality","")), "rows": int(ds.get("Rows",0)),
                "cols": int(ds.get("Columns",0)), "bits": int(ds.get("BitsAllocated",0)),
                "tr": float(ds.get("RepetitionTime",0) or 0), "te": float(ds.get("EchoTime",0) or 0),
            }
        except Exception as e:
            info["details"] = {"error": str(e)}

    # ── EDF ──
    elif name.endswith(('.edf','.bdf')):
        info["format"] = "edf"
        try:
            import mne
            mne.set_log_level('ERROR')
            with tempfile.NamedTemporaryFile(suffix='.edf', delete=False) as tmp:
                tmp.write(content); tmp_path = tmp.name
            raw = mne.io.read_raw_edf(tmp_path, preload=False, verbose=False)
            info["details"] = {
                "n_channels": len(raw.ch_names), "channel_names": raw.ch_names[:20],
                "sfreq": raw.info['sfreq'], "duration_sec": round(raw.times[-1],1),
                "n_times": len(raw.times)
            }
            os.unlink(tmp_path)
        except Exception as e:
            info["details"] = {"error": str(e)}

    # ── FASTQ ──
    elif name.endswith(('.fastq','.fq','.fastq.gz')):
        info["format"] = "fastq"
        try:
            text = content[:4096].decode('utf-8', errors='ignore')
            lines = [l for l in text.split('\n') if l.strip()]
            n_reads_sample = sum(1 for l in lines if l.startswith('@'))
            seq_sample = lines[1] if len(lines) > 1 else ""
            info["details"] = {"reads_in_preview": n_reads_sample, "sample_read_length": len(seq_sample), "sample_sequence": seq_sample[:60]}
        except Exception as e:
            info["details"] = {"error": str(e)}

    # ── FASTA ──
    elif name.endswith(('.fasta','.fa','.fna')):
        info["format"] = "fasta"
        try:
            text = content[:4096].decode('utf-8', errors='ignore')
            seqs = [l for l in text.split('\n') if l.startswith('>')]
            info["details"] = {"n_sequences_preview": len(seqs), "first_headers": [s[:80] for s in seqs[:5]]}
        except Exception as e:
            info["details"] = {"error": str(e)}

    # ── VCF ──
    elif name.endswith('.vcf'):
        info["format"] = "vcf"
        try:
            text = content[:8192].decode('utf-8', errors='ignore')
            lines = [l for l in text.split('\n') if not l.startswith('##') and l.strip()]
            header = next((l for l in lines if l.startswith('#CHROM')), '')
            data_lines = [l for l in lines if not l.startswith('#')]
            info["details"] = {"n_variants_preview": len(data_lines), "columns": header.lstrip('#').split('\t')[:10], "sample_line": data_lines[0][:200] if data_lines else ""}
        except Exception as e:
            info["details"] = {"error": str(e)}

    # ── CSV/TSV ──
    elif name.endswith(('.csv','.tsv','.txt')):
        info["format"] = "csv"
        try:
            sep = '\t' if name.endswith('.tsv') else ','
            df = pd.read_csv(io.BytesIO(content), sep=sep, nrows=5)
            info["details"] = {
                "columns": list(df.columns), "n_cols": len(df.columns),
                "dtypes": {c: str(t) for c,t in df.dtypes.items()},
                "sample_rows": df.head(3).to_dict('records')
            }
        except Exception as e:
            info["details"] = {"error": str(e)}

    return info


# ══════════════════════════════════════════════════════════════════════
#  SAFE CODE EXECUTOR
# ══════════════════════════════════════════════════════════════════════

def execute_code(code: str, file_content: bytes, filename: str) -> dict:
    """Execute Python code generated by Claude in a controlled namespace."""
    import io as _io, traceback as _tb

    # Write file to temp
    suffix = os.path.splitext(filename)[-1] or '.bin'
    if filename.lower().endswith('.nii.gz'):
        suffix = '.nii.gz'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_content)
        tmp_path = tmp.name

    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = captured = _io.StringIO()

    result = {"success": False, "output": "", "result": None, "error": "", "figures": []}

    try:
        # Safe namespace with scientific libraries
        import matplotlib
        matplotlib.use('Agg')  # non-interactive
        import matplotlib.pyplot as plt

        namespace = {
            "__builtins__": __builtins__,
            "np": np, "pd": pd,
            "file_path": tmp_path,
            "filename": filename,
            "plt": plt,
            "result": None,
        }

        # Optional imports
        for lib, alias in [("nibabel","nib"),("pydicom","pydicom"),("mne","mne"),
                           ("scipy","scipy"),("sklearn","sklearn"),("h5py","h5py"),
                           ("h2o","h2o")]:
            try: namespace[alias] = __import__(lib)
            except ImportError: pass

        exec(code, namespace)

        result["success"] = True
        result["output"] = captured.getvalue()
        result["result"] = namespace.get("result", None)

        # Capture any matplotlib figures as base64
        figs = []
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            buf = _io.BytesIO()
            fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
            buf.seek(0)
            figs.append(base64.b64encode(buf.read()).decode('utf-8'))
            plt.close(fig)
        result["figures"] = figs

    except Exception as e:
        result["error"] = str(e) + "\n" + _tb.format_exc()
    finally:
        sys.stdout = old_stdout
        try: os.unlink(tmp_path)
        except: pass

    return result


# ══════════════════════════════════════════════════════════════════════
#  LITERATURE SEARCH — PubMed + Semantic Scholar
# ══════════════════════════════════════════════════════════════════════

import urllib.request
import urllib.parse
import urllib.error

def search_pubmed(query: str, max_results: int = 5) -> list:
    """Search PubMed and return list of paper dicts with title, abstract, authors, year, pmid."""
    try:
        # Step 1: search for IDs
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = urllib.parse.urlencode({
            "db": "pubmed", "term": query, "retmax": max_results,
            "retmode": "json", "sort": "relevance"
        })
        req = urllib.request.Request(f"{search_url}?{params}",
                                     headers={"User-Agent": "NeuroGene/2.0"})
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read())
        ids = data.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []

        # Step 2: fetch abstracts for those IDs
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = urllib.parse.urlencode({
            "db": "pubmed", "id": ",".join(ids),
            "rettype": "abstract", "retmode": "json"
        })
        req = urllib.request.Request(f"{fetch_url}?{params}",
                                     headers={"User-Agent": "NeuroGene/2.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            fetch_data = json.loads(r.read())

        articles = fetch_data.get("PubmedArticleSet", {}).get("PubmedArticle", [])
        if isinstance(articles, dict):
            articles = [articles]

        papers = []
        for art in articles:
            try:
                med = art["MedlineCitation"]
                article = med["Article"]
                title = article.get("ArticleTitle", "")
                if isinstance(title, dict):
                    title = title.get("#text", str(title))

                # Abstract
                abstract_obj = article.get("Abstract", {})
                abstract_text = abstract_obj.get("AbstractText", "")
                if isinstance(abstract_text, list):
                    abstract_text = " ".join([
                        (a.get("#text", str(a)) if isinstance(a, dict) else str(a))
                        for a in abstract_text
                    ])
                elif isinstance(abstract_text, dict):
                    abstract_text = abstract_text.get("#text", str(abstract_text))

                # Authors
                author_list = article.get("AuthorList", {}).get("Author", [])
                if isinstance(author_list, dict):
                    author_list = [author_list]
                authors = []
                for a in author_list[:3]:
                    ln = a.get("LastName", "")
                    fn = a.get("ForeName", "")
                    if ln:
                        authors.append(f"{ln} {fn[0]}." if fn else ln)
                author_str = ", ".join(authors) + (" et al." if len(author_list) > 3 else "")

                # Year
                pub_date = article.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
                year = pub_date.get("Year", pub_date.get("MedlineDate", "")[:4])

                pmid = str(med.get("PMID", {}).get("#text", "") or med.get("PMID", ""))

                if title and abstract_text:
                    papers.append({
                        "title": str(title),
                        "abstract": str(abstract_text)[:800],
                        "authors": author_str,
                        "year": str(year),
                        "pmid": pmid,
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
                    })
            except Exception:
                continue

        return papers

    except Exception as e:
        print(f"PubMed search error: {e}")
        return []


def search_semantic_scholar(query: str, max_results: int = 5) -> list:
    """Search Semantic Scholar as fallback."""
    try:
        params = urllib.parse.urlencode({
            "query": query, "limit": max_results,
            "fields": "title,abstract,authors,year,externalIds"
        })
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "NeuroGene/2.0"})
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read())

        papers = []
        for p in data.get("data", []):
            abstract = p.get("abstract") or ""
            title = p.get("title") or ""
            if not title or not abstract:
                continue
            authors = [a.get("name","") for a in p.get("authors", [])[:3]]
            author_str = ", ".join(authors) + (" et al." if len(p.get("authors",[])) > 3 else "")
            pmid = p.get("externalIds", {}).get("PubMed", "")
            papers.append({
                "title": title,
                "abstract": abstract[:800],
                "authors": author_str,
                "year": str(p.get("year", "")),
                "pmid": pmid,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else
                       f"https://api.semanticscholar.org/graph/v1/paper/{p.get('paperId','')}"
            })
        return papers
    except Exception as e:
        print(f"Semantic Scholar error: {e}")
        return []


def extract_search_terms(interpretation: str, file_format: str) -> list:
    """Ask Claude to extract 2-3 good PubMed search queries from the findings."""
    if not ANTHROPIC_CLIENT:
        return []
    try:
        resp = ANTHROPIC_CLIENT.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": f"""Based on these scientific findings from a {file_format} data analysis, 
generate exactly 3 concise PubMed search queries (3-6 words each) that would find the most relevant literature.

Findings:
{interpretation[:1000]}

Return ONLY a JSON array of 3 strings, nothing else. Example: ["query one", "query two", "query three"]"""
            }]
        )
        text = resp.content[0].text.strip()
        # Extract JSON array
        import re as _re
        match = _re.search(r'\[.*?\]', text, _re.DOTALL)
        if match:
            queries = json.loads(match.group())
            return [q for q in queries if isinstance(q, str)][:3]
    except Exception as e:
        print(f"Search term extraction error: {e}")
    return []


def run_literature_grounding(interpretation: str, file_format: str) -> dict:
    """
    Full literature pipeline:
    1. Extract search terms from findings
    2. Search PubMed + Semantic Scholar
    3. Claude compares findings to literature
    Returns dict with papers and literature_context text.
    """
    if not ANTHROPIC_CLIENT:
        return {"papers": [], "literature_context": ""}

    # Step 1: extract search queries
    queries = extract_search_terms(interpretation, file_format)
    if not queries:
        # fallback generic queries based on format
        format_queries = {
            "nifti": ["MRI brain morphometry machine learning", "structural MRI biomarkers"],
            "edf": ["EEG biomarkers classification", "EEG band power brain disorders"],
            "hdf5": ["brain imaging data analysis", "neural data machine learning"],
            "fastq": ["genomic sequencing analysis", "DNA sequence features"],
            "vcf": ["variant pathogenicity prediction", "genetic variant classification"],
            "csv": ["neuroimaging features classification", "brain biomarkers"],
        }
        queries = format_queries.get(file_format, ["neuroscience data analysis machine learning"])

    # Step 2: search for papers
    all_papers = []
    seen_titles = set()
    for query in queries:
        papers = search_pubmed(query, max_results=3)
        if not papers:
            papers = search_semantic_scholar(query, max_results=3)
        for p in papers:
            if p["title"] not in seen_titles:
                seen_titles.add(p["title"])
                all_papers.append(p)
        if len(all_papers) >= 8:
            break

    all_papers = all_papers[:6]  # cap at 6 papers

    if not all_papers:
        return {"papers": [], "literature_context": "No relevant literature found for this analysis."}

    # Step 3: Claude compares findings to literature
    papers_text = "\n\n".join([
        f"Paper {i+1}: {p['authors']} ({p['year']})\nTitle: {p['title']}\nAbstract: {p['abstract']}"
        for i, p in enumerate(all_papers)
    ])

    try:
        resp = ANTHROPIC_CLIENT.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": f"""You are a scientific reviewer. Compare the following analysis findings against the retrieved literature.

ANALYSIS FINDINGS:
{interpretation[:1500]}

RETRIEVED PAPERS:
{papers_text}

Write a concise Literature Context section (3-4 paragraphs) that:
1. Identifies which findings replicate or are consistent with existing literature (cite specific papers by author and year)
2. Identifies any findings that contradict or diverge from the literature
3. Flags anything that appears novel or under-reported
4. Suggests 1-2 follow-up studies or analyses based on the gaps you see

Write in plain scientific prose. No code. No bullet points — paragraphs only. Be specific."""
            }]
        )
        literature_context = resp.content[0].text.strip()
    except Exception as e:
        literature_context = f"Literature comparison unavailable: {e}"

    return {
        "papers": all_papers,
        "literature_context": literature_context,
        "queries_used": queries
    }


# ══════════════════════════════════════════════════════════════════════
#  AGENTIC ANALYSIS — Claude reasons + executes
# ══════════════════════════════════════════════════════════════════════

def run_agentic_analysis(file_info: dict, file_content: bytes, filename: str,
                         user_message: str = "", conversation_history: list = None) -> dict:
    """
    Claude inspects the file metadata, reasons about what to do,
    writes Python code, we execute it, Claude interprets results.
    """
    if ANTHROPIC_CLIENT is None:
        raise RuntimeError("Anthropic client not available. Set ANTHROPIC_API_KEY environment variable.")

    history = conversation_history or []

    system_prompt = """You are an expert scientific data analyst with deep knowledge of:
- Neuroimaging (MRI, fMRI, EEG, MEG) — nibabel, nilearn, MNE-Python
- Genomics (RNA-seq, WGS, VCF) — biopython, pandas, scipy
- Machine learning — scikit-learn, H2O AutoML
- Statistics — scipy.stats, pandas

You have been given metadata about a scientific data file. Your job is to:
1. Understand what the data is
2. Decide what analyses are most appropriate
3. Write Python code to perform those analyses
4. Interpret the results scientifically

IMPORTANT RULES for code:
- The file is available at the variable `file_path` (a string path to a temp file)
- Store your final results in a variable called `result` (dict with keys: metrics, features, summary)
- Use print() to output progress and intermediate findings
- Always handle errors gracefully with try/except
- For HDF5: use h5py to open the file
- For NIfTI: use nib (nibabel)
- For EDF: use mne
- matplotlib figures will be captured automatically — use plt.figure() and plt.savefig() is not needed

When you want to run code, wrap it in a <code> block like:
<code>
# your python code here
result = {"metrics": {}, "features": [], "summary": ""}
</code>

After seeing the execution output, provide a clear scientific interpretation in plain prose.
CRITICAL: Your final interpretation must contain NO code, NO code blocks, NO backtick fences, NO Python syntax.
Write as a scientist reporting findings — paragraphs only, with specific values and conclusions.
Be specific about what you found — not generic. Reference actual values from the output.
"""

    # Build the initial message with file metadata
    file_summary = json.dumps(file_info, indent=2)
    initial_message = f"""I have uploaded a file for analysis.

File metadata:
{file_summary}

{f'User request: {user_message}' if user_message else 'Please explore this data, determine what it contains, decide what analyses are most appropriate, run them, and give me a scientific interpretation of the results.'}

Please examine the metadata carefully, then write Python code to explore and analyze this data."""

    messages = history + [{"role": "user", "content": initial_message}]

    steps = []
    final_interpretation = ""
    all_figures = []
    all_code_outputs = []

    # Agentic loop — Claude reasons, we execute, Claude interprets
    import re
    MAX_ROUNDS = 3  # code execution rounds
    for iteration in range(MAX_ROUNDS):
        response = ANTHROPIC_CLIENT.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            system=system_prompt,
            messages=messages
        )

        assistant_text = response.content[0].text
        messages.append({"role": "assistant", "content": assistant_text})

        # Extract code blocks
        code_blocks = re.findall(r'<code>(.*?)</code>', assistant_text, re.DOTALL)

        if not code_blocks:
            # No more code — this is the final interpretation
            final_interpretation = assistant_text
            break

        # Execute each code block
        for code in code_blocks:
            code = code.strip()
            steps.append({"type": "code", "content": code})

            exec_result = execute_code(code, file_content, filename)
            all_code_outputs.append(exec_result)

            if exec_result["figures"]:
                all_figures.extend(exec_result["figures"])

            output_summary = f"""Code executed.

Output:
{exec_result['output'][:3000] if exec_result['output'] else '(no print output)'}

{'Error: ' + exec_result['error'][:500] if exec_result['error'] else 'No errors.'}

Result variable: {json.dumps(exec_result['result'], default=str)[:1000] if exec_result['result'] else 'Not set'}

{'Figures generated: ' + str(len(exec_result['figures'])) + ' plot(s) captured.' if exec_result['figures'] else ''}"""

            messages.append({"role": "user", "content": output_summary})
            steps.append({"type": "output", "content": output_summary})

    # Always request a final interpretation if we don't have one yet
    if not final_interpretation:
        messages.append({
            "role": "user",
            "content": "Based on all the code you ran and the outputs you received, please now write a clear, detailed scientific interpretation of the results. Do not write any more code — just your expert interpretation and conclusions. Be specific about values, patterns, and what they mean scientifically."
        })
        final_resp = ANTHROPIC_CLIENT.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            system=system_prompt,
            messages=messages
        )
        final_interpretation = final_resp.content[0].text
        messages.append({"role": "assistant", "content": final_interpretation})

    # Strip any code blocks from the final interpretation — report only
    import re as _re
    clean_interp = _re.sub(r'<code>.*?</code>', '', final_interpretation, flags=_re.DOTALL)
    clean_interp = _re.sub(r'```[\s\S]*?```', '', clean_interp)  # markdown code fences
    clean_interp = _re.sub(r'`[^`\n]{1,200}`', lambda m: m.group().replace('`',''), clean_interp)  # inline code — keep text, remove backticks
    clean_interp = clean_interp.strip()

    # Run literature grounding pipeline
    lit = run_literature_grounding(clean_interp, file_info.get("format", "unknown"))

    return {
        "steps": steps,
        "interpretation": clean_interp,
        "figures": all_figures,
        "code_outputs": all_code_outputs,
        "messages": messages,
        "literature": lit,
    }


# ══════════════════════════════════════════════════════════════════════
#  API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    libs = {}
    for lib in ["nibabel","pydicom","mne","h5py","h2o","anthropic","scipy","sklearn"]:
        try: __import__(lib); libs[lib] = True
        except ImportError: libs[lib] = False
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    return {"status": "ok", "version": "2.0", "libraries": libs, "anthropic_key_set": has_key}


@app.post("/inspect")
async def inspect(file: UploadFile = File(...)):
    """Quick file inspection — returns metadata for Claude to reason about."""
    content = await file.read()
    info = inspect_file(content, file.filename or "upload")
    return info


@app.post("/analyze")
async def analyze(
    request: Request,
    file: UploadFile = File(...),
    message: str = Form(""),
    history: str = Form("[]"),
):
    # Allow API key to be passed from browser
    key = request.headers.get("X-Anthropic-Key", "")
    if key:
        os.environ["ANTHROPIC_API_KEY"] = key
        global ANTHROPIC_CLIENT
        try:
            import anthropic as _ant
            ANTHROPIC_CLIENT = _ant.Anthropic(api_key=key)
        except Exception:
            pass
    """Full agentic analysis — Claude reasons, writes code, executes, interprets."""
    content = await file.read()
    filename = file.filename or "upload"

    try:
        conv_history = json.loads(history)
    except Exception:
        conv_history = []

    # Step 1: inspect
    file_info = inspect_file(content, filename)

    # Step 2: agentic loop
    try:
        result = run_agentic_analysis(file_info, content, filename, message, conv_history)
    except RuntimeError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(500, f"Agent error: {e}\n{traceback.format_exc()}")

    return {
        "file_info": file_info,
        "steps": result["steps"],
        "interpretation": result["interpretation"],
        "figures": result["figures"],
        "messages": result["messages"],
    }


@app.post("/chat")
async def chat(
    request: Request,
    message: str = Form(...),
    history: str = Form("[]"),
    file: UploadFile = File(None),
):
    key = request.headers.get("X-Anthropic-Key", "")
    if key:
        os.environ["ANTHROPIC_API_KEY"] = key
        global ANTHROPIC_CLIENT
        try:
            import anthropic as _ant
            ANTHROPIC_CLIENT = _ant.Anthropic(api_key=key)
        except Exception:
            pass
    """Follow-up chat — ask questions about previous results."""
    if ANTHROPIC_CLIENT is None:
        raise HTTPException(422, "ANTHROPIC_API_KEY not set.")

    try:
        conv_history = json.loads(history)
    except Exception:
        conv_history = []

    content = None
    filename = None
    file_info = None
    if file:
        content = await file.read()
        filename = file.filename or "upload"
        file_info = inspect_file(content, filename)

    messages = conv_history + [{"role": "user", "content": message}]

    response = ANTHROPIC_CLIENT.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        system="You are an expert scientific data analyst. Answer questions about previously analyzed data concisely and accurately. Reference specific values and findings.",
        messages=messages
    )

    reply = response.content[0].text
    messages.append({"role": "assistant", "content": reply})

    # Check if Claude wants to run more code
    import re
    code_blocks = re.findall(r'<code>(.*?)</code>', reply, re.DOTALL)
    figures = []
    if code_blocks and content:
        for code in code_blocks:
            exec_result = execute_code(code.strip(), content, filename)
            if exec_result["figures"]:
                figures.extend(exec_result["figures"])
            # append output back
            messages.append({"role": "user", "content": f"Code output: {exec_result['output'][:1000]}"})

    return {"reply": reply, "messages": messages, "figures": figures}


if __name__ == "__main__":
    import uvicorn, platform
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    print("\n" + "="*60)
    print(f"  NeuroGene Agentic Backend  (Python {platform.python_version()})")
    print(f"  Anthropic API key: {'✓ Set' if key else '✗ NOT SET — run: export ANTHROPIC_API_KEY=your_key'}")
    print(f"  Running at  : http://localhost:8000")
    print(f"  Health check: http://localhost:8000/health")
    print(f"  Stop        : Ctrl + C")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
