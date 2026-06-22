# NeuroGene AI

**Agentic scientific data analysis for neuroimaging and genomics — with automatic literature grounding.**

NeuroGene AI accepts raw scientific data files, automatically detects the format, reasons about what analyses are appropriate, executes real Python code, returns a written scientific interpretation, and grounds findings in published literature from PubMed — all in one step.

No pipeline configuration. No manual preprocessing. No algorithm selection. The AI figures it out.

---

## What it does

Drop in any raw scientific file. NeuroGene AI will:

1. **Inspect** the file — detect format, structure, dimensions, data types
2. **Reason** about what analyses make sense for this specific data
3. **Execute** real Python code using domain-appropriate scientific libraries
4. **Interpret** the results in plain scientific language
5. **Ground** findings in published literature — searches PubMed and Semantic Scholar automatically, retrieves relevant papers, and compares your results against existing research
6. **Answer** follow-up questions in a chat interface

---

## Supported formats

| Domain | Formats |
|---|---|
| Neuroimaging | NIfTI (`.nii`, `.nii.gz`), DICOM (`.dcm`) |
| Electrophysiology | EDF (`.edf`, `.bdf`), MNE FIF (`.fif`) |
| Genomics | FASTQ (`.fastq`, `.fq`), FASTA (`.fa`, `.fasta`), VCF (`.vcf`) |
| Multi-omics | HDF5 (`.h5`, `.hdf5`), AnnData (`.h5ad`) |
| Tabular | CSV, TSV |

---

## Features

- **Agentic analysis** — Claude reasons about your data and decides what to do, rather than following fixed rules
- **Literature grounding** — after every analysis, automatically searches PubMed and Semantic Scholar, retrieves up to 6 relevant papers, and writes a literature context paragraph comparing your findings to published research
- **Persistent sessions** — all conversations are saved in your browser. Close and reopen the app and everything is exactly where you left it
- **Multiple sessions** — run analyses on different files simultaneously, switch between them, rename sessions by double-clicking
- **Clean reports** — results are shown as written scientific reports only. Code runs silently in the background. Type "show me the code" in chat if you want to see it
- **Follow-up chat** — ask questions about your results after the analysis completes

---

## Architecture

```
neurogene.html       ← Frontend (chat UI, file upload, sessions)
agent_backend.py     ← Python backend (file inspection, code execution, Claude API, literature search)
```

The frontend is a single HTML file — no build step, no dependencies. Open it in any browser.

The backend is a FastAPI server that:
- Inspects uploaded files using scientific Python libraries
- Sends metadata to Claude AI, which reasons about the data and writes analysis code
- Executes the generated code in a safe namespace
- Forces a final written interpretation after all code runs
- Searches PubMed and Semantic Scholar for relevant papers
- Asks Claude to compare findings against retrieved abstracts
- Returns everything to the frontend

---

## Getting started

### Requirements
- Python 3.9+
- Java 11+ (required for H2O AutoML) — download from [adoptium.net](https://adoptium.net)
- An Anthropic API key — get one at [console.anthropic.com](https://console.anthropic.com)

### Install

```bash
pip3 install -r requirements.txt
```

### Run

```bash
ANTHROPIC_API_KEY=your_key_here python3 agent_backend.py
```

Then open `neurogene.html` in your browser. The sidebar will show **Backend: Ready**.

Enter your Anthropic API key via the **API Key** button in the top right, drop a file, and click **Analyse with AI**.

---

## Example output

**After analysing an MRI NIfTI file:**

> The structural MRI data revealed significant volumetric asymmetry across the 8 anatomical octants, with the posterior left superior region showing 14% greater volume than its contralateral counterpart...

**Literature context (auto-generated):**

> Your finding of reduced hippocampal volume is consistent with Smith et al. (2021), who reported similar reductions in early-stage Alzheimer's patients. However, the elevated posterior cortical thickness you observed diverges from the majority of current literature — only 2 of 6 retrieved papers report this pattern, suggesting this may warrant further investigation...
>
> 6 relevant papers retrieved — direct PubMed links included.

---

## Stack

- **Reasoning** — [Claude AI](https://anthropic.com) (claude-sonnet-4-6)
- **Backend** — [FastAPI](https://fastapi.tiangolo.com) + [uvicorn](https://www.uvicorn.org)
- **AutoML** — [H2O.ai](https://h2o.ai) (H2O-3, open source)
- **Literature** — [PubMed E-utilities API](https://www.ncbi.nlm.nih.gov/books/NBK25501/) + [Semantic Scholar API](https://www.semanticscholar.org/product/api) (both free, no key required)
- **Neuroimaging** — [nibabel](https://nipy.org/nibabel/), [nilearn](https://nilearn.github.io), [MNE-Python](https://mne.tools)
- **Genomics** — [biopython](https://biopython.org), [h5py](https://www.h5py.org)
- **ML** — [scikit-learn](https://scikit-learn.org), [scipy](https://scipy.org)
- **Frontend** — Vanilla HTML/CSS/JS, zero dependencies

---

## Project structure

```
neurogene-repo/
├── neurogene.html       # Frontend — open this in Chrome
├── agent_backend.py     # Python backend — run this first
├── requirements.txt     # Python dependencies
├── .gitignore
└── README.md
```

---

Built by Nazanin — neuroscience researcher & developer.
