# NeuroGene AI

**Agentic scientific data analysis for neuroimaging and genomics.**

NeuroGene AI accepts raw scientific data files, automatically detects the format, reasons about what analyses are appropriate, executes real Python code, and returns a written scientific interpretation — with a chat interface for follow-up questions.

No pipeline configuration. No manual preprocessing. No algorithm selection. The AI figures it out.

---

## What it does

Drop in any raw scientific file. NeuroGene AI will:

1. **Inspect** the file — detect format, structure, dimensions, data types
2. **Reason** about what analyses make sense for this specific data
3. **Execute** real Python code using domain-appropriate scientific libraries
4. **Interpret** the results in plain scientific language
5. **Answer** follow-up questions in a chat interface

## Supported formats

| Domain | Formats |
|---|---|
| Neuroimaging | NIfTI (`.nii`, `.nii.gz`), DICOM (`.dcm`) |
| Electrophysiology | EDF (`.edf`, `.bdf`), MNE FIF (`.fif`) |
| Genomics | FASTQ (`.fastq`, `.fq`), FASTA (`.fa`, `.fasta`), VCF (`.vcf`) |
| Multi-omics | HDF5 (`.h5`, `.hdf5`), AnnData (`.h5ad`) |
| Tabular | CSV, TSV |

## Architecture

```
neurogene.html          ← Frontend (chat UI, file upload)
agent_backend.py        ← Python backend (file inspection, code execution, Claude API)
```

The frontend is a single HTML file — no build step, no dependencies. Open it in any browser.

The backend is a FastAPI server that:
- Inspects uploaded files using scientific Python libraries
- Sends metadata to Claude AI, which reasons about the data and writes analysis code
- Executes the generated code in a safe namespace
- Forces a final written interpretation from Claude after all code runs
- Streams results back to the frontend

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

## Example analyses

**MRI / NIfTI**
> NeuroGene AI extracts voxel statistics, intensity distributions, regional volume estimates across 8 anatomical octants, and runs clustering to identify structural subgroups.

**EEG / EDF**
> Computes band power (δ, θ, α, β, γ) per channel, inter-hemispheric coherence, RMS amplitude, and classifies cognitive states if labels are present.

**RNA-seq / CSV**
> Identifies differentially expressed gene candidates, runs phenotype classification with H2O AutoML, ranks top discriminating genes by feature importance.

**VCF**
> Parses variant annotations, allele frequencies, functional scores (CADD, SIFT, PolyPhen), and classifies variants by predicted pathogenicity.

## Stack

- **Reasoning** — [Claude AI](https://anthropic.com) (claude-sonnet-4-6)
- **Backend** — [FastAPI](https://fastapi.tiangolo.com) + [uvicorn](https://www.uvicorn.org)
- **AutoML** — [H2O.ai](https://h2o.ai) (H2O-3, open source)
- **Neuroimaging** — [nibabel](https://nipy.org/nibabel/), [nilearn](https://nilearn.github.io), [MNE-Python](https://mne.tools)
- **Genomics** — [biopython](https://biopython.org), [h5py](https://www.h5py.org)
- **ML** — [scikit-learn](https://scikit-learn.org), [scipy](https://scipy.org)
- **Frontend** — Vanilla HTML/CSS/JS, zero dependencies

## Project structure

```
neurogene-repo/
├── neurogene.html          # Frontend — open this in Chrome
├── agent_backend.py        # Python backend — run this first
├── requirements.txt        # Python dependencies
├── neurogene_logo.svg      # Logo
├── .gitignore
└── README.md
```

## Notes

- Raw scientific data files are listed in `.gitignore` and should not be committed to this repo
- Your Anthropic API key should never be committed — use environment variables only
- H2O AutoML requires Java to be installed before running the backend

---

Built by Nazanin — neuroscience researcher & developer.
