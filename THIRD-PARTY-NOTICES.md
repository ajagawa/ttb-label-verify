# Third-Party Notices and Bill of Materials

This prototype is built entirely from permissively licensed open-source components.
**No commercial licenses, subscriptions, per-seat fees, or vendor accounts are required
to build, run, or deploy it.** No component in the application dependency graph carries a
copyleft obligation, a field-of-use restriction, or a commercial-use restriction.

Exact versions are pinned in `requirements.txt` and `package-lock.json`. This document
records license terms and provenance; it is not a substitute for the lockfiles.

A machine-readable SBOM (CycloneDX) can be regenerated at any time — see
[Regenerating this document](#regenerating-this-document).

---

## 1. Runtime — Extraction

All inference runs in-process. No component in this section makes an outbound network
call at runtime.

| Component | License | Purpose |
|---|---|---|
| `rapidocr-onnxruntime` | Apache-2.0 | Text detection and recognition pipeline |
| `onnxruntime` | MIT | Inference runtime for the OCR models |
| PP-OCRv5 model weights (detection, recognition, orientation) | Apache-2.0 | OCR model parameters — see [Note 2](#note-2--model-weight-provenance) |
| `opencv-python-headless` | Apache-2.0 | Image preprocessing and typographic measurement — see [Note 1](#note-1--why-headless-opencv) |
| `numpy` | BSD-3-Clause | Array operations |
| `Pillow` | MIT-CMU (HPND) | Image decoding, EXIF orientation |

**Alternative extraction backend.** The extraction layer sits behind a provider interface
(`ExtractionProvider`). A second implementation using the reference PaddleOCR runtime is
included for comparison:

| Component | License | Purpose |
|---|---|---|
| `paddleocr` | Apache-2.0 | Reference OCR pipeline |
| `paddlepaddle` | Apache-2.0 | Inference framework |

This backend is not enabled by default: it increases the container image by roughly an
order of magnitude and offers no accuracy advantage on this corpus. It is retained to
demonstrate that the provider seam is real and that the extraction layer can be swapped
without touching the verification engine.

---

## 2. Runtime — Verification and API

The verification engine is deterministic and uses no machine-learning components.

| Component | License | Purpose |
|---|---|---|
| Python 3.11+ | PSF-2.0 | Language runtime |
| `fastapi` | MIT | HTTP API |
| `uvicorn` | BSD-3-Clause | ASGI server |
| `pydantic` | MIT | Schema validation for extraction and rule results |
| `python-multipart` | Apache-2.0 | Multipart upload parsing |
| `rapidfuzz` | MIT | Jaro-Winkler / Levenshtein comparison for brand and class-type fields |

---

## 3. Frontend

| Component | License | Purpose |
|---|---|---|
| `react`, `react-dom` | MIT | UI |
| `vite` | MIT | Build tooling |
| `typescript` | Apache-2.0 | Language tooling |

**No webfont dependency.** The interface uses the platform system font stack. This avoids
a font license question entirely, removes an outbound request to a font CDN (relevant to
the restricted-network constraint), and renders in the typeface the user's OS already
tunes for legibility.

---

## 4. Development and Test

Not present in the production container image.

| Component | License | Purpose |
|---|---|---|
| `pytest` | MIT | Test runner |
| `httpx` | BSD-3-Clause | API test client |
| `ruff` | MIT | Lint and format |
| `vitest` | MIT | Frontend tests |
| `playwright` | Apache-2.0 | Visual checks during development only |
| `cyclonedx-bom` | Apache-2.0 | SBOM generation |

---

## 5. Test Fixtures

| Asset | Provenance | Terms |
|---|---|---|
| Synthetic label images (`fixtures/labels/`) | Composed programmatically by `tools/generate_fixtures.py` | Original work, MIT with this repository |
| Fonts used to compose fixtures — Open Sans (400, 700), Libre Baskerville (700), Oswald (700) | `@fontsource/*` 5.3.0 npm packages, vendored unmodified in `fixtures/fonts/` with each family's `OFL.txt`; provenance and SHA-256 hashes in `fixtures/README.md` | SIL Open Font License 1.1. Families chosen with no Reserved Font Name, since Fontsource's subset files may count as modified versions |

The fixture corpus is generated rather than sourced, for three reasons: the ground truth
is known exactly (which makes the expected-outcome assertions meaningful rather than
eyeballed), non-compliant variants can be produced deliberately, and there is no
third-party image licensing to resolve before publishing the repository.

No real, submitted, or proprietary label artwork is included in this repository.

---

## 6. Reference Data

| Asset | Source | Terms |
|---|---|---|
| Government health warning statement text | 27 CFR § 16.21 | Public domain — U.S. Government work, 17 U.S.C. § 105 |
| Warning statement formatting rules | 27 CFR § 16.22 | Public domain |
| Standards of fill, ABV tolerance tables | 27 CFR Parts 4, 5, 7 | Public domain |

Regulatory reference data is stored as a versioned rule set (`rules/ttb-v1.yaml`) rather
than being embedded in code. Every verification result is stamped with the rule-set
version it was evaluated under. See [Note 4](#note-4--regulatory-versioning).

---

## 7. Infrastructure

| Component | License | Notes |
|---|---|---|
| Docker Engine | Apache-2.0 | See [Note 3](#note-3--docker-engine-vs-docker-desktop) |
| Caddy | Apache-2.0 | Reverse proxy, automatic TLS |
| `python:3.11-slim` base image | Mixed (Debian) | See [Note 5](#note-5--base-image-os-packages) |

---

## Notes

### Note 1 — Why headless OpenCV

OpenCV's own source is Apache-2.0. The standard `opencv-python` wheels additionally bundle
GUI toolkit and media codec components under their own terms (LGPL in several cases).
`opencv-python-headless` omits the GUI dependencies, which is the correct build for a
server container regardless — no windowing system is present — and keeps the license
surface limited to Apache-2.0 plus BSD-licensed numerics.

### Note 2 — Model weight provenance

Model weights are licensed separately from the code that loads them, so the grant was
traced rather than assumed. PaddlePaddle's published model cards for the PP-OCR detection
and recognition models declare `apache-2.0` in their front matter, meaning the grant
attaches to the weight files themselves and not only to the surrounding toolkit.

One limitation is worth stating plainly: the training corpus for these models is not
publicly disclosed. The license permits commercial use without restriction, but an agency
procuring a production system may want provenance assurances that the upstream publisher
does not currently provide. This is a general property of pre-trained OCR models rather
than a defect of this particular choice, and it would apply equally to a commercial cloud
OCR service.

### Note 3 — Docker Engine vs. Docker Desktop

Docker Engine and the Moby project are Apache-2.0 with no use restrictions. Docker
*Desktop* is separately licensed and requires a paid per-user subscription for
organizations exceeding 250 employees or $10M in annual revenue; Docker's terms name
government entities explicitly among those requiring licensure.

This distinction matters for deployment planning, not for this prototype: the container
is built and run with Docker Engine on Linux, which carries no such obligation. Nothing
in this repository requires Docker Desktop.

### Note 4 — Regulatory versioning

TTB labeling requirements are subject to active rulemaking — Notices 237 and 238 (January
2025) propose mandatory "Alcohol Facts" and major food allergen disclosures, with a
proposed compliance window of five years after any final rule. The required-field set will
change. The rule set is therefore a versioned data file rather than application logic, and
results carry the version they were evaluated against so that a decision can be reproduced
against the rules in force at the time it was made.

### Note 5 — Base image OS packages

The Debian base image contains operating-system packages under a mixture of licenses,
including GPL and LGPL components. This is a property of every Linux container image. The
application does not link against, derive from, or redistribute modified versions of these
components, so no source-disclosure obligation attaches to the application code. Noted for
completeness, as a federal software review will ask.

---

## License Obligations Summary

| License | Obligation | Satisfied by |
|---|---|---|
| Apache-2.0 | Retain copyright, license, and NOTICE; state significant changes | This document; no upstream components modified |
| MIT / MIT-CMU | Retain copyright and permission notice | This document |
| BSD-3-Clause | Retain copyright and disclaimer; no endorsement claims | This document |
| PSF-2.0 | Retain PSF copyright notice | This document |
| SIL OFL 1.1 | Fonts not redistributed as fonts; reserved names unused | Fixture generation only |

**No obligations to disclose source, pay royalties, or restrict use apply to any component
in the application dependency graph.**

Full license texts for all dependencies are reproduced in `licenses/`, generated by
`pip-licenses` and `license-checker` during the build.

---

## Regenerating this document

```bash
# Python dependency licenses
pip-licenses --format=markdown --with-urls --with-license-file --output-file licenses/python.md

# Frontend dependency licenses
npx license-checker --production --markdown > licenses/node.md

# CycloneDX SBOM
cyclonedx-py environment -o sbom.json
```

An SBOM in CycloneDX format is generated on every build and published as
`sbom.json` alongside the container image, in anticipation of the software supply-chain
attestation requirements that apply to software delivered to federal agencies.

---

## Accounts and Procurement Required

**None**, for the prototype as delivered:

- No API keys, tokens, or vendor accounts.
- No metered or usage-based services.
- No commercial licenses to acquire.
- No outbound network access required at runtime.

The only recurring cost is compute for the hosted demonstration instance
(approximately $5/month on a small VPS). The identical container image runs on an
isolated network with no external connectivity — see `README.md`,
*Running air-gapped*.
