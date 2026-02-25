# Model Documentation Report — Design Document

**Date:** 2026-02-20
**Project:** Marketing Analytics — Send-Time Optimization
**Status:** Approved

---

## Problem Statement

The project has three notebooks covering a full send-time optimization pipeline and several short reference docs (`model_card.md`, `validation_report.md`, `synthesis_methodology.md`, `data_dictionary.md`). These are brief and technical. A formal model documentation report is needed to communicate the full project to both business and technical stakeholders in one cohesive document.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Audience | Mixed (executive summary + technical depth) | Report serves both marketing stakeholders and DS peers |
| Literature review scope | Focused (4 papers cited in notebooks) | Concise and directly justified by design choices |
| Structure | Option A — single long-form narrative | Cohesive reading experience; existing short docs are not duplicated |
| Output path | `project/docs/model_documentation.md` | Alongside existing reference docs |

## Section Specifications

| # | Section | Audience | Target length |
|---|---------|----------|---------------|
| 1 | Executive Summary | Business | ~400 words |
| 2 | Literature Review | Mixed | ~500 words |
| 3 | Model Data | Technical | ~600 words + tables |
| 4 | Model Methodology | Technical | ~700 words + tables |
| 5 | Model Performance | Mixed | ~600 words + tables |
| 6 | Model Risk & Future Enhancements | Mixed | ~500 words |

## Source Material

All metrics sourced from executed notebook outputs:

| Metric | Value | Source |
|--------|-------|--------|
| AUC (LightGBM) | 0.5235 | `02_optimal_time_engagement_model.ipynb` cell 5 |
| AUC (Logistic) | 0.5114 | same |
| AUC (Naive) | 0.5039 | same |
| Brier score | 0.178859 | same |
| Top-3 hit rate | 5.07% | same cell 6 |
| Precision@3 | 34.5% | `03_client_engagement_diagnostics.ipynb` cell 16 |
| Recall@5 | 45.3% | same |
| ECE | 6.8% | same cell 14 |
| Policy: fixed blast | 23.2% | same cell 21 |
| Policy: personalized | 38.0% | same |
| Slot stability | 5.1% | same cell 12 |
| Reactivation 30d | 1.9% | same cell 7 |
| UK proportion | 91.9% | `01_eda_and_synthesis.ipynb` cell 18 |
| Null CustomerID | 22.8% | same |

## Existing Docs (cross-reference only)

- [`data_dictionary.md`](../data_dictionary.md) — feature definitions
- [`synthesis_methodology.md`](../synthesis_methodology.md) — synthetic data generation
- [`validation_report.md`](../validation_report.md) — raw metrics tables
- [`model_card.md`](../model_card.md) — brief model card
