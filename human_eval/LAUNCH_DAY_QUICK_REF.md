# Launch Day Quick Reference

**Friday, March 31, 6 PM UTC**

Keep this page open during launch.

---

## Critical Files

| File | Location | Status |
|------|----------|--------|
| `all_proof_objects_200.csv` | `pca-eval/human_eval/` | ✓ Ready for Prolific upload |
| `all_proof_objects_200.json` | `pca-eval/human_eval/` | ✓ Ready for annotation interface |
| `system_verdicts.json` | `pca-eval/human_eval/` | ✓ For post-analysis |
| `annotation_interface.html` | `pca-eval/human_eval/` | ✓ Send to experts |
| `analyze_human_eval.py` | `pca-eval/human_eval/` | ✓ Analysis ready |

---

## 6 PM Launch Sequence

1. **Open Prolific → Your Study → Settings**
2. **Verify**:
   - Title: "Insurance Claim Verification Task"
   - CSV uploaded: ✓ (200 rows)
   - Form questions: ✓ (verdict, comments, attention check)
   - Filters: ✓ (US, 95%+, 50+ tasks)
   - Payment: ✓ (£2.50 base + £0.50 bonus)
   - Auto-reject: ✓ (attention check wrong)
3. **Click "Publish Study"**
4. **Confirm launch time: 6 PM UTC**
5. **Email experts**: "Study live! You're good to start."
6. **Monitor dashboard** for first submissions (watch 8–10 PM)

---

## During Study (Mar 31 – Apr 1)

### Mar 31, 6 PM – Apr 1, 9 AM
- **Check Prolific every 4 hours**
- Target: 5–20 submissions/hour
- Flag if: <3/hour or >50% rejections

### Apr 1, 9 AM
- **Email experts**: "How's progress? On track for 6 PM?"

### Apr 1, 6 PM
- **Close study** (auto-closes in Prolific)
- **Collect expert submissions** (email or download)

---

## Post-Closure (Apr 1, 6:15 PM)

```bash
# Download results
cd pca-eval/human_eval/results/

# Validate
python ../scripts/validate_prolific_results.py \
  --results prolific_raw.csv

# Merge
python ../scripts/merge_annotations.py \
  --expert-csv expert_1.csv expert_2.csv \
  --crowd-csv prolific_raw.csv \
  --output all_annotations_200x4.csv

# Analyze
python ../analyze_human_eval.py \
  --expert-csv expert_responses.csv \
  --crowd-csv crowd_responses.csv \
  --system-verdicts ../system_verdicts.json \
  --output-dir .
```

---

## Success Criteria Check

After analysis (Apr 2):

```
Metric                          Target    Actual    Status
─────────────────────────────────────────────────────────
Fleiss' κ (overall)            ≥ 0.70    [ ]       [ ]
Majority agreement             ≥ 85%     [ ]       [ ]
System precision               ≥ 80%     [ ]       [ ]
Budget                         ≤ $1,200  [ ]       [ ]
Expert–crowd correlation (ρ)   ≥ 0.60    [ ]       [ ]
```

If all pass: ✓ STUDY COMPLETE

---

## Contacts (Live Study)

| Role | Name | Contact |
|------|------|---------|
| Expert 1 | [Name] | [Email] |
| Expert 2 | [Name] | [Email] |
| Prolific Support | — | support@prolific.com |

---

## Key Deadlines

- **Mar 31, 6 PM**: Study goes live (Prolific + expert task)
- **Apr 1, 6 PM**: Study closes (auto-close in Prolific, collect expert files)
- **Apr 2, 10 AM**: Analysis complete
- **Apr 2, 5 PM**: Payments sent, thank-yous sent

---

## If Something Goes Wrong

| Problem | Quick Fix |
|---------|-----------|
| Study won't launch | Refresh Prolific page, check internet |
| CSV upload fails | Verify column names match exactly |
| Expert unreachable | Call backup contact immediately |
| Attention check failures >50% | Pause, revise instructions, relaunch next day |
| Expert data missing | Accept Prolific-only data, document in paper |

---

## Results Location

After analysis, results will be in:

```
pca-eval/human_eval/results/
├── all_annotations_200x4.csv    ← Main data file
├── summary.json                 ← Metrics (κ, agreement, etc.)
└── results.png                  ← Visualizations
```

Copy `summary.json` metrics into paper draft.

---

## One-Click Commands

```bash
# Check Prolific ready (run before 6 PM)
open https://www.prolific.com/studies/[YOUR_STUDY_ID]

# Download results (run at Apr 1, 6:05 PM)
# - Prolific: Results → Download CSV
# - Expert 1: Check email for attachment
# - Expert 2: Check email for attachment

# Run full analysis pipeline (run at Apr 2, 8 AM)
cd pca-eval/human_eval && \
  python scripts/validate_prolific_results.py --results results/prolific_raw.csv && \
  python scripts/merge_annotations.py --expert-csv results/expert_1.csv results/expert_2.csv --crowd-csv results/prolific_raw.csv --output results/all_annotations_200x4.csv && \
  python analyze_human_eval.py --expert-csv results/expert_responses.csv --crowd-csv results/crowd_responses.csv --system-verdicts system_verdicts.json --output-dir results/
```

---

## Study Overview (Share with Team)

**What**: Proof-Carrying Answers human evaluation
**When**: Mar 31 – Apr 2, 2026 (3 days)
**Who**: 200 Prolific annotators + 2 domain experts
**What's being evaluated**: 200 proof objects (claims + evidence)
**Task**: Rate if claim is supported by evidence (YES/PARTIAL/NO/UNCLEAR)
**Goal**: Validate that non-ML-expert readers can audit our system (Fleiss' κ ≥ 0.70)
**Budget**: ~$1,200 total
**Outcome**: Human evaluation section for paper (Easter submission target)

---

## Paper Integration

After analysis (Apr 2), add to main.tex:

```latex
\subsection{Human Evaluation: Consumer-Side Auditability}

To validate consumer-side auditability, we conducted a human evaluation
with 200 proof objects across three domains (75 SciFact-derived, 75 HAGRID-
derived, 50 ClaimVerify edge cases). Each object was annotated by 4 raters
(2 insurance domain experts, 2 crowd workers from Prolific), yielding 800
claim-audit instances. Annotators assessed whether each claim was adequately
supported by its evidence using a YES/PARTIAL/NO/UNCLEAR scale.

\paragraph{Results.} Fleiss' κ = [X] (95\% CI [L, U]), indicating substantial
agreement. Majority agreement (≥3 of 4): [Y]\%. System precision: [Z]\%.
Domain expert–crowd correlation (Spearman ρ): [R]. Deflection agreement: [D]\%.

...
```

See `EXPERT_RECRUITMENT.md` → Payment & Closeout for exact template.

---

## Print This Page

If you want a printed quick reference, save as PDF:
- Bookmark this page
- Print (or save to PDF)
- Keep next to computer during launch (Mar 31, 6–8 PM)
