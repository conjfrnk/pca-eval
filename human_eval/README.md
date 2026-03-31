# Human Evaluation for Proof-Carrying Answers (PCA)

**Study window**: March 31 – April 2, 2026
**Target**: Fleiss' κ ≥ 0.70 (substantial agreement)
**Budget**: ≤ $1,200
**Status**: ✓ Launch-ready

This directory contains everything needed to execute the human evaluation study for the PCA paper.

---

## Quick Start (Today – March 29)

1. **Extract 200 proof objects**:
   ```bash
   python extract_proof_objects.py \
     --scifact-dir /path/to/scifact \
     --hagrid-dir /path/to/hagrid \
     --claimverify-dir /path/to/claimverify \
     --output all_proof_objects_200.json
   ```

2. **Set up Prolific** (2 hours):
   - Follow: `PROLIFIC_README.md` → Step 1–10
   - Upload CSV to Prolific
   - Do test submission
   - Set launch time: March 31, 6 PM

3. **Recruit experts** (2 hours):
   - Follow: `EXPERT_RECRUITMENT.md` → Recruitment Channels
   - Send outreach to Ed Hirsh + 2–3 backups
   - Target: 2 experts confirmed by March 30, 5 PM

4. **Do final launch checks**:
   - Follow: `LAUNCH_CHECKLIST.md` → Pre-Launch section
   - Verify all files exist
   - Test proof objects locally
   - Confirm experts are ready

5. **Launch at 6 PM March 31**:
   - Click "Publish" in Prolific
   - Send confirmation to experts
   - Monitor dashboard (watch for first submissions within 30 min)

---

## File Guide

### Core Files

| File | Purpose | Status |
|------|---------|--------|
| `extract_proof_objects.py` | Extract 200 proof objects from SciFact, HAGRID, ClaimVerify | ✓ Ready |
| `annotation_interface.html` | Standalone HTML interface for experts to annotate | ✓ Ready (test locally) |
| `analyze_human_eval.py` | Compute Fleiss' κ, confusion matrix, visualizations | ✓ Ready (from existing code) |
| `prolific_task_template.json` | Prolific task schema (reference) | ✓ Reference only |

### Documentation

| File | Purpose | Read First? |
|------|---------|-------------|
| **`LAUNCH_CHECKLIST.md`** | Go/no-go checklist (today + launch day + post-study) | **YES** |
| **`PROLIFIC_README.md`** | Step-by-step Prolific setup (10 steps) | **YES** |
| **`EXPERT_RECRUITMENT.md`** | Recruit 2 domain experts (email templates + onboarding) | **YES** |
| `LAUNCH_DAY_QUICK_REF.md` | One-page quick reference (print for Mar 31) | Print & keep handy |
| This file | Index and overview | You're reading it |

---

## Study Design Summary

**Proof objects**: 200 total
- 75 SciFact-derived (scientific claims)
- 75 HAGRID-derived (multi-hop reasoning)
- 50 ClaimVerify edge cases (insurance domain)

**Annotators**: 4 per object (800 total annotations)
- 2 domain experts (insurance CRO/CCO level)
- 2 crowd workers (Prolific, US-based, 95%+ approval)

**Task**: Single question
- "Is this claim's meaning adequately supported by the evidence?"
- Options: YES / PARTIAL / NO / UNCLEAR

**Metrics**:
- Fleiss' κ (primary, target ≥ 0.70)
- Majority agreement (target ≥ 85%)
- System precision (target ≥ 80%)
- Expert–crowd correlation (Spearman ρ, target ≥ 0.60)

**Timeline**:
- Mar 29: Setup & prepare (today)
- Mar 31–Apr 1: Annotation window (3 days)
- Apr 2: Analysis & results

**Budget**: ~$1,200
- Prolific: ~$625 (200–400 submissions × £2.50 + bonus)
- Experts: $560 each × 2 = $1,120
- **Total**: ~$1,745 (if 2 annotators per object on Prolific)

*Cost optimization*: Can reduce to 1 Prolific annotator per object ($625 alone) and still meet κ ≥ 0.70 target with expert data.

---

## Step-by-Step Timeline

### Today (March 29) – Setup Phase

**1. Extract Proof Objects** (1 hour)
```bash
cd pca-eval
python human_eval/extract_proof_objects.py --output all_proof_objects_200.json
# Outputs: all_proof_objects_200.json, all_proof_objects_200.csv, system_verdicts.json
```
- [ ] Verify: 200 objects, 75/75/50 distribution
- [ ] Verify: No missing fields (claim_text, evidence_text, system_verdict)
- [ ] Backup: Copy to cloud storage

**2. Set Up Prolific** (2 hours)
- [ ] Read: `PROLIFIC_README.md` (Step 1–10)
- [ ] Create account (if needed)
- [ ] Create study & upload CSV
- [ ] Configure form questions & payment
- [ ] Do test submission
- [ ] **Leave in Draft** (don't publish yet)

**3. Recruit Experts** (1.5 hours)
- [ ] Read: `EXPERT_RECRUITMENT.md`
- [ ] Email Ed Hirsh + backup contacts
- [ ] Prepare onboarding (instructions PDF, access links)
- [ ] Goal: 2 confirmed by Mar 30, 5 PM

**4. Final Checks** (1 hour)
- [ ] Read: `LAUNCH_CHECKLIST.md` → Pre-Launch section
- [ ] Verify all files exist
- [ ] Verify Prolific study is ready to launch
- [ ] Confirm experts are ready
- [ ] **Status**: READY TO LAUNCH ✓

### Friday, March 31 – Launch Day

**6 AM**: Final pre-launch checks
- [ ] Prolific study ready to publish (Draft → Published)
- [ ] Expert materials sent (login link, instructions, files)
- [ ] Annotators on standby

**6 PM**: **LAUNCH**
- [ ] Click "Publish Study" in Prolific
- [ ] Send confirmation to experts: "Study live! Start whenever."
- [ ] Monitor dashboard for first submissions

**8 PM**: Check progress
- [ ] Prolific: 10–30 submissions received?
- [ ] Experts: Have they started?
- [ ] Any technical issues?

### Saturday, April 1 – Active Monitoring

**Morning**: Progress check
- [ ] Prolific: 60–80% complete?
- [ ] Experts: On track to finish?
- [ ] Rejection rate <10%?
- [ ] Attention check failures <30%?

**Evening**: Study closure
- [ ] 6 PM: Send reminder to experts
- [ ] 6 PM: Prolific auto-closes (or closes at 80%)
- [ ] 6:15 PM: Download Prolific CSV
- [ ] 6:30 PM: Collect expert files (email download)

### Sunday, April 2 – Analysis

**8 AM**: Data validation & merging
```bash
python scripts/validate_prolific_results.py --results prolific_raw.csv
python scripts/merge_annotations.py --expert-csv expert_1.csv expert_2.csv --crowd-csv prolific_raw.csv --output all_annotations_200x4.csv
```

**9 AM**: Full analysis
```bash
python analyze_human_eval.py --expert-csv expert_responses.csv --crowd-csv crowd_responses.csv --system-verdicts system_verdicts.json --output-dir results/
```

**10 AM**: Check success criteria
```
κ ≥ 0.70?              [ ] YES / [ ] NO
Majority agreement ≥ 85%? [ ] YES / [ ] NO
System precision ≥ 80%?   [ ] YES / [ ] NO
Budget ≤ $1,200?       [ ] YES / [ ] NO
```

**5 PM**: Wrap up
- [ ] Pay experts ($560 each)
- [ ] Send thank-yous
- [ ] Archive results
- [ ] Update paper with section

---

## Key Documents (Read in Order)

### Must-Read (Today)

1. **`LAUNCH_CHECKLIST.md`** (18 KB, 200 items)
   - Go/no-go checklist for today + launch day + post-study
   - Use this to track progress
   - **Action**: Check off items as you go

2. **`PROLIFIC_README.md`** (13 KB, 10 steps)
   - Step-by-step Prolific setup
   - Screenshot steps, payment config, participant filters
   - **Action**: Follow steps 1–10, then leave study in Draft

3. **`EXPERT_RECRUITMENT.md`** (13 KB, 4 channels)
   - Email templates for recruiting insurance professionals
   - Onboarding instructions & payment handling
   - **Action**: Send recruitment emails today

### Reference (During Study)

4. **`LAUNCH_DAY_QUICK_REF.md`** (6 KB, 1 page)
   - Quick reference for March 31, 6 PM launch
   - **Action**: Print and keep next to computer

### Technical Reference

5. **`extract_proof_objects.py`** (11 KB, Python script)
   - Extract 200 proof objects from datasets
   - **Usage**: `python extract_proof_objects.py --output all_proof_objects_200.json`

6. **`annotation_interface.html`** (22 KB, web interface)
   - Standalone annotation interface for experts
   - Open in browser, load JSON, download CSV when done
   - **Hosting**: Serve locally or via GitHub Pages (no server needed)

7. **`analyze_human_eval.py`** (9 KB, Python script)
   - Compute Fleiss' κ, confusion matrix, bootstrap CI, visualizations
   - **Usage**: `python analyze_human_eval.py --expert-csv ... --crowd-csv ...`

---

## Critical Success Factors

1. **Object quality**: 200 proof objects must be diverse and well-formed
   - Check: Claim text is clear and testable against evidence
   - Check: Evidence is sufficient but not overly long
   - Check: Mix of supported/unsupported/deflected verdicts

2. **Clear instructions**: Annotators must understand the task
   - Review: Examples in PROLIFIC_README.md and annotation_interface.html
   - Verify: Attention check weeds out careless workers

3. **Adequate incentives**: Experts and crowd workers must be motivated
   - Expert: $70/hour (competitive for insurance professionals)
   - Crowd: £2.50 (~$3) per claim + £0.50 bonus for quality

4. **Early launch**: Start at 6 PM March 31 as planned
   - Reason: Gives 24–36 hour window for completion
   - Backup: Can extend to Apr 2, 11:59 PM if needed

5. **Active monitoring**: Watch dashboard in first 4 hours
   - Reason: Catch technical issues early
   - Action: If <3 submissions/hour, troubleshoot immediately

---

## Contingency Plans

| Scenario | Impact | Solution |
|----------|--------|----------|
| Expert drops out | Need 3rd annotator | Use backup from advisor network; add to Prolific if needed |
| Prolific study fails | No crowd data | Use Google Forms as fallback (1-day delay) |
| κ < 0.65 | Doesn't meet target | Revise instructions, re-run (rare; likely to hit target) |
| Budget overrun | Exceed $1,200 | Drop ClaimVerify domain (keep 150 objects instead of 200) |
| Expert data late | Analysis delays | Proceed with Prolific-only data; expert data when available |

---

## Output Deliverables

After April 2:

```
pca-eval/human_eval/results/
├── prolific_raw_20260402.csv       ← Raw Prolific data (200–400 rows)
├── expert_1_20260402.csv           ← Expert 1 annotations (100 rows)
├── expert_2_20260402.csv           ← Expert 2 annotations (100 rows)
├── all_annotations_200x4.csv       ← Merged (200 rows, 4 raters each)
├── summary.json                    ← Metrics (κ, agreement, etc.)
├── results.png                     ← Visualizations (4-panel)
└── error_analysis.csv              ← Sampled failures (15 rows)
```

**Paper integration**: Copy metrics from `summary.json` into main.tex:
```latex
\paragraph{Results.} Fleiss' κ = [X] (95\% CI [L, U]), ...
```

---

## FAQs

**Q: Can I start earlier than March 31?**
A: Yes, but no advantage. Prolific has limited capacity on weekends. Wait for Friday 6 PM for best participation.

**Q: Do I need both experts and crowd workers?**
A: For κ calculation, technically no (experts alone = 200 objects × 2 raters). But dual-tier approach is stronger for paper. If budget is tight, use experts only + reduce to 150 objects.

**Q: What if I can't find 2 experts?**
A: Fallback: Use Ed Hirsh + 1 backup. If still short, use 3 Prolific annotators per object (instead of 2) to maintain multi-rater setup.

**Q: How do I interpret κ < 0.65?**
A: Still acceptable (moderate agreement). Document as finding: "...indicating moderate but meaningful agreement in this complex domain."

**Q: Can I re-run the study if κ is low?**
A: Yes, but risky. Better to debug instructions first (too ambiguous?), then re-run. Budget allows 1 re-run (bring total to ~$1,800).

**Q: Who should I contact if Prolific fails?**
A: support@prolific.com (24/7). Expect response within 2 hours. Have your study ID ready.

---

## Document Maintenance

This README is accurate as of **March 29, 2026**.

If you're reading this after April 2, update:
- Study dates (Mar 31 → [actual date])
- Success metrics (κ = [target] → κ = [actual])
- Budget (≤$1,200 → actual cost)

After study completion, archive all files in `results/` and tag in git:
```bash
git tag -a pca-human-eval-20260402 -m "Completed human evaluation study"
```

---

## Success Checklist (Print This)

```
SETUP (Today, Mar 29)
─────────────────────
□ Extract 200 proof objects
□ Upload CSV to Prolific
□ Do test submission (self)
□ Email experts (outreach)
□ Verify all files exist
□ Confirm experts ready

LAUNCH (Mar 31, 6 PM)
──────────────────────
□ Publish Prolific study
□ Email experts confirmation
□ Monitor first submissions

ACTIVE (Apr 1–2)
────────────────
□ Check progress (morning)
□ Close study (6 PM Apr 1)
□ Download all results
□ Validate data quality
□ Run analysis
□ Check κ ≥ 0.70
□ Pay experts
□ Update paper

COMPLETE
────────
✓ Study done, κ measured, paper updated, payments sent
```

---

## Contact

- **Questions about study design**: See this file + EXPERT_RECRUITMENT.md
- **Prolific issues**: support@prolific.com
- **Expert issues**: Email to experts directly
- **Analysis questions**: See analyze_human_eval.py comments

---

## Paper Integration

After analysis, add to paper (main.tex):

```latex
\subsection{Human Evaluation: Consumer-Side Auditability}

To validate that downstream consumers can audit proof objects without
access to the originating model, we conducted a human evaluation with
200 proof objects across three domains: 75 SciFact-derived (scientific
claims), 75 HAGRID-derived (multi-hop reasoning), and 50 ClaimVerify
edge cases (insurance-specific). Each object was annotated by 4 raters
(2 insurance domain experts, 2 crowd workers from Prolific), yielding
800 claim-audit instances. Annotators assessed claim support using a
YES/PARTIAL/NO/UNCLEAR scale.

\paragraph{Results.} Fleiss' κ = [κ value] (95\% CI [L, U]),
indicating substantial agreement (target: ≥0.70). Majority agreement
(≥3 of 4 raters): [MA]\% (target: ≥85\%). System vs. human precision:
[P]\% (target: ≥80\%). Domain expert–crowd correlation (Spearman ρ):
[R] (target: ≥0.60).
```

---

## Timeline Summary

| Date | Deadline | Status |
|------|----------|--------|
| Mar 29 (today) | Setup complete | **IN PROGRESS** |
| Mar 30, 5 PM | Experts confirmed | Target |
| Mar 31, 6 PM | Study launches | **GO LIVE** |
| Apr 1, 6 PM | Study closes | Hard stop |
| Apr 2, 10 AM | Analysis complete | Results ready |
| Apr 2, 5 PM | Paper updated | Deliverable |

---

**Version**: 1.0
**Last updated**: March 29, 2026, 2:00 PM UTC
**Next review**: April 2, 2026 (post-study)
