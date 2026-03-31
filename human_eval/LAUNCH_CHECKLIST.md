# Human Evaluation Launch Checklist

**Study window**: March 31 – April 2, 2026 (3 days)
**Launch date**: Friday, March 31, 6 PM UTC
**Target metrics**: Fleiss' κ ≥ 0.70, budget ≤ $1,200
**Prepared by**: Connor (Detent.ai)
**Status**: [ ] READY TO LAUNCH

---

## Pre-Launch (Today – March 29)

### Object Extraction & Preparation

- [ ] **Extract proof objects**
  - Run: `python extract_proof_objects.py --scifact-dir [path] --hagrid-dir [path] --claimverify-dir [path] --output all_proof_objects_200.json`
  - Verify: 200 objects total (75 SciFact, 75 HAGRID, 50 ClaimVerify)
  - Verify: All fields present (object_id, claim_text, evidence_text, scenario, system_score, system_verdict)
  - Verify: No duplicates or missing data
  - Output files: `all_proof_objects_200.json`, `all_proof_objects_200.csv`, `system_verdicts.json`

- [ ] **Validate proof objects**
  - Spot check 10 random objects: claim and evidence are sensible
  - Verify scenario distribution: SciFact (75), HAGRID (75), ClaimVerify (50)
  - Verify verdict distribution: SUPPORTED, UNSUPPORTED, DEFLECTED balanced
  - Check system scores: valid floats in [0.0, 1.0]

- [ ] **Backup proof objects**
  - Copy to external drive or cloud storage
  - Reason: In case of data loss, need to regenerate analysis files

### Prolific Setup

- [ ] **Create Prolific account**
  - Visit https://www.prolific.com
  - Sign up (if needed) with research email
  - Enable two-factor authentication

- [ ] **Create study in Prolific**
  - Title: "Insurance Claim Verification Task"
  - Description: "Review AI-generated insurance claims and evaluate whether they are supported by provided evidence."
  - Study type: Questionnaire

- [ ] **Configure study instructions**
  - Copy instructions from `PROLIFIC_README.md` → Step 2
  - Include examples (BRCA1, policy coverage, etc.)
  - Paste attention check (correct answer: insurance-related)

- [ ] **Upload proof objects CSV**
  - File: `all_proof_objects_200.csv`
  - Columns: object_id, claim_text, evidence_text, scenario, system_score, system_verdict
  - Verify: 200 rows (plus header)
  - Enable: "Iterate CSV rows into separate tasks"

- [ ] **Create form questions**
  - Q1 (Main verdict): Multi-choice radio, 4 options (YES/PARTIAL/NO/UNCLEAR), required
  - Q2 (Comments): Text area, optional, 500 char max
  - Q3 (Attention check): Multi-choice radio, 4 options, required, correct=insurance

- [ ] **Configure participant filters**
  - Country: United States only
  - Language: English (native)
  - Approval rate: ≥95%
  - Prior task count: ≥50
  - Device: Desktop/laptop preferred

- [ ] **Set payment & bonuses**
  - Base reward: £2.50 (~$3 USD) per submission
  - Bonus: £0.50 for high-quality (correct attention, time 1–10 min, comment provided)
  - Auto-rejection: Attention check wrong → message: "...attention check was incorrect..."

- [ ] **Review Prolific settings**
  - Study status: Draft (until final review)
  - Estimated completion time: 15 minutes
  - Quota: 200–400 (depending on 1 vs. 2 annotators per object)
  - Completion window: 48 hours (Mar 31, 6 PM → Apr 2, 6 PM)
  - Auto-close: Yes, when 80% submissions received

- [ ] **Do test submission**
  - Complete the task yourself (as Prolific user, if possible)
  - Verify: CSV data loads correctly
  - Verify: Form renders properly
  - Verify: Can submit + get confirmation
  - Verify: Auto-rejection works (intentionally fail attention check on test)
  - Document any issues

### Expert Recruitment

- [ ] **Draft outreach emails**
  - Primary contact: Ed Hirsh (PwC)
  - Secondary: LinkedIn posts + DMs
  - Template in `EXPERT_RECRUITMENT.md` → Recruitment Channels

- [ ] **Send expert recruitment emails**
  - Email Ed Hirsh + 2–3 backup contacts from advisor network
  - Include: Study details, timeline, compensation ($70/hr, ~8 hrs), task preview
  - Request responses by March 30, 5 PM

- [ ] **Prepare expert onboarding package**
  - Google Form with 100 claims (copy questions from Prolific)
  - OR prepare HTML interface files (annotation_interface.html + all_proof_objects_200.json)
  - Create 1-page PDF instruction handout
  - Prepare login credentials / access links

- [ ] **Set up expert data collection**
  - If Google Form: Create form, enable response CSV download
  - If HTML interface: Test locally, prepare for file download
  - If email-based: Set up shared Google Sheet as fallback

- [ ] **Confirm expert participation**
  - Target: 2 experts confirmed by March 30, 5 PM
  - Backup: 3rd expert lined up in case of dropout

### Analysis & QA Setup

- [ ] **Prepare analysis script**
  - Script: `analyze_human_eval.py` (already exists)
  - Test locally with mock data (100 objects, 4 annotators each)
  - Verify outputs: Fleiss' κ, confusion matrix, bootstrap CI

- [ ] **Prepare validation script**
  - Script: Validate Prolific CSV (completeness, verdict values, etc.)
  - Script: Check expert annotations (same validation)
  - Script: Merge expert + crowd annotations into unified format

- [ ] **Create results directory structure**
  ```
  pca-eval/human_eval/
  ├── results/
  │   ├── prolific_raw.csv        [downloaded from Prolific]
  │   ├── expert_1.csv            [from expert 1]
  │   ├── expert_2.csv            [from expert 2]
  │   ├── all_annotations_200x4.csv [merged]
  │   ├── summary.json            [analysis output]
  │   └── results.png             [visualizations]
  └── ...
  ```

- [ ] **Set up data backup**
  - Cloud storage: Google Drive / Dropbox folder
  - Automated: Set up nightly backup of results/ folder

### Documentation & Comms

- [ ] **Finalize README files**
  - ✓ `PROLIFIC_README.md` (20 checklist items)
  - ✓ `EXPERT_RECRUITMENT.md` (payment, onboarding)
  - ✓ `LAUNCH_CHECKLIST.md` (this file)

- [ ] **Create study description (for paper)**
  - Dataset: 200 proof objects (75 SciFact, 75 HAGRID, 50 ClaimVerify)
  - Annotators: 4 per object (2 domain experts, 2 crowd)
  - Scale: 800 claim-audit instances
  - Question: YES/PARTIAL/NO/UNCLEAR (4-way rating)
  - Metrics: Fleiss' κ (primary), majority agreement, confusion matrix vs. system

- [ ] **Prepare post-study communication template**
  - Thank-you email to experts (with payment info)
  - Attribution offer (name/company in paper)
  - Follow-up participation inquiry

---

## Launch Day (Friday, March 31)

### Morning Checklist (Mar 31, 8 AM)

- [ ] **Final Prolific review**
  - Verify study is in Draft state
  - Re-read instructions for typos
  - Check all form fields are labeled clearly
  - Confirm auto-rejection logic is enabled
  - Verify participant filters match goals (95%+ approval, 50+ tasks)

- [ ] **Send expert confirmation emails**
  - Confirm experts are ready
  - Send study link + password
  - Send 1-page instruction PDF
  - Send proof objects files (JSON for interface, OR pre-filled form)
  - Request confirmation of receipt + planned start time

- [ ] **Verify data files are accessible**
  - all_proof_objects_200.json exists in pca-eval/human_eval/
  - all_proof_objects_200.csv exists (for Prolific upload)
  - system_verdicts.json exists (for analysis)
  - annotation_interface.html is tested and works offline

- [ ] **Do final dry-run**
  - Open annotation_interface.html in browser
  - Load all_proof_objects_200.json
  - Annotate 5 objects, download CSV
  - Verify CSV has correct format (object_id, verdict, comments, timestamp)

### Evening Checklist (Mar 31, 5 PM)

- [ ] **Launch Prolific study**
  - Set study status: Published
  - Verify: Study appears in Prolific queue within 30 min
  - Set launch time: 6 PM UTC (or timezone-adjusted time)
  - Confirm: Prolific sends notifications to eligible participants

- [ ] **Monitor Prolific dashboard**
  - Check: Submissions start flowing in (~5–20/hour)
  - Check: No errors in responses
  - Check: Form validation works (required fields enforced)
  - Check: Attention check auto-rejects working (some failures expected)

- [ ] **Confirm experts are active**
  - Email/Slack with experts
  - Confirm they've received study materials + can access
  - Set expectation: Work anytime Mar 31 – Apr 1, submit by Apr 1, 6 PM
  - Share your contact for technical issues

- [ ] **Enable result downloads**
  - Prolific: Confirm "Download Results" button is visible
  - Google Form (if used): Confirm response collection is on
  - Set reminder to check results tomorrow morning

---

## Active Period (Saturday, April 1)

### Morning (Apr 1, 8 AM)

- [ ] **Check Prolific progress**
  - Goal: 60–80% submissions received by now
  - Check rejection rate (<10% is good)
  - Check attention check failure rate (<30% is good)
  - If rejections high: Send message to Prolific participants clarifying task
  - If failures high: Consider loosening attention check rigor (post-hoc if needed)

- [ ] **Check expert progress**
  - Email experts: "How's it going? On track to finish by 6 PM tonight?"
  - Offer technical support if needed
  - Confirm receipt of any issues/questions

- [ ] **Spot-check response quality**
  - Download a sample of Prolific responses
  - Read 10 comments: Are they thoughtful or generic?
  - Flag any suspicious patterns (identical answers, too-fast submissions)

### Evening (Apr 1, 5 PM)

- [ ] **Send study closure reminder**
  - Email experts: "Reminder: Deadline 6 PM tonight. Please submit if not done."
  - Post message on Prolific (optional): "Study closes in 1 hour. Finish up!"

- [ ] **Monitor for late submissions**
  - Prolific: Auto-closes at 6 PM (or when 80% reached)
  - Experts: Collect submissions by email/form/interface downloads
  - Prolific results: Download full CSV at 6:05 PM

### Post-Closure (Apr 1, 6:15 PM)

- [ ] **Download all Prolific results**
  - Go to Prolific → Results → Download CSV
  - Save as: `results/prolific_raw_20260401.csv`
  - Check file size: ~50–100 KB (should have ~200–400 rows)

- [ ] **Collect expert annotations**
  - Expert 1: Email or download from Google Form / HTML interface
  - Expert 2: Email or download from Google Form / HTML interface
  - Save as: `results/expert_1_20260401.csv`, `results/expert_2_20260401.csv`

- [ ] **Quick validation check**
  - Prolific: 200 rows (or 400 if 2 annotators per object)
  - Expert 1: 100 rows
  - Expert 2: 100 rows
  - All have required columns: object_id, verdict (YES/PARTIAL/NO/UNCLEAR), timestamp

---

## Analysis Phase (Sunday, April 2)

### Data Processing (Apr 2, 8 AM)

- [ ] **Validate data quality**
  ```bash
  python scripts/validate_prolific_results.py \
    --results prolific_raw_20260401.csv \
    --system-verdicts system_verdicts.json
  ```
  - Verify: No missing verdicts
  - Verify: All verdicts in {YES, PARTIAL, NO, UNCLEAR}
  - Flag: Attention check failures (should be <30%)
  - Flag: Suspicious patterns (too fast, too slow, identical answers)

- [ ] **Merge annotations**
  ```bash
  python scripts/merge_annotations.py \
    --expert-csv expert_1.csv expert_2.csv \
    --crowd-csv prolific_raw_20260401.csv \
    --output all_annotations_200x4.csv
  ```
  - Verify output: 200 rows (one per object)
  - Verify columns: object_id, expert_1, expert_2, crowd_1, crowd_2, scenario, system_verdict
  - Backup: Copy to cloud storage

### Analysis & Metrics (Apr 2, 9 AM)

- [ ] **Run primary analysis**
  ```bash
  python analyze_human_eval.py \
    --expert-csv expert_responses.csv \
    --crowd-csv crowd_responses.csv \
    --system-verdicts system_verdicts.json \
    --output-dir results/
  ```
  - Outputs: summary.json, results.png
  - Metrics computed: Fleiss' κ, bootstrap CI, majority agreement, confusion matrix
  - Scenario breakdown: κ per scenario (SciFact, HAGRID, ClaimVerify)

- [ ] **Check success criteria**
  ```
  ✓ Fleiss' κ overall ≥ 0.70? (target: yes)
  ✓ Majority agreement ≥ 85%? (target: yes)
  ✓ System precision ≥ 80%? (target: yes)
  ✓ Budget ≤ $1,200? (target: yes)
  ✓ Expert–crowd correlation ≥ 0.60? (target: yes)
  ```

  **If any metric misses**:
  - κ < 0.65 → Revise instructions (too ambiguous)
  - Precision < 80% → Lower system thresholds
  - Budget overrun → Reduce dataset size (drop ClaimVerify domain)
  - Crowd–expert divergence → Document as key finding

- [ ] **Generate error analysis**
  - Sample 15 objects where system ≠ human majority
  - Document error type: paraphrase, multi-hop, arithmetic, etc.
  - Table: [ID | Claim | Evidence | System | Humans | Error Type]

- [ ] **Create visualizations**
  - Plot 1: Confusion matrix (system vs. human)
  - Plot 2: Kappa by scenario
  - Plot 3: Annotation time distribution
  - Plot 4: System score bins vs. agreement
  - All use Detent color scheme (#1B5E3B, #C4A94F, #D4453A)

### Reporting (Apr 2, 11 AM)

- [ ] **Document results for paper**
  - Copy summary.json metrics into paper.md
  - Include all kappas, CIs, majority agreement
  - Describe error analysis findings
  - Describe expert vs. crowd correlation

- [ ] **Save all artifacts**
  ```
  pca-eval/human_eval/results/
  ├── prolific_raw_20260401.csv
  ├── expert_1_20260401.csv
  ├── expert_2_20260401.csv
  ├── all_annotations_200x4.csv
  ├── summary.json
  ├── results.png
  └── error_analysis.csv
  ```

- [ ] **Create summary memo**
  - Title: "Human Evaluation Results Summary"
  - Date: Apr 2, 2026
  - Headline: "κ = [X], meeting target of ≥0.70"
  - Key findings: 3–5 bullet points
  - Budget: Actual vs. estimated
  - Recommendations for next round (if any)

---

## Post-Study (April 2–7)

### Payment & Follow-Up

- [ ] **Pay experts**
  - Process payment: $560 each via PayPal/ACH
  - Send invoice/confirmation email
  - Thank-you message + offer paper acknowledgment

- [ ] **Close Prolific study**
  - In Prolific dashboard: Mark study as complete
  - Approve all non-rejected submissions
  - Send thank-you message to participants (via Prolific)
  - Option: Offer future study participation

- [ ] **Update paper with results**
  - Add human evaluation section to main.tex
  - Include Table 1b (human eval metrics)
  - Include error analysis (Appendix B)
  - Cite all annotators (if they consent)

### Archive & Cleanup

- [ ] **Archive raw data**
  - Compress: `tar -czf pca-human-eval-results-20260402.tar.gz results/`
  - Upload to: Cloud storage (Google Drive backup)
  - Delete: Any temp files, lock files, .DS_Store

- [ ] **Update MEMORY.md**
  - Entry: "Human eval complete. κ = [X], budget: $[Y]. Paper section updated."
  - Learning: Any insights about task design, annotator behavior, edge cases

- [ ] **Update project status**
  - Mark in MASTER_PLAN.md: "PCA human evaluation: ✓ complete (κ = [X])"
  - Update project timeline if paper timeline shifts

---

## Contingency Plans

### Scenario: Expert Drops Out Before Study

**Action**:
- [ ] Immediately contact backup expert from advisor network
- [ ] Offer expedited completion (can do 100 claims in 1 day at $70/hr)
- [ ] If no response within 2 hours: Use Prolific only (single rater per object)
- [ ] Document in paper: "Expert data incomplete; crowd-only analysis shown"

### Scenario: Prolific Study Fails to Launch

**Action**:
- [ ] Create Google Form with 100 claims (subset for quick pivot)
- [ ] Send form link to Prolific participants manually via email
- [ ] Or: Use Appen / Amazon Mechanical Turk as fallback
- [ ] Timeline: May delay by 12–24 hours, but recoverable

### Scenario: >50% Attention Check Failures

**Action**:
- [ ] Pause study (Mar 31, 8 PM)
- [ ] Revise instructions to clarify task
- [ ] Relaunch (Mar 31, 10 PM) with new wording
- [ ] Accept: May need to increase budget for additional submissions

### Scenario: Fleiss' κ < 0.65

**Action**:
- [ ] Don't panic — substantial agreement still achieved
- [ ] Document in paper: "...κ = 0.63, indicating moderate but not substantial agreement"
- [ ] Analyze: Were certain scenarios easier/harder?
- [ ] Consider: Adding domain expert review (you + Ed Hirsh adjudicate disagreements)
- [ ] Plan: Next iteration with refined task or clearer guidelines

### Scenario: Budget Overrun

**Action**:
- [ ] If Prolific cost exceeds $700: Drop ClaimVerify domain (use only SciFact + HAGRID)
- [ ] Rerun analysis with 150 objects instead of 200
- [ ] Document decision in paper
- [ ] Still achieves goal (κ ≥ 0.70 likely with 2 larger scenarios)

---

## Final Verification Checklist

**Go/No-Go Decision: Friday, March 31, 5 PM**

Answer each:

1. **Proof objects ready?**
   - [ ] 200 objects extracted, validated, backed up
   - [ ] CSV has correct format and 200 rows

2. **Prolific study ready?**
   - [ ] Study in Draft, all fields filled
   - [ ] CSV uploaded, test submission works
   - [ ] Filters set, payment configured, auto-rejection enabled
   - [ ] Study can launch at 6 PM with 1 click

3. **Experts confirmed?**
   - [ ] 2 experts confirmed by email
   - [ ] Onboarding materials sent (link + instructions)
   - [ ] Backup expert on standby

4. **Analysis pipeline ready?**
   - [ ] analyze_human_eval.py tested locally
   - [ ] Merge script ready
   - [ ] Results directory structure set up
   - [ ] Cloud backup configured

5. **Communication ready?**
   - [ ] Post-study emails drafted
   - [ ] Paper template ready for results
   - [ ] Memory update planned

**If ALL checked**: ✓ GO LIVE
**If ANY unchecked**: ✗ HOLD — fix before 6 PM launch

---

## Contact & Support

- **Prolific issues**: support@prolific.com
- **Expert issues**: Direct email to experts (24/7)
- **Technical questions**: Your email
- **Post-study**: Archive contact info for follow-up studies

---

## Success Looks Like

**March 31, 6 PM**: Prolific study launches, first submissions come in within 30 min
**April 1, 6 PM**: 80% submissions received, both experts submit data
**April 2, 10 AM**: Analysis complete, κ = [≥0.70], paper section drafted
**April 2, 5 PM**: All payments sent, thank-you emails sent, results archived

→ **Proof-Carrying Answers human evaluation: ✓ COMPLETE**
