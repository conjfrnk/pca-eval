# Prolific Study Setup: Insurance Claim Verification

**Study window**: March 31 – April 2, 2026
**Annotators needed**: 200 crowd workers (2 per object)
**Budget**: ~$600 (150–200 submissions × $3 per completion + quality bonus)
**Timeline to go-live**: ~2 hours

---

## Quick Start

1. **Create Prolific account** (if needed) at https://www.prolific.com
2. **Create new study** with settings below
3. **Upload CSV** of 200 proof objects
4. **Set payment & quality rules**
5. **Launch at 6 PM Friday, March 31**

---

## Step-by-Step Setup

### Step 1: Log In & Create Study

1. Go to https://www.prolific.com → Researcher Dashboard
2. Click **"Create New Study"**
3. Fill in basic info:
   - **Title**: "Insurance Claim Verification Task"
   - **Description**: "Review short insurance claims and evaluate whether they are supported by provided evidence sources."
   - **Study Type**: "Questionnaire" (standard form-based study)

---

### Step 2: Configure Task Instructions

1. In the **"Instructions"** section, paste the following:

```
TASK: Evaluate Claim Support

You will review short insurance-related claims and decide whether they are
well-supported by their source evidence.

For each claim, you will see:
1. A claim (a statement about insurance, coverage, or risk)
2. A source text passage that the system used as evidence
3. A simple 4-choice question

IMPORTANT: Judge the claim's MEANING based on the evidence provided, not
on external knowledge.

RESPONSE OPTIONS:
- YES: The claim is clearly supported by the evidence. No reasonable doubt.
- PARTIAL: The claim is mostly supported, but there's a minor logical gap,
  heavy paraphrasing, or context needed from elsewhere.
- NO: The claim is contradicted by the evidence, unsupported, or requires
  external knowledge you don't have.
- UNCLEAR: You cannot determine from the evidence provided whether the claim
  is true or false.

EXAMPLES:

Example 1:
Claim: "BRCA1 mutations increase breast cancer risk."
Evidence: "Individuals carrying BRCA1 pathogenic variants face cumulative
breast cancer risk >70% by age 80."
Your answer: YES (Evidence clearly supports the claim)

Example 2:
Claim: "The policy covers medical expenses up to $500K."
Evidence: "Annual aggregate limit: $500,000. Excludes experimental treatments."
Your answer: YES (Clear support, though there are exclusions)

Example 3:
Claim: "This insurer has better coverage than its competitors."
Evidence: (none provided, or not mentioned in the evidence)
Your answer: UNCLEAR (This requires information not in the evidence)

TIME ESTIMATE: 2–3 minutes per claim.
BONUS: High-quality responses (correct attention check, thoughtful comments)
may receive £0.50 bonus on future tasks.

If you're unsure, choose UNCLEAR rather than guessing.
```

---

### Step 3: Configure Study Design

1. **Study Format**: Select **"Single Page"** (all claims shown one-at-a-time)
2. **Data Format**: Select **"CSV upload"** (you'll upload the proof objects)
3. **Iterating over rows**: Enable **"Iterate CSV rows into separate tasks"**
   - Each row becomes one task instance
   - Annotator sees one claim per task submission

---

### Step 4: Upload Proof Objects (CSV)

1. Export the proof objects as CSV:
   ```bash
   python extract_proof_objects.py --output all_proof_objects_200.json
   # Generates all_proof_objects_200.csv
   ```

2. In Prolific, go to **"Data"** tab
3. Click **"Upload CSV"**
4. Select `all_proof_objects_200.csv`
5. Verify columns are recognized:
   - `object_id` → Task ID
   - `claim_text` → Claim text
   - `evidence_text` → Evidence passage
   - `scenario` → (optional, for filtering)
   - `system_score` → (optional, metadata)
   - `system_verdict` → (optional, metadata)

---

### Step 5: Build the Form (Questions)

In **"Study Content"** → **"Add Question"**, add the following:

**Question 1: Main verdict**
- Type: Multiple choice (radio buttons)
- Question text: "Is this claim's meaning adequately supported by the evidence?"
- Options:
  - `YES` — Claim is clearly supported
  - `PARTIAL` — Mostly supported, minor gap
  - `NO` — Contradicted or unsupported
  - `UNCLEAR` — Cannot determine from evidence
- Required: **YES**
- Display: Pull `claim_text` and `evidence_text` from CSV as read-only fields above this question

**Question 2: Optional comments**
- Type: Long text (textarea)
- Question text: "Optional: Any observations about the evidence or claim?"
- Placeholder: "E.g., heavy paraphrase, missing context, multiple interpretations..."
- Required: **NO**
- Max length: 500 characters

**Question 3: Attention check** (shown at the very end of the study)
- Type: Multiple choice (radio buttons)
- Question text: "[Attention Check] What domain did this task focus on?"
- Options:
  - `insurance` — Insurance coverage and claim verification ✓ (correct)
  - `cooking` — Recipe development and cooking techniques
  - `vehicles` — Vehicle maintenance and repair schedules
  - `programming` — Programming software bugs
- Required: **YES**
- Helper text: "(This ensures you read carefully. If incorrect, we may reject your submission.)"

---

### Step 6: Set Participant Filters

1. Go to **"Participants"** tab
2. Set eligibility:
   - **Country**: United States only
   - **Language**: English (native)
   - **Approval rate**: 95% or higher
   - **Prior task count**: 50+ completed
   - **Devices**: Desktop/laptop (no mobile)
   - **Additional**: Add **screening task** (optional, see below)

---

### Step 7 (Optional): Screening Task

If you want to ensure annotators understand the task, create a short screening study:

**Screening Study Title**: "Insurance Claim Evaluation Screening (5 questions)"

Upload a small CSV with 5 representative claim-evidence pairs. Require 4/5 correct.

After annotators complete screening and pass, add them to an **allow list** for the main study.

---

### Step 8: Set Payment & Bonus Rules

1. **Base reward**:
   - Amount: **£2.50** (~$3 USD)
   - Per task: **Per single claim evaluation** (one submission = one claim reviewed)

2. **Bonus rules** (optional):
   - **High quality bonus**: £0.50 for annotators with:
     - Attention check correct
     - Comment provided (if visible)
     - Submission time 1–10 minutes
   - Set to apply to: "All tasks submitted by this participant"

3. **Approval criteria** (auto-reject if):
   - Attention check answer is wrong → Auto-reject with message: "Thank you for participating. Unfortunately, we couldn't approve this submission because the attention check was incorrect. This helps us ensure quality responses."
   - Submission time <30 seconds → Auto-reject: "Your submission was completed too quickly (likely <30 sec per task). We need responses based on careful review."
   - Submission time >15 minutes → Flag for manual review

---

### Step 9: Study Settings

1. **Study duration**:
   - Estimated completion time: **15 minutes** (for 100 claims; Prolific asks for this)
   - Actual: 2–3 min per claim

2. **Number of places** (quota):
   - Set to: **200** (200 unique annotators, or 100 annotators × 2 tasks each)
   - OR set to **400** if you want 2 annotators per claim (preferred for Fleiss' kappa)

3. **Study status**:
   - Set to: **Scheduled** (or **Draft** until you're ready)
   - **Launch date/time**: Friday, March 31, 6 PM GMT (adjust for your timezone)

4. **Completion window**:
   - Allow up to **48 hours** (closes Saturday, April 1, 6 PM)
   - Enable: "Auto-close when 80% submissions received"

---

### Step 10: Review & Launch

1. **Do a test submission**: Complete one task yourself to verify the form works
   - Check that CSV data loads correctly
   - Verify instructions are clear
   - Test submission and feedback

2. **Review all settings**:
   - ✓ Base reward is correct
   - ✓ Filters are set (US, 95%+ approval, 50+ tasks)
   - ✓ Attention check is required and auto-rejects on wrong answer
   - ✓ CSV is uploaded with all 200 rows
   - ✓ Time limits are reasonable

3. **Launch**:
   - Click **"Publish Study"**
   - Prolific will submit to their queue (usually visible within 30 minutes)
   - Monitor submissions in real-time via **"Results"** tab

---

## Monitoring & Quality Control (During Study)

### Real-Time Dashboard

1. Go to **"Results"** tab
2. Watch for:
   - **Submissions rate**: Should see 5–20 per hour initially
   - **Rejection rate**: Flag if >10% getting rejected (indicates unclear instructions)
   - **Attention check failures**: If >20% failing, instructions may be too ambiguous

### Mid-Study Troubleshooting

If attention check failure rate is high (>30%):
1. Edit instructions to clarify the task
2. Or consider lowering attention check rigor (make it less critical)

If submissions are too fast (<30 sec):
1. Increase time minimum to 45 sec
2. Add a comment requirement

---

## After Study Closes (April 2)

### Step 1: Download Results

1. Go to **"Results"** tab
2. Click **"Download all data"** → CSV format
3. File will contain:
   - `object_id` (from CSV)
   - `verdict` (annotator's choice: YES/PARTIAL/NO/UNCLEAR)
   - `comments` (optional text)
   - `attention_check` (CORRECT/WRONG)
   - `submission_time` (in seconds)
   - `participant_id` (Prolific ID)
   - `status` (APPROVED/REJECTED/PENDING)

### Step 2: Validate Data

Run quality checks:

```bash
python validate_prolific_results.py \
    --results-csv prolific_results.csv \
    --system-verdicts system_verdicts.json
```

This script will:
- Flag missing annotations
- Count rejected submissions
- Verify attention check answers
- Compute early submission statistics

### Step 3: Merge Crowd & Expert Annotations

Once you have expert annotations (collected separately), merge them:

```bash
python merge_annotations.py \
    --prolific-csv prolific_results.csv \
    --expert-csv expert_annotations.csv \
    --output all_annotations_200x4.csv
```

Output format:
```
object_id,expert_1,expert_2,crowd_1,crowd_2,scenario,system_verdict
scifact_001,YES,PARTIAL,YES,UNCLEAR,scifact,SUPPORTED
scifact_002,NO,NO,NO,NO,scifact,UNSUPPORTED
...
```

---

## Budget Estimate

| Item | Cost |
|------|------|
| Base: 200 submissions × £2.50 | £500 (~$625) |
| Bonus: ~50 annotators × £0.50 | £25 (~$31) |
| **Total** | **~£525 ($656)** |

To stay under $1,200:
- You can afford 2 annotators per object (400 submissions = ~$1,250 including bonus)
- Or 1 annotator per object (200 submissions = ~$625)

**Recommended**: 2 annotators per object for robust agreement statistics (Fleiss' κ requires multiple raters).

---

## Support & Troubleshooting

| Problem | Solution |
|---------|----------|
| CSV upload fails | Check column names match exactly (`object_id`, `claim_text`, `evidence_text`) |
| Annotators see wrong claim | Verify CSV row order is correct; clear Prolific cache (browser) |
| Too many rejections | Clarify instructions; check attention check isn't too strict |
| Submissions too fast | Increase minimum time; add required comments |
| Low approval rate from participants | May indicate task is unclear; send message asking for feedback |

For Prolific support: https://www.prolific.com/contact

---

## Files You'll Need

- ✓ `all_proof_objects_200.csv` — Exported from `extract_proof_objects.py`
- ✓ `system_verdicts.json` — For analysis post-study
- ✓ These instructions (screenshot for annotators, if needed)

---

## Next Steps

1. **Day 1 (Mar 29)**: Prepare this setup, upload CSV
2. **Day 2 (Mar 30)**: Do test run, review results, adjust filters/instructions
3. **Day 3 (Mar 31, 6 PM)**: **Launch live**
4. **Day 4 (Apr 1)**: Monitor, send reminder if needed
5. **Day 5 (Apr 2, 6 AM)**: Download results, validate
6. **Days 6–7**: Merge with expert data, run analysis

---

## Checklist Before Launch

- [ ] Prolific account created & verified
- [ ] CSV file ready: `all_proof_objects_200.csv` (200 rows)
- [ ] Column names correct: `object_id`, `claim_text`, `evidence_text`, `scenario`, `system_score`, `system_verdict`
- [ ] Study title & description set
- [ ] Instructions copied (with examples)
- [ ] CSV uploaded to Prolific
- [ ] Form questions created (verdict, comments, attention check)
- [ ] Participant filters set (US, 95%+, 50+ tasks, English)
- [ ] Base reward set to £2.50
- [ ] Attention check auto-rejects wrong answers
- [ ] Study status: **Scheduled** for March 31, 6 PM
- [ ] Test run completed (1 full submission, yourself)
- [ ] Results dashboard is accessible
- [ ] You have contact info for expert annotators (separate recruitment)
- [ ] Backup plan if Prolific fails (Google Forms fallback ready)
