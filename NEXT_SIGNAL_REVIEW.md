## October 13: targeted hearing validation passed

One GPT-4o-mini reviewer call approved the exact saved proposal: "Alfozan is due to appear in federal court again on Oct. 19 for a detention hearing." The evidence was rebuilt from cached publisher text with zero network retrievals and matched the original writer cache key. One selected source passage supported the action and date. The reviewer marked it qualifying because it directly advances the legal process. Its overall reason was null; the source comparison supplied the explanation, and the normal parser accepted the verdict.

Usage: 1,551 input tokens and 57 output tokens; no retries, writer calls, or full pipeline run. Full response, evidence, code hash, and parsed decision are preserved under `october_13_hearing_present_schedule_validation` in `outputs/next_signals_v18/focused_review.json`. Live summaries, production reports, and operational caches were not updated. Together with the 56 offline tests, this validates the scoped scheduling-language fix; it does not establish perfect recall on other stories or resolve the possible Pope-trip omission.

---

## October 13: present-tense scheduling cue repaired

Scope: Next Signal only. Active version is `next-signal-v20-present-schedule-2026-10-13`. The shared future-framing expression now recognizes the grammatical construction `am/is/are due to`. Previously every `due to` phrase was excluded to avoid causal language, so the exact source-supported October 19 hearing proposal was rejected before semantic review. No event names, dates, or action-verb dictionaries were added.

This is a candidate cue, not a determination that the sentence is a genuine schedule. Present-tense causal phrases can remain ambiguous and must be rejected by the existing semantic reviewer. Past dates, historical `was due to`, grounding, conditional countdown, and uncertainty rules remain in place. The validator and prefilter versions changed so old local rejection caches cannot suppress newly eligible review. Writer prompts, model, call count architecture, daily runner, and other features were not changed.

Validation: all 56 offline tests passed (52 existing plus four new tests covering multiple cases). New checks cover four subjects across three dates/years, the exact saved hearing proposal, past/causal uses, semantic rejection preservation, cache version separation, and the October issue's saved proposals. Mock reviewer approval demonstrates the path works; it is not a fresh model judgment. No API or publisher calls were made.

Cached-only evidence replay covered all 20 stories in the September 25 and October 13 issues. October evidence selection was unchanged; only the hearing's saved proposal changed at the local wording gate. September story 9 gained the source's `is due to attend a prayer vigil` sentence. Its relevance/ordering still requires semantic review; this replay does not claim the old displayed signal remains optimal. The regression expectation was updated to explicitly require that recovered schedule.

Pope October omission: rejection of `The Vatican will confirm ... next month` is justified because the source gives a planned trip date, not a dated confirmation announcement. The reviewer's alternative Nov. 6-17 qualification alone does not establish full story relevance or resolve the recovery condition. No forced replacement, automatic retry, or speculative prompt change was added. This remains a possible missed signal requiring editorial review, not a confirmed deterministic defect.

Live October summaries, raw review decisions, caches, and saved evidence were preserved. The existing issue therefore still displays no Next Signals. A targeted semantic review of the hearing is the next operational validation if publication is desired; no full pipeline rerun is needed. Images and summary issues remain outside this change. Historical freeze statements below describe the earlier version.

---

# Next Signal: promoted and frozen

## Final status (October 13, 2026)

User approved promotion, daily pipeline integration, and removal of confirmed obsolete testing files. `generate_next_signals.py` now contains the tested `next-signal-v19-countdown-scope-2026-10-12` implementation. Its executable content exactly matched the approved root draft before that duplicate was removed; only header comments changed.

`run_pipeline_by_date.py` now runs Next Signal once after successful summarization. It explicitly passes global receipts on the successful global path, or the same fresh expanded/final input chosen by the baseline summarizer. Both discovery failure and later global-stage failure use baseline evidence. A failed Next Signal stage returns failure instead of printing pipeline success. Existing upstream stages and feature code were not changed.

Verification: all 52 offline tests pass against active production code, including 48 feature regressions and four orchestration tests with multiple path/date/failure scenarios. Dry-run command plans were also checked without executing stages. No paid API calls or publisher requests, and no daily pipeline execution or live-output regeneration. The first real daily run remains operational validation; future-story accuracy is not guaranteed. The conservative countdown omissions and reviewer limitations documented below still apply.

Cleanup removed the duplicate root draft, rejected experiment and its bytecode, obsolete paid evaluation runner/fixtures, v17 inspection script/code/bytecode, stale test bytecode, and six redundant experimental summary copies. Empty experiment/fixture/bytecode directories were removed. The legacy test class for the rejected experiment was removed; the retained tests import the active implementation.

Retained intentionally: operational caches and dated live outputs; offline regression tests; `focused_review.json` and its source evidence; historical model reports, claim-evaluation responses, and experimental caches. These preserve paid responses and explain past failures without buying those calls again. They are historical records, not runnable alternate implementations. No further cleanup of those records is needed for operation.

The sections below are historical iteration notes. Earlier statements that the draft is unpromoted, blocked, or needs approval describe the state at that time and are superseded by this status.

---

# Next Signal v18 — applied date repair and review record

**Current cost authorization (September 29):** the user permits necessary API batches costing less than $0.01 without another permission request. Keep calls minimal and evaluate the cost of the whole batch. Ask before a batch estimated at $0.01 or more. This does not change the agreed review-before-code-change workflow or the frozen components.

## Applied: complete dates and shared sentence handling

After user approval, the active [generate_next_signals.py](C:/Users/jchst/Desktop/nominal-news/generate_next_signals.py) was updated from v17 with the narrow calendar repair described below. Its overall version is now `next-signal-v18-complete-calendar-dates-2026-09-26`.

- Article-excerpt extraction uses the existing shared sentence splitter.
- Abbreviated month dates retain their period, day, and year through sentence splitting and calendar parsing.
- The model, model prompts, and number of model stages remain as in v17. Model-stage cache versions were preserved; only the overall and prefilter version labels changed.
- The review file [generate_next_signals_v18.py](C:/Users/jchst/Desktop/nominal-news/generate_next_signals_v18.py) now holds the v19 draft that reviews a fixed proposal, described below. Its existing filename was retained to avoid creating another draft file. It is not the active implementation.

**Verification:** 18 offline calendar checks passed, including all supported month abbreviations, historical captions, year boundaries, and sentence boundaries. The 30-story cached-evidence replay reproduced 55 → 47 candidate passages and 20 → 18 stories with candidates. Eleven stories had changed evidence; 57 model-stage cache keys across the 19 unchanged stories remained identical. Syntax and function-level comparisons confirmed that only the two intended functions changed, alongside the calendar regex and version labels.

**Cost and files:** zero paid API calls, zero publisher fetches, and no new files or folders. Only the active generator and this review document were edited. Hashes confirmed that saved summaries, reports, caches, application UI files, and the older experimental draft were unchanged. The live mobile preview still displays the existing saved newsletter. No new model-generated results were produced.

This fixes the demonstrated date corruption. Participant/action errors and the other limitations documented below still need work; the complete Next Signal feature is not yet finalized.

## October 12: countdown scope refined with date-independent grammar

Applied the authorized refinement to the existing draft (`next-signal-v19-countdown-scope-2026-10-12`). No story names, event noun dictionaries, or date-specific branches were added. The change uses the duration's position, existing sentence splitting, semicolon boundaries, and grammatical prerequisite relationships.

- Checks now focus on the sentence/clause containing the selected duration. A historical “asked if…” in another sentence no longer disqualifies an otherwise unconditional countdown.
- Durations attached to an event through “after,” “following,” or “upon” are withheld rather than assumed to start on the article date. These connectors work across subjects; the implementation does not look for “approval” or other event names.
- Explicit prerequisites in adjacent sentences/clauses remain guarded, including pending conditions and dependency statements. Removing the condition from generated prose does not bypass the source check.
- The validation cache version changed again so earlier approvals cannot bypass the refinement. Model prompts, stages, evidence selection, and operational caches remain unchanged.

**Verification:** all **48 current-draft offline tests passed**. All nine saved boundary cases now behave as expected. Additional checks exercise four subjects across reference dates in 2026, 2027, and 2030 (84 variations), including attached prerequisites, unrelated historical questions, unconditional timing, and the tested completed-start wording. Explicit cross-sentence prerequisites remain blocked. The saved ten-story replay still removes only the bad Hormuz signal and retains the five supported signals. No fresh semantic judgments were obtained.

**Limits:** these are reusable grammatical rules, not a full language parser or a guarantee for every future article. Event-relative clocks remain conservatively withheld when their start cannot be resolved; the code does not infer completed prerequisites from arbitrary prose. Ambiguous nearby prerequisites can still cause omissions. Keep the existing semantic reviewer for action, actor, relevance, and event ordering.

**Next step:** review the final outputs and approve promotion/freezing of this scoped draft. No further paid testing or speculative model changes are proposed for the two repaired boundary cases. Keep cleanup separate. Results are under `countdown_scope_offline_review_2026_10_12` in the existing review JSON.

**Scope and cost:** zero API calls and zero publisher requests. Only the existing draft, tests, review JSON, and this document changed. No new files/folders, promotion, or live-output changes. Purifier, clusters, summaries, market cards, UI, and operational caches remain frozen.

## October 12: expanded countdown testing — original failure fixed, two boundary gaps remain

Retested the unchanged draft: all **45 existing tests pass**. Nine additional offline cases produced seven expected outcomes and two mismatches. Network and model calls were blocked; **zero API cost**.

- **Confirmed fix:** the actual Hormuz signal and a rewrite dropping its condition are blocked before review. Pending approval is also blocked. Explicit calendar conditions, ordinary forecasts, and the tested completed-start wording still reach review.
- **Missed prerequisite:** “The port will reopen within 7 days after the permit is granted” passes the local check although the clock has no established start. This tests only the local gate; it does not show that a model reviewer would approve it.
- **Overbroad rejection:** “Officials asked if repairs were complete. The port will reopen in 7 days” is rejected because the guard finds the unrelated historical “if.” This can suppress a valid candidate.

**Next recommendation:** refine this same local guard to handle duration-linked future triggers such as “after approval” and to distinguish the relevant prerequisite from unrelated wording. Preserve actual cross-sentence prerequisites and confirmed completed starts. Test those boundaries before freezing. No model upgrade, extra stage, or paid check is needed to diagnose these deterministic failures.

These results narrow the prior readiness assessment: the saved issue is repaired, but the guard is not yet ready to freeze more generally. Details are under `conditional_countdown_boundary_tests_2026_10_12` in the existing review JSON. Only the existing review document and JSON changed; code, tests, caches, and live outputs were not modified.

## October 12: conditional countdown guard applied; saved failure blocked offline

The existing draft now uses version `next-signal-v19-conditional-countdown-2026-10-12`. Before semantic review, it checks whether a proposed numeric duration such as “in 7 days” or “within 30 days” comes from a source that still states a prerequisite such as “if,” “once,” “pending,” or “subject to.” Such a countdown is withheld. Checking the source also blocks a paraphrase that removes or changes the condition. The bad Hormuz proposal therefore stops before a reviewer request.

This is a conservative local rule, not a general grammar or condition-completion resolver. A candidate that still contains conditional language can be omitted even if wider context might resolve it; clearer evidence is needed. Unconditional durations, explicit calendar/weekday checkpoints, and forecasts retain their existing handling. The guard does not establish that every other duration has a valid start or solve arbitrary paraphrases.

The validation cache version changed to prevent the old approval from bypassing this check. Writer versions, prompts, model, stages, evidence selection, and prefilter are unchanged. Operational caches and raw failure evidence were preserved.

**Verification:** all **45 current-draft offline tests passed**, with network access blocked. Checks cover the actual saved failure, retained/removed conditions, several prerequisite forms, fixed-date conditional events, ordinary durations, unrelated dates that cannot establish a clock start, and a pipeline test seeded with the old cached approval. An initial cache test needed its evidence fixture held fixed because the normal evidence selector changes the rows used in cache keys; the corrected test passed without a production-code change.

**Saved-issue replay:** the Hormuz signal is now omitted, while the five previously supported signals remain unchanged: October 5 filings, Monday court appearance, November referendum, the qualified storm turn, and the Sunday meeting. Four prior omissions remain, giving five signals and five omissions. Five saved reviewer responses were simulated to check local behavior; these are not fresh model approvals. Historical evidence dates were preserved. Details are in `conditional_countdown_offline_replay_2026_10_12` in the existing [focused_review.json](C:/Users/jchst/Desktop/nominal-news/outputs/next_signals_v18/focused_review.json).

**Next steps:** review these final results together, then approve promotion of the draft and freezing this feature. The known reviewer explanation/overlapping-window limitations remain documented; they do not justify another speculative prompt or model change in this iteration. Any cleanup or pipeline integration remains a separate agreed step. This sample does not prove accuracy on unseen issues.

**Cost and scope:** zero paid API calls, zero publisher requests, and no full pipeline run. Only the existing draft, tests, review JSON, and this document changed. Active code, purifier, clustering, summaries, cards, UI, live outputs, and operational caches remain unchanged. No files or folders were added or removed; the draft is still unpromoted.

## September 29: full saved-issue review — one confirmed blocker before freezing

Completed the review of all ten stories with the current `next-signal-v19-simple-writer-2026-09-29` draft. Cached-only evidence reconstruction found the same candidates. Source review supports **five signals and four omissions; one generated signal is incorrect**. No source code was changed.

| Story | Reviewed result |
| --- | --- |
| Hormuz proposal | **Reject.** The writer and reviewer approved a seven-day reopening signal despite the source establishing no start for that conditional clock. |
| Pope's AI remarks | Correct omission: no timed substantive follow-up in the available cached coverage. |
| White House access | Supported: “Both sides must file arguments by Oct. 5 regarding a preliminary injunction.” A filing deadline, not a promised ruling date. |
| New Jersey resignation | Correct omission: resignation completed; replacement arrangements lack a new explicit date. |
| Tasia Fortune case | Supported: “A suspect charged with murder will appear in court Monday.” Source context identifies the defendant; September 30 concerns a separate later hearing. |
| University security report | Correct omission: the timed candidate is a newsletter advertisement. |
| Missouri map | Supported: the November referendum decision. Exact current-version writer/reviewer decisions were reused without new calls. |
| Nor'easter | Supported: expected turn toward New Jersey Saturday night into Sunday. Strict ordering against other overlapping forecast windows remains unresolved. |
| Pope's France visit | Supported: Sunday meeting with seven French victims of clergy abuse. Routine travel alone is not a substantive checkpoint. |
| Butler victim's death | Correct omission: no timed future hearing, funeral, or other follow-up in the available cached evidence. |

### Why the Hormuz result fails

The cited sentence says Iran proposes reopening “in 7 days **if the US meets its conditions**.” The output says: “Iran will attempt to reopen the Strait of Hormuz in 7 days if U.S. conditions are met.” It treats an unstarted conditional duration as a check-back time, changes whose conditions must be met, and adds an attempted action. This was approved by the model reviewer despite the existing instruction requiring a confirmed countdown start.

Offline replay shows the local validator compares matching `relative:in 7 days` keys without establishing the trigger. It accepts the actual output, a paraphrase retaining the correct condition, and a rewrite removing the condition entirely. `_UNCERTAINTY_RE` covers forecast/modality words but does not enforce an `if` prerequisite. This is a confirmed local gap plus a failed semantic review, not a request for more publisher evidence.

**Next proposed fix:** reject a duration-based checkpoint when its supporting source makes the start depend on an unmet or unconfirmed prerequisite. Check the source even if generated wording drops the condition. Keep this distinct from explicit calendar checkpoints with conditions. Apply the guard before the reviewer call, and change the validation cache version so the saved approval cannot bypass it. **A conservative version of this fix was subsequently authorized and implemented on October 12, as recorded above.** It adds no model stage or upgrade.

### Cost, coverage, and limits

The batch allowed at most twelve new requests and **$0.009 total**, reusing exact current-version decisions where possible. It completed with **eleven requests**: six writers and five reviewers. Missouri reused two decisions; three stories had no candidates and made no calls. Usage was 22,085 input tokens and 1,231 output tokens, zero cached input; **$0.00405135** (about 0.405 cents), using the [official rates](https://developers.openai.com/api/docs/models/gpt-4o-mini). This cost covers only this Next Signal validation. No retries, research, publisher fetches, or upstream pipeline run occurred.

All ten cases finished without response-format errors. Source review, rather than a transport or parsing error, found the blocker. Raw replies and the known-bad Hormuz decision are retained as failure evidence under `simple_writer_saved_issue_final_validation` in [focused_review.json](C:/Users/jchst/Desktop/nominal-news/outputs/next_signals_v18/focused_review.json); that decision must not be imported as approved operational work.

Reviewer audit caveats remain: it conflates later/overlapping timing with nonqualification in the Tasia and storm cases, and assigns unsupported earlier order to the Sunday Mass in the Pope case. These explanations are flawed, but do not by themselves establish that those five supported displayed signals are wrong. This is one saved issue containing some September 26 articles, not an unseen-date or historical-availability evaluation.

**Recommendation: do not freeze or promote yet.** Address the conditional-clock blocker with the narrow local fix above. Only this document and the existing review JSON changed. All 351 protected files stayed unchanged during the paid run; prior result sections were preserved. Draft/active source code, tests, frozen features, live outputs, and operational caches are unchanged. No files or folders were added or removed.

## September 29: simpler writer applied — referendum now passes; storm review caveat remains

The approved simplification is in the existing draft, version `next-signal-v19-simple-writer-2026-09-29`. Both writers return their selected sentence/source and existing decision metadata without a source-by-source comparison report. They still receive all evidence and instructions to choose the earliest material checkpoint, preserve timing and uncertainty, and avoid starting conditional countdowns without a confirmed start.

The independent reviewer still checks the exact proposal against every source. Its prompt, response schema, quote/coverage checks, ordering checks, and version are unchanged. The local grounding checks and prefilter are also unchanged. Writer version labels changed; old results and caches remain available.

**Lower output limits:** local writer 260 tokens; optional external writer 220 tokens. Previously these allowed an additional 75 tokens per source. The reviewer retains 240 + 90 tokens per source. No model or stage was added, and there is no automatic regeneration loop.

### Verification and results

All **41 current-draft offline checks passed**, with networking blocked and model responses simulated. The saved Missouri sentence now reaches review without its incorrect writer report. Supplying those same incorrect source quotes as a reviewer response still fails validation. A reviewer that marks the selected event unqualified still prevents publication and caches the rejection. Failure preservation, fixed wording/source, timing checks, and external-path review also passed. Source comparisons confirmed the reviewer function, parser, and pipeline body were unchanged.

One bounded paid check used **three GPT-4o-mini requests**, capped at $0.005 for the entire batch: a new Missouri writer and review, then a review of the saved storm proposal. No storm generation was repeated.

| Case | Source assessment |
| --- | --- |
| Missouri referendum | **Passed:** “Voters will decide on the new congressional map in a November referendum.” The writer preserved the unresolved decision and original timing; the reviewer approved the exact sentence/source. Each comparison now quotes its assigned source. The blocked map revival is excluded, although the explanation could identify the halted action more precisely. The sentence does not promise approval or a change to the map used for the underway 2026 election. |
| Nor'easter | **Supported sentence approved:** “Nor'easter expected to make a sharp turn Saturday night into Sunday.” The self-ordering error is gone. However, the reviewer still disqualifies possible flooding because its timing overlaps and calls future overnight rain/wind merely ongoing. Those explanations are faulty. The supplied windows do not establish a strictly earlier alternative, so the faulty explanations alone do not prove the selected sentence is wrong. Exact earliest order remains unresolved. |

**Cost:** 6,284 input tokens, 382 output tokens, zero cached input; **$0.0011718** (about 0.117 cents), using the [official rates](https://developers.openai.com/api/docs/models/gpt-4o-mini). All replies completed. No retries, research, publisher requests, upstream pipeline run, or publication. The writer's unsolicited search queries under `SIGNAL` were discarded by the existing handler. Raw replies, evidence, decisions, usage, and assessment are under `simple_writer_referendum_storm_review` in [focused_review.json](C:/Users/jchst/Desktop/nominal-news/outputs/next_signals_v18/focused_review.json).

**Next step:** retain the simpler writer and review the remaining saved stories against the feature goal before freezing, starting with cached evidence and completed decisions. This two-case check does not establish full-issue or unseen-date accuracy. No additional code change or paid batch was performed.

**Preservation:** only the existing draft, tests, review JSON, and this document changed. No files or folders were added or removed. All 351 protected files stayed unchanged during the paid run, and previous result sections were preserved. Active code, frozen components, live outputs, and operational caches remain unchanged. The draft is still unpromoted.

## September 29: source wording and alternative-only ordering — applied to draft

The approved follow-up is in the existing review draft, version `next-signal-v19-source-wording-2026-09-29`. The active generator and frozen components remain unchanged.

- Both writer prompts now ask for a complete future-action sentence, the source's original timing, and an unresolved vote/decision rather than a promised outcome. Inferred calendar dates remain blocked locally.
- The review response schema omits `before_selected` for the selected source. Other sources still require a boolean or unknown ordering judgment. Complete source coverage, selected-event qualification, and the guard against acknowledged earlier events remain required. Old self-ordering metadata is discarded during parsing.
- The reviewer is explicitly told that an overlapping window does not make a relevant event unqualified. This is an instruction, not a deterministic guarantee of correct reasoning.
- Same model, stages, output caps, evidence selection, and prefilter version. Writer/reviewer version labels changed to separate new decisions from old ones; caches were preserved.

**Offline verification: 40 current-draft checks passed**, with model responses simulated and networking blocked. Saved invented-date and incomplete-phrase failures still stop before review. Replaying the saved storm reply no longer fails on self-ordering; its old judgments are not corrected by replay. Tests retain failures for missing/invalid alternative ordering, incomplete evidence coverage, unsupported quoted timing, and an unqualified selection. Fixed wording/source, caching, and output preservation checks also passed.

### Focused paid check and source review

The planned batch covered the Tasia and Missouri writers/reviewers plus one storm review reusing its saved proposal. It was capped at five requests and $0.009 for the whole batch. It stopped after **three requests**, with no retry, research, publisher fetch, upstream pipeline run, or publication.

| Case | Result |
| --- | --- |
| Tasia appearance | **Supported signal:** “The suspect will appear in court Monday.” Writer and reviewer kept Monday without inventing September 28. The cited context identifies the 45-year-old defendant; the output is less specific. The new schema omitted self-ordering successfully. The reviewer still used later timing to label the separate September 30 hearing unqualified, so its qualification reasoning remains imperfect. |
| Missouri referendum | **Wording improved, processing still failed:** “Voters will decide on the new congressional map in a November referendum.” This passes the unchanged local grounding check and leaves the outcome unresolved. However, the writer copied the referendum event and timing from source 3 into comparisons for sources 1 and 2. Their candidate sentences do not contain that timing. The quote check correctly stopped processing before review. This is not a validated final signal. |
| Storm | Not reached in the paid batch. The old reply was replayed offline to check parser behavior; new storm selection/relevance reasoning remains untested. |

**Cost:** 5,959 input tokens, 396 output tokens, zero cached input; **$0.00113145** (about 0.113 cents), using the [official rates](https://developers.openai.com/api/docs/models/gpt-4o-mini). Every API reply finished normally; the stop was a local source-comparison error. The current batch is complete. Raw replies, decisions, evidence, usage, and offline diagnosis are saved under `source_wording_alternative_order_review` in [focused_review.json](C:/Users/jchst/Desktop/nominal-news/outputs/next_signals_v18/focused_review.json).

### Next suggestion at that point

Simplify the writer's response to the proposed sentence and supporting source. Keep its instructions to inspect all candidates, and keep the independent reviewer's full-source comparison and earliest-event checks. The current design asks both stages to produce a source-by-source report; a bad entry in the writer's duplicate report prevented an otherwise locally supported sentence from reaching review. Removing that duplicate report could reduce output tokens and this failure mode without adding calls. It would not itself establish better semantic accuracy. **This simplification was subsequently authorized and implemented, as recorded above.**

**Preservation:** only the existing draft, tests, review JSON, and this document changed. No files or folders were added or removed. All 351 protected files stayed unchanged during the paid run, and earlier result sections were preserved. Active code, frozen components, live outputs, and operational caches remain unchanged. The draft remains unpromoted and the feature is not ready to freeze.

## September 29: deadline and same-day fixes applied; validation stopped on remaining output problems

The two local fixes are implemented in the existing review draft. Its current version is `next-signal-v19-deadlines-same-day-2026-09-29`; the active generator remains unchanged.

### What changed and what passed offline

- Explicit dated obligations such as “gave both sides until October 5 to file” and “must file by October 5” can now reach review. The same scheduling recognition is used when checking proposed wording. Standing durations without a confirmed start do not gain a calendar date from this exception.
- Same-day weekday references with a time of day can be retained as candidate windows when the action has prospective language. Present qualified forecasts such as “is possible” are recognized. The original window and uncertainty stay intact; a plain same-day weekday still does not roll into the next week. Historical, cancelled, and stale same-day schedules remain excluded in the tested cases.
- Calendar comparisons can display the source day for these windows but do not claim an exact hour or strict order between overlapping windows. Window keys distinguish morning, afternoon/evening, and night, preventing tested rewrites from changing the period.
- The model, number of stages, strict response schema, and output limits are unchanged. Prefilter and stage versions changed because the evidence selection, local checks, and timing comparison changed. Existing saved decisions and caches were preserved.

**All 38 current-draft checks passed offline.** Rebuilding evidence for all ten stories used cached pages with network/model access blocked. The White House evidence gained the October 5 deadline. The storm evidence gained Saturday-evening possible flooding, replacing the weaker Monday easing passage within the existing per-source limit; its expected turn remained available. Candidate sentences for the other eight stories were unchanged, including the referendum, Monday appearance, and Sunday meeting. No evidence budget was increased.

### Bounded model validation

Under the user's standing permission for batches below one cent, one run was capped at **14 GPT-4o-mini requests and $0.009 total**. Before every request, the runner checked that measured prior usage plus a conservative reservation for the next response stayed within that cap. It used cached evidence and saved decisions only to the existing review JSON, with retries, research, publisher requests, and publication disabled.

The run stopped after **eight requests** on a reviewer response-contract error. Seven stories completed processing, the storm stopped during review, and the final two stories were not reached. This is **not a completed full-issue validation**.

| Case | Actual result and assessment |
| --- | --- |
| White House deadline | **Passed:** “Both sides must file arguments by Oct. 5” was generated and approved unchanged. It is a filing deadline, not a promised ruling date. |
| Tasia appearance | Writer selected the supported Monday appearance but added “September 28,” absent from the cited sentence. That inferred calendar date triggered the local timing check. The previously supported signal was lost in this run. |
| Missouri referendum | Writer now selected the referendum, but wrote “Voter approval for new map in November referendum.” It lacks future-action wording and may imply approval is assured. Local validation rejected it. Its comparisons also incorrectly qualified revival of the blocked map. |
| Storm | Recovered Saturday-evening flooding reached both stages. The writer chose the expected turn. Reviewer returned approval with `before_selected: null` for the selected source, where the contract requires `false`; parsing stopped. It also treated overlapping timing as a reason to disqualify the flooding event, which confuses relevance with ordering. |
| Hormuz | Omitted after the generated phrase lacked future-action wording and omitted conditions. Appropriate omission, but not proof the model correctly understands the unstarted countdown. |
| Pope AI remarks / New Jersey resignation | No candidates and no model calls. |
| University security report | Correctly excluded the unrelated newsletter with one call. |
| Pope visit / Butler death | Not reached in this paid batch. Prior supported meeting and absent Butler candidates remain historical/offline evidence, not fresh decisions. |

**Cost:** 17,094 input tokens, 1,303 output tokens, zero cached input; **$0.0033459** (about one-third of a cent), using the [official rates](https://developers.openai.com/api/docs/models/gpt-4o-mini). All responses finished normally at the API; the stop was local validation of the reviewer's content. There were no retries or subsequent paid calls. Raw replies, recovered evidence, decisions, usage, and assessment are under `deadline_same_day_issue_validation` in [focused_review.json](C:/Users/jchst/Desktop/nominal-news/outputs/next_signals_v18/focused_review.json). Completed decisions remain stored for appropriate reuse.

### Next proposal for discussion

Keep the two local fixes in the unpromoted draft. Align the writer's output with the checks: a complete future action, the original timing phrase, no inferred calendar dates, and no claim that a pending vote's outcome is settled. Remove the unnecessary model judgment about whether the selected event precedes itself; retain independent comparisons with alternative events. Do not add a model stage or automatic rewrite/retry loop. **This follow-up was subsequently authorized and implemented; results are recorded above.**

**Preservation:** only the existing draft, tests, review JSON, and this document changed in this iteration. No files or folders were added or removed. All 351 protected files stayed unchanged during the paid run; earlier result sections were preserved. Active code, frozen components, live outputs, and on-disk operational caches remain unchanged. The feature is not ready to freeze.

## September 29: offline review of all 10 saved stories — two timing fixes proposed

Reviewed the saved September 25 issue, its model evidence and decisions, the latest storm result, and relevant passages in the existing article-text cache. Checked local behavior with model calls and networking blocked. **Zero paid calls, publisher requests, or code changes.** Only this review document was updated; saved results and all other project files remain unchanged.

This is a source review of existing results, not a fresh run of the current draft across ten stories. The full-issue decisions predate the latest writer/reviewer contract; only the storm has a paid result for the current revision. The saved issue also contains September 26 articles. Missing or failed cached pages were not fetched.

### Story review

| Saved story | Assessment from existing evidence |
| --- | --- |
| Hormuz proposal | Omission is appropriate. The seven-day period depends on acceptance and other conditions; no confirmed start is established. A future-tense paraphrase can pass the local gate, so the reviewer must still enforce this rule. |
| Pope's AI remarks | Omission is appropriate. The cached passages describe completed remarks and unrelated dates; no timed substantive follow-up to those remarks was found. |
| White House press access | **Missed checkpoint:** both sides have until October 5 to file arguments on a longer-lasting injunction. This is a filing deadline, not a promised date for the judge's decision. |
| New Jersey resignation | Omission is appropriate. The resignation already occurred. Replacement and acting-official arrangements lack a new explicit future date. |
| Tasia Fortune case | The Monday court appearance is supported. Cached context identifies the 45-year-old defendant; the September 30 hearing concerns Ratliff and is a separate later step. |
| University security report | Omission is appropriate. The timed model passage advertises a newsletter; the cached report coverage has no dated next action addressing its findings. |
| Missouri congressional map | **Missed checkpoint in the earlier run:** the November referendum concerns approval of the disputed map. The writer invented a court action, which the reviewer correctly rejected, but the separate referendum was left unused. |
| Nor'easter | The latest expected Saturday-night/Sunday turn is supported. **Earliest selection remains unproven:** cached text also contains forecasts for later Saturday that were removed before the model saw them. |
| Pope's France visit | The Sunday meeting with seven abuse survivors is supported and substantive within the ongoing visit. Earlier routine travel alone does not establish a better checkpoint; the Sunday Mass has no precise order relative to the meeting in the supplied passages. |
| Butler rally victim's death | Omission is appropriate. The saved coverage mentions an existing lawsuit but supplies no timed hearing, funeral, or other future development. |

### Two confirmed local problems

1. **Explicit deadlines are not recognized consistently.** Three cached White House articles contain the October 5 filing deadline. The saved enrichment audit already includes it, and the current calendar parser correctly produces `date:2026-10-05`. However, `_anchor_is_action_timing` rejects “The judge gave both sides until Oct. 5 to file their arguments…” because it lacks one of the recognized future phrases. The sentence is 159 characters, so the short-headline fallback does not apply. It never reaches the model. An illustrative, source-supported sentence using “must file … by October 5” also fails the final future-language gate. This is a scheduling-language problem, not another abbreviated-date corruption.
2. **Later same-day windows are lost.** The cached ABC storm article, published September 26 at 15:50 UTC, forecasts possible flooding during high tide “later Saturday evening,” and continued/spreading rain “On Saturday afternoon and evening.” Using its actual Saturday reference date, `_future_anchor_records` returns no anchors for either sentence. Plain same-day weekdays are deliberately excluded, but the existing relative-time rules do not capture these explicit later-day formulations. Both passages are absent from the four model candidates. The possible-flooding wording also needs to retain its qualification. Do not solve this by rolling every same-day weekday into the next week or asserting exact order between overlapping windows.

**Follow-up:** these two local scheduling changes were subsequently authorized and implemented in the draft, as recorded above. The purifier, clusters, summaries, cards, and UI remain frozen.

### Selection checks before freezing

The Missouri referendum was already source 3 in the earlier model evidence. The cached AOL article also confirms an active campaign to reject the map in that November referendum. A concise illustrative referendum sentence passes the current local grounding check; this is not a new model result. **Correction to the earlier issue assessment:** rejecting the invented court decision was correct, but treating the entire story as having no useful future checkpoint overlooked the referendum. Do not imply that the referendum changes which map is used for the already-underway 2026 congressional election.

Test the current writer's required source assessments on this case after the local timing fixes. The latest writer has not been tested here, so another prompt change or model upgrade is not yet justified. Preserve the existing two stages and fixed-proposal review; an automatic regeneration loop is not proposed.

The storm's Monday comparison and the Hormuz countdown remain useful negative checks: retaining a word such as “expected” does not prove that a date belongs to the right action, and a conditional duration is not a running clock. Offline probes confirm those semantic mistakes can pass the local gate. The existing reviewer must reject them. Incorrect refund-exclusion wording alone does not justify another model stage.

**Next recommendation:** implement the two local timing fixes, inspect their offline results, then perform one bounded final validation that includes the referendum, these recovered candidates, and the previously supported signals. This review does not authorize code changes or start that paid validation. No new files or folders were created.

## September 29: approved two-call check — supported signal, comparison caveats remain

The writer selected source 3 and the existing reviewer approved its exact sentence:

> Nor'easter expected to turn back toward New Jersey coast Saturday night into Sunday.

This is supported by the source's forecast of a sharp turn toward the New Jersey coast. It keeps the uncertainty and timing, and the path change could materially update the storm damage/disruption story. Both replies completed with all four source entries. The reviewer preserved the sentence and source index.

**Remaining limits:** the writer still incorrectly says the refund passage contains no future event, and pairs Monday with the start of easing rather than lingering impacts. The reviewer calls the rain/wind alternatives ongoing rather than future, despite their future language; its exclusions need more precise explanations. The selected turn and continued rain/wind have overlapping windows, so strict order between them is not established. The supported final signal is an improvement on the prior result, not proof that all comparisons are correct.

This paid proposal retained “expected,” so it did not directly exercise the rain/wind uncertainty exception; that fix remains checked offline. This also reuses the saved September 25 issue with September 26 passages, rather than testing an unseen date or historical evidence availability.

**Cost and scope:** exactly **two GPT-4o-mini requests**, 4,515 input tokens and 380 output tokens, zero cached input. Calculated cost **$0.00090525** (about 0.09 cents), below the $0.01 cap, using the [official rates](https://developers.openai.com/api/docs/models/gpt-4o-mini). No retries, research, publisher requests, pipeline regeneration, or publication. All 351 protected files stayed unchanged during the run. Only the existing results JSON and this document changed afterward; prior result sections were preserved. No code changed and no files or folders were added.

**Recommendation:** retain this supported result for review and agree on bounded final validation before freezing. Keep the draft unpromoted; this single case does not establish general accuracy. No additional code changes or paid calls were made. The approved batch is complete.

Raw responses, comparisons, source evidence, usage, and assessment are saved under `simpler_writer_uncertainty_storm_review` in [focused_review.json](C:/Users/jchst/Desktop/nominal-news/outputs/next_signals_v18/focused_review.json).

## September 29: simpler writer and narrower uncertainty check — applied to draft

The approved changes are in the existing `generate_next_signals_v18.py` review draft. The active generator remains unchanged.

- **Writer:** retains each source's event, quoted timing, qualification, and explanation. It no longer returns `before_selected`. The existing independent reviewer supplies ordering judgments against the fixed proposal. Complete source coverage and qualification checks remain required; selecting a source marked unqualified still stops before review. Stale writer ordering flags are discarded.
- **Uncertainty:** the local check now permits an unqualified matching clause when uncertainty appears in a separate clause with no shared content words or backward references. This allows the saved rain/wind sentence past the flooding qualification. Direct uncertainty, shared wording, and references such as “it may be delayed” retain the conservative block. The complete source still goes to the reviewer.
- **Cost controls:** same model and stages; no retry loop. Removing the writer's ordering field reduces its output allowance from 90 to 75 tokens per source. With four sources, local generation allows 560 tokens, review 600, and optional external generation 520. These are ceilings, not measured savings. New prompt versions preserve old decisions while separating them from this revision.

**Verification:** all **32 current-draft tests passed**, with model responses simulated and networking blocked. The actual saved storm reply now parses as a provisional proposal and reaches the existing reviewer. A simulated rejection still suppresses it. Its incorrect refund explanation remains visible; replay does not correct the model's judgments. Tests also cover direct and ambiguous uncertainty, unchanged proposal wording/source, reviewer ordering conflicts, caching, and output preservation on failures.

**Limits:** this is a narrow lexical exception, not a general parser of uncertainty. Paraphrases can still be blocked; semantic scope, relevance, event/time attachment, and overlapping forecast windows still require review. The subsequently approved paid check is recorded above. Keep the draft unpromoted and the feature unfrozen.

**Scope:** zero paid API calls or publisher requests. Only the existing draft, test file, and this document changed. No files or folders were added or removed. Frozen components, the active generator, newsletter outputs, saved paid results, and caches remain unchanged. Another paid batch requires separate approval.

## September 29: approved required-comparison check stopped after one call

The writer changed its proposed signal to **“Heavy rain and strong wind gusts will continue overnight into Sunday morning.”** It cited source 1 and returned comparisons for all four sources. However, those comparisons were internally inconsistent, so the local validator stopped processing before the reviewer. This is not a successful accuracy result.

### What the reply showed

- The selected source had `before_selected: null`; the contract requires `false` for the selected event itself. This triggered the stop.
- Monday's event had `before_selected: true`, while its explanation explicitly called it later than the selection.
- The expected Saturday-night/Sunday turn was marked both qualifying and earlier, yet the writer chose the other forecast. Bypassing the first error would therefore still encounter the acknowledged-earlier-event guard. The two forecast windows overlap, so the claimed strict order also needs scrutiny.
- The refund passage was described as containing no explicit future event, although it says tickets will be refunded within 30 days. Excluding it may be appropriate, but that explanation is inaccurate.
- The response attached “into Monday” to the start of easing, while the source attaches it to potential lingering impacts. It also supplied search queries with `SIGNAL` status; the code would discard them, and no research was performed.

The new schema was accepted by the API, the response finished normally, and all comparison entries were present. The problem is their content. A free replay reproduced the precise validation failure: `Next Signal's selected event cannot precede itself.`

### Additional offline finding

Passing the raw proposed sentence through the existing grounding check also rejected it for removing uncertainty. The cited source qualifies **isolated flash flooding** as possible in another clause; the proposed sentence describes continued rain and wind. The existing whole-sentence uncertainty rule treats that qualification as applying to the entire sentence. This confirms a previously recorded open issue on the newly generated proposal. Removing the ordering-field error alone would not get this proposal through the local checks.

**Follow-up:** the writer simplification and narrow uncertainty correction were subsequently approved and implemented offline, as recorded above. They keep the same model and existing stages. They do not, by themselves, establish correct relevance judgments or event order.

**Cost and scope:** one request to `gpt-4o-mini-2024-07-18`; 2,402 input tokens and 225 output tokens, with zero cached input. Calculated cost **$0.0004953** (about 0.05 cents), below the approved $0.01 cap, using the [official rates](https://developers.openai.com/api/docs/models/gpt-4o-mini). The second call was skipped. No retry, additional model request, publisher fetch, research, full-pipeline run, or publication occurred. All 351 protected files stayed unchanged through the run, and previous review sections were preserved. Only the existing results JSON and this document changed afterward. Frozen components, active code, the draft, tests, live outputs, and caches were untouched.

Raw reply, comparisons, usage, preservation checks, and assessment are under `required_comparison_storm_review` in the existing [results file](C:/Users/jchst/Desktop/nominal-news/outputs/next_signals_v18/focused_review.json). The authorized run is complete; its unused second request is not an automatic retry authorization. Keep the draft unpromoted and the feature unfrozen.

## September 29: required source comparisons implemented in the review draft

After approval, the existing draft now requires a short comparison for **every supplied source** within each existing model response. Each entry reports the event, a timing phrase quoted from the candidate sentence, whether the event qualifies, whether it precedes the selected checkpoint, and a brief explanation. This replaces the optional request to explain skipped earlier events with a response the code can inspect.

### Behavior

- A response missing a source, repeating an index, using invalid decision types, omitting an explanation, or quoting timing absent from its sentence raises an error. Truncated responses and refusals also stop processing. Failed model stages are not cached as editorial `NO_SIGNAL` decisions; completed cached work is preserved and newsletter/report files are left intact.
- If the returned comparisons acknowledge a qualifying earlier event, the code blocks a later selection or approval. A first-pass contradiction stops before the paid reviewer. A selected source marked unqualified also blocks a positive decision. These are checks of the model's own stated judgments, not independent proof of relevance or chronology.
- An earlier excluded event needs its own explanation. Equal/overlapping windows, duplicate checkpoints, uncertain ordering, and unrelated earlier dates do not automatically displace a useful signal. Timing still needs to belong to the event being described.
- The reviewer produces its own comparisons. It receives the original proposal and source evidence, without the writer's rationale or comparisons. It still cannot replace the sentence or source index. Comparison entries are preserved in the existing decision/cache/report structures.
- All three model entry points use a strict JSON response schema: local generation, the existing reviewer, and optional external-evidence generation. [Official OpenAI documentation](https://developers.openai.com/api/docs/guides/structured-outputs) supports this format on GPT-4o-mini, including required fields and array limits, while warning that structured replies can still contain factual mistakes. No SDK upgrade or new dependency was needed.

### Cost controls and verification

The model and stage count are unchanged. The normal path still has one writer and one reviewer; the optional research path remains separate. No new repair or regeneration loop was added. Response limits now include **90 tokens per source** for the required comparisons, with at most eight sources:

| Existing stage | Previous output limit | With four sources | With eight sources |
| --- | ---: | ---: | ---: |
| Local generation | 260 | 620 | 980 |
| Review | 240 | 600 | 960 |
| Optional external generation | 220 | 580 | 940 |

These are ceilings, not measured usage. Replies and schema-bearing inputs will be longer. Actual usage and adequacy of the new limits need an approved run; no cost or accuracy improvement is claimed. Changed prompt versions keep old decisions intact but prevent reusing them for the new response contract.

**Verification:** all 29 current-draft tests passed with network access blocked. Checks cover complete comparisons, timing quotes, invalid indices/types, contradictory choices, irrelevant/duplicate/uncertain alternatives, fixed wording/source preservation, cache reuse, and output preservation on failures in generation, review, and the optional external path. Replaying the two saved storm replies and six older verdict-only replies now raises a missing-comparison error; this is a format check, not a corrected model judgment. A separate offline check ran the installed OpenAI 0.28.1 client's serialization for all three entry points against an intercepted transport and confirmed that the schema and output limits reached the request body. It made no network requests and did not test server acceptance or model output.

**Scope:** zero paid calls or publisher requests. Only `generate_next_signals_v18.py`, `tests/test_next_signals.py`, and this document changed. No files or folders were added or removed. The active generator, live newsletter, paid results, and on-disk caches remain unchanged. Earlier length rejection, clause-scoped uncertainty, conditional-clock semantics, and daily-runner integration remain outside this change.

**Status:** ready for review as an unpromoted draft. This makes omissions and contradictions observable; it does not yet show that the model will correctly select the earlier storm forecast. Any new paid check needs separate approval.

### Preparation for the completed required-comparison check

**Status:** approved and completed. The run stopped after one request; the result is recorded above.

The next check is limited to the same saved storm story: **up to two GPT-4o-mini requests, with a $0.01 total cap**. Generate once, then review the exact new proposal only if it passes the local checks. No retries, publisher fetches, external research, full-pipeline regeneration, or publication. The purifier, clusters, summaries, market cards, Worth Knowing, UI, active generator, and live outputs remain frozen.

The OpenAI payload would include the saved headline, summary, four source excerpts and metadata, timing hints, and required response schema. Review would also receive the new signal/source index, without the writer's explanations or comparisons. Success requires complete comparisons and a supported earlier material weekend checkpoint that the independent reviewer approves. Dropping the signal or blocking a contradiction would demonstrate a guard working, not successful event selection.

Offline preparation confirmed the real request shapes: generation allows 620 output tokens and review 600 for these four sources. Conservative byte-based reservations, including the schema and overhead, total about **$0.00449**, within the proposed $0.01 limit. Actual usage is unknown until a run. The placeholder replies used for preparation are not model results. **Zero paid calls have been made for this revision.** If approved, raw replies, usage, and assessment will be added to the existing results JSON under `required_comparison_storm_review`; earlier results and live caches will be preserved.

## September 29: approved two-call storm check failed — keep the draft unpromoted

The updated writer again returned **“Winds and rain will begin to lighten with potential lingering impacts into Monday.”** It cited source 2, exactly as in the previous issue review. The reviewer returned `{"eligible":true}`. The timing comparison was present, including source 3's Sunday reference before source 2's Monday reference, but this did not change the result.

Source 3 still supplies the earlier material forecast: the storm is expected to turn toward New Jersey Saturday night into Sunday. The writer explained why the later forecast mattered, without explaining why earlier candidates did not qualify. The review supplied no explanation, so its reply does not show that it compared the earlier checkpoint. Also, source 2 attaches Monday to potential lingering impacts, rather than explicitly dating the start of easing. These remain selection and event/timing-association problems; calendar ordering alone is insufficient.

**Result:** the targeted accuracy check failed. Do not promote this draft or freeze the feature. The offline checks correctly established code behavior, but did not establish improved model selection.

**Cost and scope:** exactly two requests to `gpt-4o-mini-2024-07-18`, both completed normally. Usage was 3,719 input tokens and 54 output tokens, with no cached input. Calculated cost was **$0.00059025** (about 0.059 cents), below the approved $0.01 limit, using the [official rates](https://developers.openai.com/api/docs/models/gpt-4o-mini). No retries, additional calls, publisher fetches, research, full-pipeline regeneration, or publication occurred. All 351 protected files were unchanged through the run, and all previous review sections were preserved. Only the existing results file and this document were updated afterward; no code changed and no files or folders were created.

Raw replies, usage, exact evidence, prompt hashes, reusable decisions, and assessment are saved under `storm_timing_comparison_review` in the existing [results file](C:/Users/jchst/Desktop/nominal-news/outputs/next_signals_v18/focused_review.json). The paid authorization is complete; further paid testing needs separate approval.

**Next proposal, not implemented:** make the comparison observable within the existing calls. Have the writer return a compact event/timing/qualification entry for each source before choosing its signal, and require the existing review result to explicitly account for earlier candidates. This would let the code detect a skipped comparison and make incorrect judgments easier to diagnose. It would still require semantic judgment and would not guarantee accuracy. Review the response-contract change and token budget before implementation; keep the same model and two-stage design, without automatic regeneration or another model pass.

## September 29: timing comparison added to the review draft — offline implementation

The saved storm result exposed a selection failure: both model stages already instructed the model to choose the earliest relevant checkpoint, and the earlier forecast reached both stages. The code supplied passages ranked by heuristic evidence score/source order, with weekday keys but no compact calendar comparison. The exact cause of the model's choice cannot be established from its reply.

The existing [draft](C:/Users/jchst/Desktop/nominal-news/generate_next_signals_v18.py) now provides a separate, locally calculated timing comparison to the first writer and reviewer, including the optional external-evidence writer. It preserves the original evidence and source indices. The prompts ask the model to compare all event/time pairs before writing, recognize material changes before a process ends, and briefly explain why an earlier candidate does not qualify when choosing a later event.

- In the storm evidence, source 3's Sunday reference maps to September 27, before source 2's Monday reference on September 28. The full Saturday-night/Sunday window remains in the original passage; the comparison does not assign an exact event start.
- Full and abbreviated dates compare equally. Weekday hints require future language, the existing action-timing screen, and a compatible publication/reference date. Completed or stale weekday passages, conditional countdowns, broad months, and relative phrases remain unresolved. Same-day weekdays are not rolled into the following week.
- Calendar matches are explicitly unverified. They do not establish which action a date belongs to, future tense, relevance, or the order of overlapping windows. All passages remain available, including those whose timing is unresolved. An earlier irrelevant date cannot automatically veto a supported later signal.
- The model, number of stages, response format, and output limits are unchanged. The comparison adds a short input section. Prompt cache versions changed, and the cache key now includes the reference date used by the comparison. Existing decisions are preserved; changed prompts will need fresh decisions during a future approved run.

**Verification:** all 19 checks in the existing `FixedProposalReviewChecks` class passed with simulated model responses and network access blocked. New checks cover the actual saved storm passages reaching all three existing model entry points, unchanged source indices/wording/call counts/output limits, equal dates, different reference dates, unresolved conditional windows, completed/stale weekdays, relevance-dependent review, and cache specificity. An initial offline check exposed an invented future date for the completed Australian disclosure; the weekday restrictions address that input case. The existing cache, failure-recovery, and saved-reply parser tests also passed.

**Limit:** these checks verify the comparison and code behavior. They do not show that the model now chooses the earlier storm event. No replacement signal was generated or published, and the feature is not ready to freeze on this evidence alone. Assess actual selection during the next approved Next Signal run; no additional model stage or upgrade is proposed. Clause-scoped uncertainty, early length rejection, conditional-clock semantics, and daily-runner integration remain separate open items.

**Scope and cost:** zero paid API calls, zero publisher requests, and no new files or folders. Only the existing draft, test file, and this review document changed. The active generator, saved newsletter, paid results, and caches remain unchanged. No cleanup or promotion was performed.

### Preparation for the completed two-call check

**Status:** subsequently approved and completed; the failed selection result is recorded above.

Test only the saved September 25 storm story through the draft's existing first writer and fixed-proposal reviewer. This directly checks whether the timing comparison changes the known wrong choice. Maximum **two GPT-4o-mini requests**, with a **$0.01 total limit**; skip review if generation supplies no usable proposal. No retries, external research, publisher fetches, full-pipeline regeneration, or publication.

The payload sent to the OpenAI API would contain the saved headline, summary, four source excerpts and metadata, calculated timing hints, and the new proposal for its review. Results would go into the existing `focused_review.json` under `storm_timing_comparison_review`. Existing paid results and live caches would be preserved.

Success requires a fresh, source-supported earlier material forecast, retaining its uncertainty and timing, followed by approval of that exact proposal. Source 3's Saturday-night/Sunday turn toward New Jersey is the expected useful checkpoint. Omission, local rejection, or another Monday selection would not establish that selection improved. A passing result would address this known case, not establish general accuracy or complete feature freeze.

An offline dry run captured both request shapes with networking blocked: 9,160 input UTF-8 bytes/260 maximum output tokens for generation and 8,321 input bytes/240 output tokens for review of a placeholder proposal. Byte-based conservative reservations, including overhead, total approximately $0.00323 at the [official rates](https://developers.openai.com/api/docs/models/gpt-4o-mini); the real review input depends on the generated proposal. Runtime limits must still enforce two requests and $0.01. **Zero actual API calls were made.** Draft hash: `4dfea57baa5f929afcf963f6bef0d487ba5464911f03b6fe65a58e749022cbff`.

## September 29: complete September 25 issue reviewed — earliest-event selection still needs work

The approved run exercised the draft's actual `main()` across all 10 saved September 25 stories, including fresh first-pass generation. It used cached article enrichment, disabled new external research, and captured newsletter/report writes in memory. The active generator and live newsletter were not changed.

### Results

| Story | Draft result | Source review |
| --- | --- | --- |
| Hormuz proposal | Removed the old seven-day signal | Appropriate omission: no supported start to the conditional plan. The local language screen blocked the regenerated proposal before another call. |
| Pope's remarks about AI | No signal; zero calls | No candidate in the supplied evidence. |
| White House access order | No signal | Candidates concerned completed hearings and newsletter text. |
| New Jersey resignation | No signal; zero calls | No candidate in the supplied evidence. |
| Tasia Fortune case | “A 45-year-old man charged with murder will appear in court Monday.” | Supported. The first pass regenerated the exact approved proposal, allowing its review to be reused. |
| University security report | No signal | The only timed passage advertised a recurring newsletter. |
| Missouri ruling | Removed the old future-decision signal | The reviewer rejected a fresh, incorrect claim that the Supreme Court would determine control of Congress in the November elections. |
| Nor'easter | “Winds and rain will begin to lighten with potential lingering impacts into Monday.” | Needs correction: an earlier material storm change was available in the same evidence. |
| Pope's France visit | “Pope Leo XIV will meet seven French victims of abuse by clergy on Sunday.” | Supported substantive activity during the ongoing visit. Original first-pass wording was retained. |
| Butler rally victim's death | No signal; zero calls | No candidate in the supplied evidence. |

### Remaining selection problem

The storm evidence forecasts a sharp turn toward New Jersey **Saturday night into Sunday**, before the Monday impacts. That change could materially update the damage/disruption story. The first pass selected the later forecast, and the reviewer approved it despite seeing the earlier source.

This is not a length or local-gate limitation: the 13-word sentence “The storm is expected to turn toward New Jersey Saturday night into Sunday.” passes the draft's existing structural checks against source 3. That was checked offline, with no model call or replacement output. It demonstrates an available candidate; it does not supply a new model judgment.

**Recommendation:** do not freeze or automatically promote this draft. Next, investigate and fix selection of the earliest relevant event offline within the existing stages. Adding a model pass or upgrading the model is not justified by this run. The first pass still makes factual mistakes, so the existing review remains necessary.

### Cost, caching, and limits

- **10 actual requests:** seven triage calls and three reviews; one prior review reused and one proposal blocked locally. Three stories made no model calls. No retries or publisher requests; no case requested new research.
- Returned model: `gpt-4o-mini-2024-07-18`. Usage: 14,842 input tokens, 365 output tokens, zero cached input tokens. Calculated cost **$0.0024453** (about 0.245 cents), below the approved $0.02 cap, using [published rates](https://developers.openai.com/api/docs/models/gpt-4o-mini). This is a token-based calculation, not a billing invoice.
- A second, entirely offline replay used the saved decisions and reproduced all 10 results with **zero model requests**. All summary fields other than Next Signal remained identical.
- Cached evidence includes articles dated September 26 in the September 25 issue. This verifies the saved-data flow, not evidence availability as of September 25 or performance on a new date. The external-research branch was disabled. Clause-scoped uncertainty, the early length limit, and daily-runner integration remain open.

The existing [results file](C:/Users/jchst/Desktop/nominal-news/outputs/next_signals_v18/focused_review.json) stores raw replies, all source candidates, complete audit rows, before/after signal text, reusable decisions, and this assessment under `daily_signal_review_2026_09_25`. All 37 protected files were unchanged through the run. Only that results file and this document changed; no new files or folders were created, and nothing was published.

## Preparation for the completed issue review

An offline inspection selected the latest saved issue, **September 25, with 10 stories**. This review will run the draft's complete local Next Signal flow with fresh first-pass decisions, using existing summaries, clusters, and cached article text. It will not rerun news collection, clustering, summarization, Market & Economy, or Worth Knowing.

- Three stories have no qualifying timed evidence and require no model calls.
- Seven stories reach triage. None has a reusable decision for the current triage prompt.
- Review runs only if triage produces an eligible proposal that passes the structural screen: at most seven further calls, for **14 GPT-4o-mini calls maximum across the entire issue**, with a **$0.02 spending limit**.
- The earlier Tasia approval can be reused if the new proposal has exactly the same wording and source. Its actual request payload was verified identical offline. Existing decisions remain preserved; no forced cache refresh.
- New external research is disabled for this bounded review. Any request for it will be flagged, so this run will not establish that branch's behavior or performance on new dates.
- Results and reusable decisions will be stored in the existing review JSON. Live summaries, reports, caches, and the active generator will remain unchanged. No new output folders are needed.

**Integration finding:** `run_pipeline_by_date.py` currently finishes after summarization and does not invoke Next Signal. The immediate review should therefore call the Next Signal draft directly. Connecting the approved implementation to the daily runner is a separate change to review afterward; running the full runner now would regenerate frozen upstream work without exercising this feature.

**Status:** subsequently approved and completed as recorded above. Preparation made no paid calls or publisher requests. The run used 10 of the authorized maximum 14 calls. This is a review of a complete saved issue, not a new-date pipeline run or feature-freeze evidence by itself.

## September 29: four-call follow-up — all four decisions matched expectations

The approved test ran the current review function once on each of four saved proposals, using the same evidence as the previous test. The expected decisions and exact prompt hashes were fixed before the requests. No prompts or code were changed during the run.

| Case | Result | Review |
| --- | --- | --- |
| Estate hearing | Kept: “A hearing is scheduled for October 7.” | Correct. Duplicate full/abbreviated dates no longer caused rejection in this case. |
| Tasia appearance | Kept: “A 45-year-old man charged with murder will appear in court Monday.” | Correct. The separate September 30 hearing no longer displaced the cited Monday appearance. |
| Iran participants | Rejected | Correct. The explanation identifies the cited meeting as Trump–Delcy, not Trump–Iranian officials. |
| Australian disclosure | Rejected | Correct decision: the source describes an already completed revelation, not a new future disclosure. The explanation incorrectly assigns September 24 to the event. |

**Limitation in the last explanation:** the source says Wednesday, while its publication/reference date is September 24, 2026 (Thursday). The model appears to have used that metadata as the event date. This internal explanation was not published as a signal. The correct rejection does not prove that the model's calendar reasoning is reliable.

Both approvals returned only `eligible: true`, which the simplified format now handles successfully. Original signal wording and source indices were preserved. All four responses completed normally; there were no format errors, retries, or publisher requests.

**Cost:** four requests to `gpt-4o-mini-2024-07-18`; 4,230 input tokens, 70 output tokens, zero cached input tokens. Calculated cost **$0.0006765** (about 0.068 cents), below the approved $0.01 limit, using [published GPT-4o-mini rates](https://developers.openai.com/api/docs/models/gpt-4o-mini). This is a token-based calculation, not a billing invoice.

**Recommendation:** the targeted decision test passed, so review this draft for a normal pipeline run. Keep activation subject to the user's review. Feature freeze still needs a normal-run review and a decision on the remaining limitations: clause-scoped uncertainty, the early length limit, conditional countdowns, and behavior on new stories. This test reused known first-pass proposals; it did not test fresh generation or unseen cases. It does not justify more automatic test calls or a model upgrade.

The existing [results file](C:/Users/jchst/Desktop/nominal-news/outputs/next_signals_v18/focused_review.json) contains the complete run under `cited_passage_review`, including raw replies and the explanation caveat. All 35 protected files were unchanged through the run. Only this document and that results file changed afterward; the active generator, draft, caches, saved newsletter, and tests remain unchanged. No new files or folders were created.

## September 29: simplified reply format and cited-passage review applied to draft

The approved follow-up changes only `validate_candidate_signal_semantics` and the overall/review version labels in the existing draft. The active generator remains v18.

- **Simpler replies:** `eligible` must still be a real JSON boolean. Explanations are optional, and model-supplied category labels are ignored. Missing explanations are recorded explicitly. Incomplete replies, missing decisions, and values such as the string `"true"` still stop review safely. The code continues to publish only the original proposed sentence and source.
- **Cited passage first:** the reviewer now sees the fixed proposal and cited passage before other evidence, preserving every source index. Its instructions distinguish duplicate calendar dates, different actors or procedural steps, and genuinely conflicting schedules. An earlier-checkpoint rejection must identify a distinct, directly relevant event, its source, and timing strictly before the proposal. These comparison rules are instructions to the model, not a new deterministic proof of chronology.

**Verification:** 13 tests passed in the existing `FixedProposalReviewChecks` class, with network access blocked. Replaying all six saved paid replies through the revised function removed both response-format errors: the lawsuit approval is retained, and the Iran rejection keeps its source-based explanation. The estate and Tasia rejections remain rejections during replay; this does not establish that the new prompt will correct them. The tests also check boolean validation, optional metadata, original wording/source preservation, evidence numbering, cache reuse, and failure recovery.

**Cost and scope:** zero paid calls or publisher fetches; no new files/folders. Only the existing draft, tests, and this document changed. All 33 protected active-code, output, and cache files remained unchanged, including the paid results file. The model, output cap, number of model stages, triage prompt, and external-generation prompt are unchanged. Only the changed reviewer gets a new model-stage cache version; saved decisions are preserved.

The reply-format fixes were verified offline. The subsequently approved four-call test above also passed its targeted decision checks, with one internal explanation error. This remains an unadopted review draft pending the user's review.

## Earlier September 29 six-call accuracy test — keep the tested draft unadopted

The user approved sending the prepared saved news evidence and proposals to OpenAI. Six GPT-4o-mini requests tested the existing review function; two additional examples were rejected locally without a model request. The first-pass drafts were reused, with their cited passages matched to current cached v18 evidence. No first-pass regeneration, publisher fetches, prompt changes, or publication occurred.

**Result: none of the three supported signals completed review successfully.** Two were wrongly rejected; the third received an approval that failed the response format. Do not promote this draft or freeze the feature based on this test.

| Case | Actual result | Assessment |
| --- | --- | --- |
| Lawsuit hearing | Approved, but omitted the required reason | The draft raises an error. The supported hearing cannot complete this review. |
| Estate hearing | Rejected for an allegedly earlier checkpoint | Wrong: both passages describe October 7, one abbreviated as Oct. 7. A duplicate is not an earlier event. |
| Tasia appearance | Rejected because another passage dates a hearing September 30 | Wrong: the cited passage explicitly supports one man's Monday appearance. The separate September 30 passage does not contradict it; Monday is September 28. |
| Iran participants | Correctly identified the Trump–Delcy meeting, but copied the entire list of reason-code options | The reasoning rejects the wrong participants correctly, but the malformed code causes a review error. |
| Australian disclosure | Rejected the invented future disclosure | Correct substantive rejection: the source describes a disclosure that already occurred. |
| Sheeran tour | Rejected the background tour end | Correct relevance decision. The returned category says timing, although the explanation concerns relevance. |
| Hormuz proposal | Blocked locally for missing future/scheduling language | Correct omission of this saved wording; no evidence that the model handles conditional countdowns generally. |
| Missouri ruling | Blocked locally for missing future/scheduling language | Correct omission of the saved completed action; no model review was needed. |

The first malformed reply stopped the original test runner. Its response was then reused locally, and the five remaining cases ran once each. No model request was retried. In a normal pipeline run, that first error would stop processing before newsletter outputs were replaced. This isolated test continued to expose the other issues without changing the draft.

### Proposed revision (subsequently implemented above)

1. Simplify the reply format. The prompt currently presents reason-code alternatives inside a JSON string, and one response copied that string literally. Use concrete valid examples, remove the category requirement, and avoid treating an otherwise valid approval as an API failure merely because its audit explanation is absent. Preserve strict validation of the actual boolean decision.
2. Make the cited passage the basis of the factual check. A duplicate date or a different event elsewhere is not a contradiction. If review rejects for an earlier checkpoint, require an identifiable, distinct earlier event and its source; equality is insufficient. A later event cannot invalidate an earlier supported checkpoint merely because their dates differ.

These target the observed failures without adding a model stage or upgrading the model. The subsequent offline checks above verify the reply-format changes; the effect on model judgments remains untested. First-pass wording, generalization beyond these known cases, and the remaining limitations below still need evaluation before adoption.

**Cost:** six completed requests to `gpt-4o-mini-2024-07-18`; 5,692 input tokens, 262 output tokens, zero cached input tokens. Calculated cost **$0.001011** (about 0.10 cents), using [published GPT-4o-mini rates](https://developers.openai.com/api/docs/models/gpt-4o-mini), below the $0.01 limit. This is token-based cost, not a billing invoice.

Raw replies, exact evidence/proposals, expected outcomes set before the run, errors, decisions, and file hashes are under `fixed_proposal_review` in the existing [results file](C:/Users/jchst/Desktop/nominal-news/outputs/next_signals_v18/focused_review.json). All 16 protected files were unchanged through the paid run. Only that results file and this document were updated for this iteration; no new files or folders. The active v18 generator, review draft, saved newsletter, and active caches remain unchanged.

## September 29: fixed-proposal review implemented in the existing draft

After user approval, the review draft was rebuilt from active v18 with three changes:

1. The first pass writes the signal. The existing reviewer approves or rejects that exact sentence and source index. Returned replacement wording is ignored; the writer's explanation is excluded from review input.
2. Before review, the draft requires explicit future/scheduling language and rejects extra recognized future dates absent from the cited sentence. The reviewer must then check participants, action, date-to-action relationships, conditions, relevance, and earlier useful checkpoints.
3. Evidence is labeled as candidate source text and unverified time references, rather than proven future events.

The model and number of stages stay unchanged. A structurally rejected proposal needs no review call. Reviews are cached by the exact proposal and source; changed prompts have new cache versions. Existing cache entries remain intact, but these changed prompts will require fresh decisions when an approved run occurs. No review failure is cached as an editorial rejection: completed decisions are saved for reuse, and newsletter outputs are left intact.

**Offline verification:** all 10 tests in `FixedProposalReviewChecks`, added to the existing [test file](C:/Users/jchst/Desktop/nominal-news/tests/test_next_signals.py), passed with simulated responses and network access blocked. They cover preserving the three supported saved drafts, ignoring replacement text/source indices, neutral review inputs, rejecting the saved completed-ruling wording and extra dates before a call, valid fronted dates and qualified wording, malformed responses, cache reuse/invalidation, failure recovery, and the optional external-evidence path. Main-function checks used an in-memory filesystem. The older experimental test class was not run as evidence for this draft.

**Limits:** simulated approvals establish code behavior, not model accuracy. The lexical screen is not a tense parser, and the date parser does not recognize every timing expression. Clause-scoped uncertainty and the early length limit remain unchanged. A better alternative found during review causes rejection, not another call or rewrite. The subsequently approved model test above found blocking failures; this draft remains unadopted.

**Scope and cost:** three existing files edited: the review draft, existing tests, and this document. No new files/folders, paid calls, publisher fetches, or published results. Hashes confirmed all 14 protected active files, outputs, and caches unchanged. The active generator remains v18.

## Earlier offline follow-up: preserve the first draft; review it without rewriting

No additional model calls were made. This investigation traced the timing gate, tried sentence preparation in memory, and inspected all ten saved positive triage drafts across the 30-story review set. It changes the recommendation: do not expand the strict quote renderer into an automatic sentence writer.

### Why Missouri passes the current gate

In the saved sentence, `November` dates the elections. The future cue `will` belongs to the following clause, “which will determine control of Congress.” It is only 18 characters from the date. `_anchor_is_action_timing` accepts a future cue within 220 characters, so that separate action causes the entire passage about an already completed ruling to be labeled as future evidence.

Two other rules make a vocabulary patch unreliable:

- The past-action screen uses a finite verb list and a 90-character window. Replacing “halted” with the already recognized “rejected” still lets the saved long sentence pass because the date is beyond that window.
- A shorter completed-action sentence, “The court halted the map for November elections,” passes through the fallback for text under 110 characters. Removing the relative clause therefore does not solve the general problem.

A simple comma-boundary check rejected the bad cross-clause association but also rejected a valid construction, “On Monday, the team will publish its findings.” It also failed to distinguish an old Monday schedule from a new Wednesday schedule in a postponement sentence. These probes argue against another quick regex rule presented as a complete date-to-action solution.

### What automatic wording preparation can and cannot do

Enumerating the draft's permitted source cuts can produce the correct 12-word lawsuit hearing and complete estate hearing. However, the same structural checks also admit unrelated past guidance and solidarity statements from the lawsuit coverage. Formatting a passage successfully does not make it a valid future event.

The Monday Tasia appearance still requires resolving “He.” Enumerating prefixes of the preceding sentence produced eight candidate replacements: one correct phrase and seven malformed expansions such as “A 45-year-old man turned was charged…”. Do not offer those as supposedly valid choices or add a topic-specific noun rule. The later September 30 hearing can be copied, but retaining it does not recover the earlier Monday appearance.

### Useful work already exists in the first pass

The saved triage drafts already contain:

| Case | First-pass wording |
| --- | --- |
| Lawsuit | “Kelly scheduled a hearing on the CNN/MS NOW/Politico suit for Wednesday afternoon.” |
| Estate | “A hearing is scheduled for October 7.” |
| Tasia | “A 45-year-old man charged with murder will appear in court Monday.” |

Those three drafts are supported by their saved source passages. The current second pass independently writes another signal; the tested strict renderer likewise required new wording fields and discarded these usable sentences.

This does **not** make triage trustworthy by default. Its saved Iran draft has wrong participants, the Australian website-breach draft invents a future revelation, the Sheeran draft concerns background tour scheduling, and the Missouri draft describes a completed action. The reviewer still has substantial work to do. Historical triage outputs are diagnostic examples, not cache hits for changed v18 evidence.

### Recommendation from that analysis (now implemented in the review draft above)

1. Preserve the first-pass proposed sentence and its source index. Replace the existing second-stage rewrite with approval or rejection of that exact sentence against the source context and all available candidates. Omit the first writer's reasoning from the review input. The review must still check participants, action, timing, conditions, relevance, and whether an earlier material checkpoint exists.
2. Add a conservative structural screen on the proposed sentence itself: require explicit future/scheduling language in the prose before spending the review call. In the ten saved positive drafts, this preserves the eight with such language and screens out the completed Missouri draft and the unstarted Hormuz proposal. This is not semantic proof: wrong-participant and invented-future-action drafts can still contain “will,” and valid terse calendar fragments would need clearer prose. The model review remains necessary.
3. Present heuristic evidence neutrally as candidate source passages and detected time references. The current label “Grounded future sentence” overstates what the distance/word-list checks establish. Removing that label is an input correction; whether it improves model decisions is untested.

Keep the same model and existing number of stages. Version the changed review contract and include the exact proposed sentence/source index in its cache context. Never publish a replacement sentence returned by the reviewer. If review identifies an earlier checkpoint that the first draft missed, reject that draft; this narrower flow may sacrifice some recovery of better alternatives and must be assessed before adoption.

This proposal is a simplification, not a demonstrated cure. The earlier extra-checker experiment wrongly approved a two-person claim, so an approve/reject model response is not a guarantee of factual support. A future test would need the erroneous first-pass examples above alongside the three useful drafts. Implementation was subsequently approved; a new paid test remains unapproved.

**This round:** only this existing review document changed. Code, the unadopted proposal, saved outputs, caches, and mobile-preview files remain unchanged. No new files/folders, API requests, publisher fetches, or new model judgments.

## Latest paid result, September 29: do not adopt the wording draft

The approved six-call test is complete. It used the actual draft's final-stage function on the six prepared sets of cached evidence, with the exact prompt hashes and draft hash checked before each request. Only the estate hearing survived. Two clearly useful signals were lost, and the Missouri selection still showed the original misunderstanding of a completed ruling.

| Case | Result | Review |
| --- | --- | --- |
| Lawsuit hearing | Omitted | The model correctly chose the hearing, but copied 23 words through the later Justice Department clause and stopped at an unsupported boundary. The code rejected that cut. The useful 12-word hearing clause was available. |
| Estate hearing | Kept: “A hearing is scheduled for October 7.” | Supported wording, correct event. |
| Iran response | Omitted | Correct substantive rejection: the model identified the meeting as Trump–Delcy Rodriguez and unrelated to the Iranian president's response. |
| Hormuz proposal | Omitted | Conservative outcome. The returned story state recognized contingency on U.S. actions, but its generic rejection reason did not explicitly explain the missing start of the seven-day clock. |
| Tasia appearance | Omitted | The model selected the correct one-person court sentence but left `subject_quote` empty. The code rejected the unresolved “He,” losing a supported signal. |
| Missouri ruling | Omitted by length limit | The model selected the completed ruling and called it a “future ruling” in its explanation. The 25-word quote failed the length limit. The existing grounding check accepts that quote, so this is not a demonstrated semantic fix. |

### What the test establishes

The code stopped the tested free rewriting, but the model did not reliably satisfy the new output rules. The draft kept one of the three clearly supported checkpoints targeted for retention. Omission of the Missouri error was incidental to length. Do not count all omitted signals as successful decisions or promote this draft.

The next proposed investigation is offline: determine whether valid display candidates can be prepared in code before the existing final call, leaving the model to select or reject them by identifier. That could remove the two demonstrated quote-construction failures, but it would not by itself fix the Missouri event-selection error. Evaluate both limitations before another paid run; no new implementation or further calls were authorized by this test.

### Cost and preservation

- Six requests completed, with normal `stop` finishes. No retries, extra model stages, or publisher fetches.
- Returned model: `gpt-4o-mini-2024-07-18`. API-reported usage: 9,519 input tokens, 551 output tokens, no cached input tokens.
- Calculated cost: **$0.00175845** (about 0.18 cents), below the approved $0.01 budget, at [the published GPT-4o-mini rates](https://developers.openai.com/api/docs/models/gpt-4o-mini). This uses reported token usage, not a billing-dashboard invoice.
- Raw responses, source evidence, final decisions, six reusable decision entries, and file hashes are appended under `source_wording_check` in the existing [results file](C:/Users/jchst/Desktop/nominal-news/outputs/next_signals_v18/focused_review.json).
- Hashes confirmed 14 protected files unchanged: the active generator, draft, app/UI files, saved summaries/reports, and active caches. No output was published. Only the existing results file and this review document changed; no new files or folders.

## Wording draft tested above

After approval to continue, the existing proposal file was rebuilt from the active v18 date repair. It replaces the earlier unadopted extra-checker proposal. The active generator and saved newsletter remain unchanged.

**Scope:** one new local helper, `_render_source_signal`, and changes to the existing `validate_candidate_signal_semantics` function. The overall draft version and that function's cache version changed. Every other existing function matches active v18, including the approved calendar repair. No new model stage, dependency, or retry loop was added.

### What the draft enforces

- The final model returns `event_quote` and optional `subject_quote`. The code constructs the published wording from those fields. A separate freely written `signal` field is ignored.
- The quote must copy the beginning of the cited sentence without changing or rearranging words. It may use the whole sentence or stop immediately before a trailing `, and` clause. A conservative check blocks that cut when the removed text contains recognized conditions, uncertainty, or negation.
- A leading `He` or `She` can be replaced only with a copied phrase starting the immediately preceding sentence. This first draft requires an explicit singular determiner (`A`, `An`, or `One`) and rejects coordinated/plural-quantity phrases. Other unresolved pronouns and unsupported replacements are omitted.
- The existing final length, timing, uncertainty, and relevance gates still run. Accepted and wording-rejected decisions include a `source_construction` record so the selected quote, replacement, and rendered text can be reviewed.

These checks constrain editing; they do not prove English sentence meaning. The model must still choose a relevant future event, identify a correct antecedent, and retain governing conditions. The suffix checks recognize a limited set of grammatical cues. This is not permission to regard every copied prefix as a supported standalone claim.

### Offline results

Twenty-two simulated-response cases passed through the actual final-stage function, with network requests and test file writes blocked:

- Kept the lawsuit hearing, estate hearing, and one man's Monday court appearance using the saved source wording.
- Rejected the saved invented Missouri decision and Iran participants, the two-person Tasia rewrite, an extra date, internal rewriting, and tested removals of conditions/uncertainty.
- A response containing a correct source quote plus a contradictory free-form `signal` produced the source-derived wording. This specifically tests the disconnect that defeated the earlier quote-and-boolean proposal.
- Retained editorial rejection, rejected malformed source references and missing quotes, and enforced the final length limit.
- Confirmed that storm shortening is still rejected under this narrow draft. Its first clause ends before `as`, outside the permitted shortening rule; the whole forecast exceeds 22 words. The other known uncertainty bug also remains unchanged.

These are code checks using supplied responses, not an assessment of new model decisions. No paid model output was generated. The old tests under `tests/test_next_signals.py` target an earlier broad experiment and were not used to validate this draft.

Three cache checks confirmed that triage and external-final keys stay the same for unchanged evidence; only the existing semantic-adjudication stage gets a new version. Old cache entries remain intact. The model and output cap (240 tokens for this stage) are unchanged. The prompt is longer, so equal call counts do not imply identical token cost.

### Review decision and limits

Keep this as a review draft. It demonstrably blocks the tested rewrites, but its stricter extraction can omit useful signals. For example, the saved Pope meeting sentence needs shortening and pronoun handling outside the supported rules. Relevance, conditional countdowns, clause-scoped uncertainty, the early triage length rejection, and an unseen-date review remain before feature freeze.

The six-call test was subsequently authorized and completed, as recorded above. Its failures leave this proposal unadopted.

**Files and cost:** only the existing proposal file and this review document changed. Hashes confirmed 13 protected files unchanged, including active code, app/UI files, summaries, reports, and caches. No new files/folders, publisher fetches, API requests, or mobile-preview changes.

The prompt uses explicit source constraints and small formatting examples, consistent with [OpenAI's prompting guidance](https://developers.openai.com/api/docs/guides/prompt-engineering#few-shot-learning). That documentation does not establish this draft's semantic accuracy; it still needs the agreed review and evaluation.

## Prior offline analysis: basis for this draft

Reconstructed the current v18 evidence for all 30 saved stories: 47 candidate passages across 18 stories. API calls, network access, and file writes were blocked during the replay. Reviewed those passages against the earlier saved errors. This analysis made no new model decisions and did not regenerate the newsletter.

### Finding

The final stage still produces a freely written signal. Requiring supporting quotes did not fix this in the previous experiment because the displayed sentence could contradict the quotes. The next proposal should connect the displayed wording directly to the chosen source passage, within the existing final model call.

Copying an entire sentence is too restrictive: the useful lawsuit sentence has 34 words, and the three storm forecasts have 39, 25, and 23 words. The final limit is 22. However, complete shorter event clauses are available in some of these sentences. Arbitrarily deleting words is unsafe because it can remove a condition or change the actor.

### Proposed next iteration

- Have the existing final stage select a complete event clause from one evidence sentence, including its timing and any qualifications that govern that event.
- Build the display text from that selected clause instead of accepting a separate free-form rewrite. Verify the selected text exists in the cited sentence; permit only defined punctuation/spacing normalization and a narrowly specified pronoun replacement.
- A pronoun replacement must identify its source antecedent and preserve singular/plural meaning. Quote membership alone cannot establish that the antecedent is correct. The implementation must state what it can actually enforce; ambiguous cases should be omitted rather than filled from the story headline.
- Keep the existing exact-story relevance decision. Verbatim text can still concern an unrelated person, a completed action, or an unstarted conditional plan.
- Keep the current model and number of stages. Do not add a checker or retry loop. A changed final-stage contract needs its own cache version; keep existing cached entries rather than deleting them.

These are proposed constraints, not a tested production solution. Selecting clause boundaries, resolving an antecedent, and preserving conditions still involve semantic judgment. In particular, merely accepting any substring would let a model truncate the condition from the Hormuz proposal. That must remain a rejection case in the review.

### Concrete examples from saved evidence

The following examples were prepared during this offline review. They are not new model outputs or published changes.

| Case | Source-based wording or decision | What it establishes |
| --- | --- | --- |
| Lawsuit hearing | “Kelly scheduled a hearing on the CNN/MS NOW/Politico suit for Wednesday afternoon.” | The 12-word event clause is a contiguous part of the source. The later clause about the Justice Department's response is unnecessary to state the checkpoint. |
| Estate hearing | “A hearing is scheduled for October 7.” | The complete seven-word source sentence already works under its story card. |
| Tasia court appearance | “A 45-year-old man was charged with murder and will appear in court Monday.” | The 13-word example preserves the source sentence after replacing “He” with its immediate preceding antecedent. “Two men” is not that antecedent. This remains a case for testing the proposed pronoun rule. |
| Missouri ruling | Reject the saved “justices will decide” wording. | The selected passage says the justices already “halted” a ruling. A separate November referendum passage must be evaluated on its own; it cannot supply a date for an invented judicial action. |
| Iran response | Omit the Trump–Delcy Rodriguez meeting from this cluster. | Accurate copying alone would not establish relevance to the Iranian president's response. |
| Hormuz proposal | Preserve the condition and require an established start before treating seven days as a live countdown. | The existence of a proposed duration does not establish a scheduled checkpoint. |

The storm has a valid 20-word source clause: “Overnight into Sunday morning, rounds of heavy rain and strong wind gusts will continue from Maryland to southern New England.” The current validator still rejects that wording because a different clause calls flash flooding “possible.” Do not claim the wording proposal alone fixes this; uncertainty needs to be tied to the selected event while retaining any condition that governs it.

### Remaining work before freezing

After reviewing the wording change, address the separate deterministic defects: unsupported additional dates, uncertainty applied to the wrong clause, and the 22-word rejection at triage before final editing. Moving the length check may allow some stories to reach the already-existing final pass, so unchanged stages do not guarantee an unchanged total call count.

Review the known bad outputs and useful retained signals together, then one previously unseen date. Paid evaluation requires separate approval for a specific model, maximum call count, and estimated cost. No such evaluation was requested or performed here.

**This analysis changed only this existing review document.** No code, caches, saved outputs, folders, or mobile-preview content changed. Zero API calls and zero publisher fetches.

## Diagnosis and agreed scope

### Working guidelines

- Work from the feature goal and the workflow in **Continue Project Context**: inspect outputs and code, run useful offline checks, explain the proposed resolution, and review each version with the user before adopting it.
- A Next Signal is an explicitly timed future checkpoint that could materially update the specific article cluster. Sharing a person, topic, or date is insufficient. Omit the feature when no supported checkpoint exists.
- Changes must generalize across arbitrary stories and dates. Do not add production exceptions for particular news topics or examples.
- Minimize daily operating cost. Use saved responses, cached articles, and offline checks for ordinary analysis. Do not default to extra model passes or model upgrades.
- Ask for permission before paid API calls. Explain the decision or missing information the call will resolve, the model, maximum call count, and estimated cost. Cheap or moderate calls can be justified; repeated paid testing is not the default.
- Preserve caches and the frozen systems. Cleanup remains pending agreement on what to retain.

### Offline diagnosis completed

Re-read the previous discussion and traced the active v17. Seven offline probes confirmed:

1. **The final validator does not establish factual support.** It accepted the saved incorrect Iran meeting and invented Missouri decision. It also accepted the two-person Tasia wording from the failed v18 experiment; active v17's original Tasia wording was singular and supported. Matching a date does not verify who performs the action or whether that action is stated by the source.
2. **Sentence splitting loses calendar details.** `Oct. 7, 2026` becomes `Oct.` plus `7, 2026.`; a historical `Tuesday, Sept. 22, 2026` caption becomes a future Tuesday candidate.
3. **Uncertainty is checked across the entire sentence.** A possible flash-flood clause caused v17 to reject a signal that merely repeated the source's unqualified prediction that rain and wind would continue.
4. **An extra unsupported date can pass.** The final validator requires one overlapping time key, rather than requiring all generated times to be supported.

Code inspection also found that triage can discard a provisional signal for exceeding 22 words before the final editor sees it. This makes an early wording issue decide whether the story gets evaluated further.

### Approved first patch: preserve complete dates and sentences

Use the existing shared sentence splitter in the article-excerpt stage, protect abbreviated month dates from splitting, and allow the calendar parser to recognize the abbreviation's period. This is a general text/date repair, with no additional model call or news-topic rule.

Applied only in memory, this repair passed four calendar checks: abbreviated and full month dates kept their day, a future year remained intact, and a historical caption was rejected.

A replay of all 30 saved stories used cached publisher text and made **zero API calls, zero publisher requests, and zero file writes**:

- Model-ready candidate sentences changed from 55 to 47.
- Stories with candidates changed from 20 to 18. Two stories would avoid entering the model stage on this evidence.
- Eleven stories had changed evidence; nineteen had identical evidence.
- The estate hearing retained October 7, and the source text exposed an additional September 30 hearing continuation in the Tasia coverage.
- Some noise remains: a recurring Sunday activity became another Pope-visit candidate. This repair is not a complete future-event or relevance solution.

The in-memory replay establishes what the parser changes, not what new model decisions would be. A later approved run would reuse cached decisions where the evidence is unchanged; changed inputs may need fresh decisions. No such run is authorized by this diagnosis alone.

**Approved scope:** start from active v17 with this narrow parser repair. This has now been applied and verified as described above. The experimental extra-checker draft remains unadopted. Further work should address the remaining source-to-signal wording contract within the existing model stages. A date repair alone will not prevent wrong participants or invented actions, and arbitrary paraphrases cannot be fully verified with simple date/keyword checks.

During the initial diagnosis only this review document was updated. The subsequent approved implementation changed the active generator as recorded above.

---

## Historical experiment: independent checker passed four of five cases

The approved five-call check is complete. The independent checker rejected the invented Missouri decision and accepted the three source-supported signals. It still approved the Tasia sentence claiming that two men would appear in court, although the source schedules one man's appearance. **Do not adopt this revision yet.**

| Saved signal | Expected | Actual | Assessment |
| --- | --- | --- | --- |
| CNN/MS NOW/Politico hearing on Wednesday | Approve | Approve | Correct |
| Estate hearing on October 7 | Approve | Approve | Correct |
| Two men appearing in court on Monday in the Tasia case | Reject | Approve | Incorrect: one person's scheduled action was assigned to two people |
| Justices deciding on the Missouri map by November | Reject | Reject | Correct: the source describes a completed ruling and dates the elections |
| Pope meeting seven French abuse victims on Sunday | Approve for source accuracy | Approve | Correct on the source claim; editorial value still needs review |

The Tasia checker explicitly reasoned that a source scheduling one man's appearance supported a claim about two men. This is a failed factual comparison, not a missing source or a failure to run the checker.

This test used the actual checker function on the five exact saved signals and their original supporting passages. It did not regenerate candidates or repeat the full pipeline. Calls used `gpt-4o-mini`; API-reported usage was 2,560 input tokens and 229 output tokens. All five responses completed normally.

Results and the five cache entries were appended under `independent_claim_check` in the existing [results file](C:/Users/jchst/Desktop/nominal-news/outputs/next_signals_v18/focused_review.json). No new files or folders were created. Hashes confirmed that application code, draft code, templates, stylesheet, active summaries, reports, and caches remained unchanged. No result was applied to the mobile preview.

The model-comparison proposal following this round has been withdrawn. The current recommendation above starts with offline code and evidence corrections. No model upgrade or further paid run followed these five checks.

## Historical tested revision — since replaced in the proposal file

The earlier v18 proposal added an independent check after the finished signal was written. It received that exact sentence and the cited source, without the writer's earlier verdict or explanation. It could approve or reject; it could not rewrite the signal. That proposal has since been replaced in the existing draft file; its saved results remain historical evidence.

- Checks participants, number of people, action, timing, and necessary conditions.
- Keeps approved wording unchanged and records rejected wording, source, and reason.
- Caches the verdict by signal wording, source evidence, model, and checker version.
- Counts the additional call in the existing report. Failed or incomplete checks stop the date's run before published files are written and are not cached as editorial rejections.
- Uses the same model. It does not change which candidates are generated or reconsider their editorial relevance.

Offline checks passed using simulated verdicts: approval/rejection handling, exact wording preservation, cache reuse, changed-wording cache separation, and safe handling of failed responses. These checks exercised the saved stories in memory and made no paid calls or new output files. They do not establish whether the model can identify the factual errors.

The five-call test described above followed these offline checks. Active v17 and the mobile preview still use the existing newsletter.

## Previous focused run — September 26, 2026

## Decision from that run: do not adopt the tested draft

The first focused proposal changed only v17's final semantic review, plus its version identifiers. It asked for source quotes and participant/action checks in the existing model call. The results below belong to that earlier revision; its exact code hash is saved in `outputs/next_signals_v18/focused_review.json`.

One approved run covered all 30 stories from September 20, 23, and 25. It produced **five signals: three supported by the supplied source text, including one with debatable editorial value, and two incorrect signals**. The draft has not replaced active v17.

The input articles were already saved locally. This review checks signals against that evidence; it does not independently verify the news reports.

## Main findings

| Case | Focused v18 result | Assessment |
| --- | --- | --- |
| Iran response, Sep 23 | No signal | Improvement: removed the meeting falsely attributed to Iranian officials. The actual meeting passage concerns Trump and Delcy Rodriguez. |
| Missouri map, Sep 25 | “The justices will decide on the Missouri congressional map by November 2026.” | Still wrong: the quoted court action is completed. November dates the elections, not a future court decision. A separate passage mentions a referendum, but the draft did not select it. |
| Tasia Fortune, Sep 25 | “Two men charged with murder will appear in court on Monday.” | New error: the source schedules one man's appearance. It separately says two men were arrested. V17's singular wording was supported. |
| Lawsuit hearing, Sep 20 | “A hearing on the CNN/MS NOW/Politico lawsuit is scheduled for Wednesday afternoon.” | Supported procedural continuation; retained. |
| Estate hearing, Sep 23 | “Check back after the hearing scheduled for October 7.” | Supported procedural continuation; retained. “Check back” is redundant under the feature heading. |
| Hormuz proposal, Sep 25 | No signal | Removes v17's misleading wording. Conservative omission is reasonable because implementation depends on unmet conditions; the model's generic rejection reason does not show it understood that distinction. |
| Nor'easter, Sep 25 | No signal | Still misses supported future weather information. The model claimed supporting quotes were missing despite the explicit overnight forecast. |
| Pope visit, Sep 25 | “Pope Leo XIV will meet seven French victims of abuse on Sunday.” | Supported by the source. Its material value as a reason to check back remains an editorial question. |

## What this tells us

Requiring real quotes did not establish that the generated sentence followed from those quotes. Both incorrect signals included genuine passages and returned `same_participants=true` and `same_action=true`.

For Tasia, the action quote says “He,” but the output says “Two men.” For Missouri, the action quote says the justices “halted” a ruling, but the output says they “will decide.” The code correctly checked that the quotes existed; the model incorrectly judged their meaning.

The rejected outputs also often returned positive check flags and inconsistent relevance fields. An omission alone is therefore not proof that the model applied the intended rule correctly.

## Review of all 30 stories

“Omit” means the run produced no signal. Findings about missing evidence refer to the saved inputs, not to every possible source on the internet.

| Date | Story | Result and review |
| --- | --- | --- |
| Sep 20 | Austin ICE shooting | Omit; no usable timed checkpoint in the saved inputs. |
| Sep 20 | Proposed triumphal arch | Omit; no usable timed checkpoint. |
| Sep 20 | Sheeran controversy | Omit; appropriately rejects the wider tour as a next step in this controversy. Rejection reason improved from v17's timing-only failure. |
| Sep 20 | Haitian assassination suspects extradited | Omit; supplied timed text describes a past request for comment. |
| Sep 20 | Ukrainian drone attack | Omit; no usable timed checkpoint. |
| Sep 20 | CNN/MS NOW/Politico lawsuit | Keep; Wednesday hearing is supported. |
| Sep 20 | German regional election result | Omit; no usable timed checkpoint. |
| Sep 20 | Trump–Mamdani meeting | Omit; meeting already happened in the story; the UN itinerary is background. |
| Sep 20 | Houthi missile attack | Omit; a news-roundup date does not establish a future event. |
| Sep 20 | Airport cable disruption | Omit; one traveler's hoped-for flight does not update the repaired system. |
| Sep 23 | Trump–Xi summit | Omit, but still for the old triage length limit. This run does not validate the model's ability to reject unrelated itinerary items. |
| Sep 23 | Weinstein sentencing | Omit; no usable timed checkpoint. |
| Sep 23 | Estate restraining order | Keep; October 7 hearing is supported. |
| Sep 23 | Arctic security agreement | Omit; supplied timing describes the completed signing. |
| Sep 23 | Iranian response to Trump | Omit; improvement over v17's incorrect participants. |
| Sep 23 | Senate war-powers vote | Omit; no usable timed checkpoint. |
| Sep 23 | Iranian president's UN speech | Omit; no usable timed checkpoint. |
| Sep 23 | Netanyahu walkout | Omit; no usable timed checkpoint. |
| Sep 23 | Australian government website breach | Omit; the dated revelation is a completed event. Rejection now gives that substantive reason. |
| Sep 23 | Hurricane Polo | Omit; timed future candidates concern a different storm near Hawaii. |
| Sep 25 | Hormuz reopening proposal | Omit; conservative outcome, but the generic explanation is weak. |
| Sep 25 | Pope's technology warning | Omit; no usable timed checkpoint. |
| Sep 25 | White House access restored | Omit; supplied hearing and ruling are completed events. |
| Sep 25 | New Jersey lieutenant governor resigns | Omit; candidate dates come from old election captions. |
| Sep 25 | Tasia Fortune case | Incorrect signal; expands one man's Monday appearance to two men. |
| Sep 25 | University security report | Omit; a recurring newsletter promotion is not a story checkpoint. |
| Sep 25 | Missouri congressional map | Incorrect signal; invents a future judicial decision. |
| Sep 25 | Nor'easter | Omit; missed opportunity despite an explicit future forecast. |
| Sep 25 | Pope's France visit | Source-supported signal; editorial value remains debatable. |
| Sep 25 | Butler rally survivor dies | Omit; no usable timed checkpoint. |

## Run record and file changes

- Completed 10 paid model calls using the draft's existing `gpt-4o-mini` configuration: two, three, and five calls by date.
- Reused all 20 applicable triage decisions. The other 10 stories had no model-ready timed evidence. No quality reruns were performed.
- API-reported usage: 15,987 input tokens and 1,043 output tokens. No dollar estimate was calculated.
- The first connection attempt was blocked by the sandbox before a response returned. The authorized run then completed with network access; that failed attempt was not cached as an editorial decision.
- Publisher and external research fetches: zero. The reconstructed model evidence matched the saved v17 inputs for every story.
- Ran the actual draft's main function for each date, capturing its normal report, summary, and cache writes in memory. The console's normal “Wrote” messages refer to these captured writes.
- Saved one new artifact: `outputs/next_signals_v18/focused_review.json`. It contains reports, raw model responses, token usage, and the ten new cached decisions.
- Hashes confirmed active generator, proposal, root reports, summaries, and caches stayed unchanged. All non-signal summary fields also matched.
- Updated this existing review document. No new folders, code edits, cleanup, commits, or publication.

## Proposed next step from that run

Add a separate check of the finished signal against its cited passage. It would only approve or reject, checking who is involved, how many people are involved, what action occurs, and whether that action is future or completed. Keep relevance rules and the model unchanged for this focused comparison.

For this batch, checking only the five generated signals would add five model calls. The code is now drafted as described at the top of this document; its model behavior has not been tested or approved for publication.

---

# Earlier broad experiment — historical record

The sections below describe the earlier file under `experiments/`, not the focused root-level v18 draft reviewed above. Their test results do not validate the current proposal.

# Next Signal review: v17 findings and v18 experiment

## Decision

**Do not freeze When to Check Back yet.** v17 has source-attribution and relevance failures. The v18 experiment fixes structural defects but still produces inconsistent semantic decisions, so it has not been promoted.

The production `generate_next_signals.py` was restored byte-for-byte from the local v17 snapshot taken at the start of this review. The production reports, summary files, and caches were not replaced. Existing uncommitted project work is preserved.

The experimental implementation is `experiments/generate_next_signals_v18.py`. Its outputs and caches are under `outputs/next_signals_v18/`. The original code, reports, summaries, and reconstructed model inputs are under `outputs/next_signals_v17/`. Generated output folders are ignored by Git.

## Findings in v17

This review covers all 30 saved stories from September 20, 23, and 25, 2026. It compares signal decisions with supplied article evidence; it does not independently verify the news articles themselves.

| Case | Finding |
| --- | --- |
| September 23, Iran response | Published a Trump–Iran meeting even though the source's “the two” referred to Trump and Venezuelan interim president Delcy Rodriguez. |
| September 25, Missouri map | Invented a future Supreme Court finalization. The source describes a completed ruling and separately identifies a November referendum. |
| September 25, Hormuz proposal | Recast “if the US meets [Iran's] conditions” as “U.S. conditions.” An unaccepted conditional proposal also does not necessarily start a seven-day clock. |
| September 20, Sheeran | Omission was caused by a timing check, while both semantic passes still treated the wider tour as the central story. That is not a reliable rejection of the irrelevant tour ending. |
| September 23, Trump–Xi | The triage proposal was discarded for length before refinement. Earlier versions had also accepted an unrelated itinerary item. |
| September 25, storm | The generated wording was rejected for lost uncertainty. Rejected wording was not retained, limiting diagnosis. |
| Dates across reports | Abbreviated calendar dates were split into fragments, losing days/years and allowing historical captions to resemble future checkpoints. |

The lawsuit hearing, estate hearing, and charged man's court appearance have clear procedural links to their stories. The narrower Pope-visit checkpoint is a more judgment-dependent case: its relevance depends on how much substantive information the scheduled meeting can add to the visit story.

## Changes implemented in the experiment

- Preserve complete abbreviated dates and remove invisible characters that disrupt sentence boundaries.
- Keep nearby source context for unnamed actions as well as pronouns.
- Allow long triage proposals to reach refinement; enforce the display limit on the final draft.
- Require a quoted source passage, actor, action, timing, conditions, and expected new information before drafting.
- Require the action and timing to come from the candidate sentence that passed the structural gate. Nearby context may explain actors but cannot supply a different checkpoint.
- Verify the exact drafted wording separately, without allowing that verifier to rewrite it.
- Check every generated calendar reference, including an unsupported extra date alongside a valid one.
- Scope uncertainty checks to the supported claim while reviewing necessary conditions in the full source context.
- Retain model inputs, rejected drafts, source plans, and final verification checks in reports.
- Cache stages independently by model and prompt version. Save successful decisions as they complete, and leave published files intact if an API call fails. Do not cache outages or truncated responses as editorial rejections.

No production news-topic exceptions were added. Summarization, clustering, Worth Knowing, Market & Economy, and the site UI were outside the change.

## What the tests established

**15 offline tests pass.** They check calendar parsing, evidence boundaries, unsupported dates, uncertainty, literal verification booleans, cache isolation, API-failure behavior, rejected-draft audit data, and preservation of non-signal summary fields.

Live runs used the cached articles: the final three-date run made zero publisher fetches and zero external research fetches. Repeated runs reused stage caches rather than using `--refresh`. Non-signal summary fields were compared with the v17 snapshots and remained identical.

The semantic model comparison did not establish a freeze-ready configuration:

- Repeating checks with `gpt-4o-mini` still accepted the unrelated tour ending and rejected useful hearings.
- `gpt-5-mini` remained inconsistent about conditional timelines and harmless omitted descriptors.
- `gpt-5.4-mini` passed all six invented claim cases in one iteration, but the final version passed **5/6**. It incorrectly treated the omitted adjective “repaired” as a necessary condition in an otherwise supported future vote.
- The final three-date run accepted the **National Archives visit** as a checkpoint for the trade summit and omitted the useful estate hearing. This is a semantic regression despite the passing structural tests.

Final experimental outputs:

| Date | Published experimental signals | Assessment |
| --- | --- | --- |
| September 20 | CNN/MS NOW/Politico hearing | Supported; unrelated tour ending omitted. |
| September 23 | Trump–Xi National Archives visit | Background itinerary false positive; estate hearing omitted. |
| September 25 | Tasia court appearance; overnight storm continuation | Supported wording; misleading conditional proposal omitted. Missouri and Pope checkpoints omitted. |

These experimental results were not applied to the newsletter.

## Next refinement

1. **Build paired relevance cases before another production change.** Use the same actors and dates with different central developments, so a model must distinguish a direct next step from a background itinerary. Include the current false positives and valid procedural continuations.
2. **Evaluate relevance separately from source fidelity.** A sentence can be copied accurately and still be an irrelevant signal. Require an explicit account of what unresolved story state the checkpoint can change, then independently check whether the wording follows from the source.
3. **Separate logical prerequisites from descriptive detail.** Conditions such as “if approval is granted” must survive paraphrasing. Omitting an established adjective or location does not automatically change the claim.
4. **Require stable results across repeat evaluations.** One passing batch is insufficient when small prompt changes cause old false positives to return. Choose the model and decision contract against the full benchmark.
5. **Promote the tested structural changes together with a passing semantic path, then inspect a new news date.** Keep the final newsletter chronology audit as subsequent work.

The user constraint remains: every production change must generalize across arbitrary stories and dates. Example stories belong in test data, not production exception lists.

## Running the saved tests

From the project folder:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

The optional live check uses the experimental generator and the configured OpenAI API key. Results are cached by fixture, model configuration, and verifier version:

```powershell
.\.venv\Scripts\python.exe -X utf8 tests/evaluate_next_signal_claims.py --report outputs/next_signals_v18/claim_evaluation_gpt54mini.json
```

The experiment defaults to `gpt-4o-mini` for triage and `gpt-5.4-mini` with low reasoning effort for semantic work. `NN_NEXT_SIGNAL_SEMANTIC_MODEL` changes the experimental semantic model. Production v17 retains its original model configuration.

GPT-5.4 Mini has higher token rates: OpenAI lists $0.75 per million input tokens and $4.50 per million output tokens; reasoning also consumes the completion budget. These are published rates, not measured total costs for this review. [Model documentation](https://developers.openai.com/api/docs/models/gpt-5.4-mini), [reasoning documentation](https://developers.openai.com/api/docs/guides/reasoning).
