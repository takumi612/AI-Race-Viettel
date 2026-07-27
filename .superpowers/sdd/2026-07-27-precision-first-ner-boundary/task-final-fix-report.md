# Final whole-branch review fix report

## Scope

Resolved the two actionable findings from the final review:

1. Runtime loading of the document-level NER calibration artifact.
2. Consistency among the Qwen entity-validation prompt, guided JSON schema,
   and strict trim parser.

The optional cumulative-maximum fallback change was intentionally deferred
because the production `RequiredQwenRefiner` snapshot/delta path is covered
and no related regression was found.

## Root causes

- `write_document_ner_calibration()` introduced schema version 2 while
  `load_ner_confidence_threshold()` still accepted only schema version 1.
  Phase 12 therefore rejected a calibration file produced by the training
  phase even though both versions store the selected runtime threshold in the
  same `confidence_threshold` field.
- The Qwen validator had three conflicting contracts:
  - the prompt demonstrated `relative_start: 0`;
  - the guided schema required only `action` and left trim fields optional;
  - the parser correctly required both boundaries to be strictly inside the
    original entity (`0 < start < end < len(text)`).

## Changes

- Updated the NER calibration loader to accept schema versions 1 and 2 while
  preserving the existing default, threshold conversion, range validation,
  and fail-closed behavior for unknown schema IDs/versions.
- Added a CPU round-trip unit test that writes a schema-v2 document
  calibration and loads its selected threshold through the runtime loader.
- Replaced the loose Qwen guided schema with action-specific `oneOf` variants:
  - `keep` and `drop` require only `action`;
  - `trim` requires `action`, `relative_start`, `relative_end`, and
    `entity_type`, forbids extra fields, and applies entity-length-aware
    interior bounds;
  - entities shorter than three characters expose no impossible trim variant.
- Made sampling schemas entity-specific so the guided bounds match the entity
  being validated.
- Replaced the invalid trim prompt example with an entity-length-aware example
  that the runtime parser accepts. Very short entities explicitly offer only
  keep/drop.
- Added focused CPU tests covering prompt-example parsing, action-specific
  schema requirements, a valid trim, and invalid attempts that preserve either
  original boundary or collapse the span.

## Verification

- Red phase:
  - the schema-v2 calibration round trip failed with
    `NER calibration schema is unsupported`;
  - the prompt trim example failed strict-boundary parsing;
  - the guided-schema contract test failed because the schema was not
    entity-aware/action-specific.
- Focused green run:
  - `6 passed` for the new calibration/prompt/schema/trim cases.
- Relevant regression run:
  - `50 passed` across NER calibration/filtering, Qwen entity validation,
    required-Qwen phase, and inference data-flow tests.
- Full `v2` suite first run:
  - `474 passed, 2 skipped, 1 failed`;
  - the lone failure was the unrelated, timing-sensitive
    `test_rxnorm_dictionary_and_evidence_are_byte_reproducible`, where two ZIP
    fixtures received different archive hashes across a ZIP timestamp boundary.
- Immediate isolated rerun of that test:
  - `1 passed`.

## Final verification

- Full `v2` unit suite rerun: `475 passed, 2 skipped`.
- `python -m compileall -q clinical_nlp_lab tests`: exit code 0.
- `git diff --check`: exit code 0.
