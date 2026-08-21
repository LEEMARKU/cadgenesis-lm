# Dataset Validation Report

**Source:** `data/curriculum/train.jsonl (first 2000)`

**Total records:** 2000
**Valid records:** 2000
**Pass rate:** 100.00%
**Duplicate records (MinHash):** 311

## Check Summary

- cad_seq_length:pass: 2000
- cad_tokens_are_strings:pass: 2000
- cad_tokens_recognized:fail: 929
- cad_tokens_recognized:pass: 1071
- cad_tokens_valid:pass: 2000
- metadata_is_dict:pass: 2000
- text_present:pass: 2000
- type_is_string:pass: 2000

## Duplicate Examples

- use the slot tool to build sketch_rect (96.0 mm) slot
- describe the geometry of: CYLINDER NUM_039 EXTRUDE BOX
- use the hole tool to build sketch_rect (13.0 mm) slot
- create sketch_rect (30.0 mm) slot
- describe the geometry of: THREAD NUM_027 NUM_14 CYLINDER

## Statistics

- Records: 2000
- Tokens total: 11058
- Vocabulary coverage: 9.95% (129/1296)
- Avg text length (words): 8.53
- Avg CAD length (tokens): 5.53

### Records by type

- constraint: 235
- error2correction: 208
- geometry2description: 210
- nl2ir: 214
- nl2program: 245
- parameter: 218
- planning: 210
- program2explanation: 205
- tool: 255

### Top tokens

- EXTRUDE: 1695
- SKETCH_RECT: 613
- HOLE: 468
- CYLINDER: 451
- SLOT: 448
- BOX: 433
- NUM_10: 187
- SPHERE: 179
- COUNTERBORE: 178
- BASE: 168
- PEG: 168
- FIXTURE: 165
- SPACING: 165
- PATTERN: 165
- RECT: 153
- BRACKET: 152
- MOUNT: 152
- DOWEL: 149
- PART: 142
- THREAD: 138
