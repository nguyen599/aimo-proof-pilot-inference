# IMO 2025 final artifacts

Run: `imo2025-full-p36-r4-p2-sft750-handoff-priority-2node-20260718T184959Z`

- `grader_input/records.jsonl`: six full selected-proof records written by `write_grader_input_records`, ordered by problem ID.
- `submission.csv`: six final answers, IDs 1-6, with no blank or duplicate IDs.
- `manifest.json`: exact inference configuration and model/runtime metadata.
- `SHA256SUMS.txt`: integrity hashes for the grader input, submission, and manifest.

Both distributed ranks exited with status 0. The full runtime and LLM-call logs remain on the NII shared filesystem and are intentionally not included in this commit.
