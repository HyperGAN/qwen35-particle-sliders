# Experimental GitHub releases

Upload the reviewed model bundle to a draft GitHub prerelease. Assign a new
version tag and set the draft's target to the full source commit already in
`main`. Include the adapter config, model license and notice, weights, and
release archive. The `SHA256SUMS` attachment must list every other attachment.

Open a pull request adding or updating `release.json` at the root of the
repository.

| Field | Value |
| --- | --- |
| `tag` | New version tag, such as `v0.2.0` |
| `release_id` | Numeric GitHub draft release ID |
| `source_commit` | Full Git commit used to create the reviewed bundle |
| `files` | Map of exact attachment filenames to objects containing `bytes` and `sha256`, including `SHA256SUMS` |

From a checkout that includes the reviewed request, run
`python scripts/publish_release.py audit` with GitHub CLI access to the draft.
The audit checks the source and compares the staged files with the reviewed
list and the manifest. Keep the draft fixed while its release request is open.

Merging the request to `main` rechecks the draft, publishes it as an
experimental prerelease, and verifies the public release and exact source tag.
Rerunning the job accepts the same published prerelease after verification.

Delta League submission continues to use the evaluation criteria in the
[project README](../README.md).
