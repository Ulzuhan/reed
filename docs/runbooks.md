# Operations runbooks

These procedures assume the supported Reed v0.3 topology: one Reed API process and one logical
corpus. Replace `uv run` with the installed `reed` command when using a wheel.

## Backup and verify

1. Stop Reed and wait for the process to exit. A live copy can split SQLite, uploads and Qdrant
   across different moments and is not a supported backup.
2. Keep the archive outside `REED_DATA_DIR`.
3. Create and immediately verify it:

   ```bash
   uv run reed backup create ../reed-backup-20260804.tar.gz
   uv run reed backup verify ../reed-backup-20260804.tar.gz
   ```

4. Store the archive with access controls appropriate for the source documents and API data.

The manifest records schema, Reed version, creation time and SHA-256 for every file. Creation
refuses symlinks and existing destinations. Verification rejects traversal, links, devices,
unexpected members, missing files and checksum mismatches.

Embedded Qdrant lives below `REED_DATA_DIR` and is included. When `REED_QDRANT_URL` points to a
server, the Reed archive contains the registry and retained originals but **not** remote vectors.
With Reed stopped, take a Qdrant collection/storage snapshot according to that deployment's own
procedure and retain it beside the Reed archive.

## Restore

1. Keep Reed stopped.
2. Verify the archive on the target host:

   ```bash
   uv run reed backup verify ../reed-backup-20260804.tar.gz
   ```

3. Point `REED_DATA_DIR` at a path that does not exist or is completely empty.
4. Restore:

   ```bash
   REED_DATA_DIR=./restored-data \
   uv run reed backup restore ../reed-backup-20260804.tar.gz
   ```

5. If Qdrant was remote, restore its coordinated snapshot before starting Reed.
6. Start Reed, require `GET /ready` to return `200`, inspect `reed index status`, list documents,
   and ask a known-answer and a known-negative question.

Restore verifies the archive first, extracts into an adjacent scratch directory and atomically
renames it into place. It refuses a non-empty target rather than overwriting recoverable data.

## Change embedding or chunking configuration

1. Stop the server and create a verified backup.
2. Change the model, immutable identity override, prefixes or chunking settings.
3. Ensure the model is locally available, then build the candidate:

   ```bash
   uv run reed index reindex
   ```

4. Inspect generations:

   ```bash
   uv run reed index status
   ```

5. Start Reed, wait for readiness and run a retrieval-only evaluation against the target corpus.

Reindex validates retained originals against registry hashes, builds a separate physical
collection, verifies committed counts and activates it in one SQLite transaction. If anything
fails, the old active generation is untouched and the candidate collection is removed.

## Roll back an index

1. Stop Reed.
2. Restore the exact configuration used by the previous generation—model tag/digest, prefixes,
   chunk size/overlap and sparse configuration.
3. Confirm the previous physical collection still exists with `reed index status`.
4. Activate it:

   ```bash
   uv run reed index rollback
   ```

5. Start, verify readiness and run known-answer checks.

Rollback refuses incompatible fingerprints or a missing physical collection. It retains the
replaced active generation as `previous`, so a mistaken rollback can be reversed after restoring
that generation's configuration.

## Clean old generations

Only after the active generation has passed real-corpus evaluation and the rollback window has
expired:

```bash
uv run reed index cleanup --keep 2
```

The active generation is never deleted. `--keep 2` retains active plus the newest previous
generation and removes older/failed physical collections and registry rows.

## Recover after interruption

- Durable `queued` rows resume automatically on the next start.
- Rows interrupted in `parsing`, `embedding` or `indexing` become `error`; upload the same original
  again to use the hash-based retry path.
- Uncommitted Qdrant points are filtered from retrieval and scheduled for cleanup before searches.
- A collection fingerprint mismatch is not transient. Run a safe reindex or restore the matching
  configuration; restarting repeatedly cannot make incompatible vectors valid.
- `/ready` returning `503` for a transient provider/store outage includes `Retry-After`; Reed
  retries vector initialization with bounded backoff.

## Monitor and diagnose

Use liveness for process supervision and readiness for traffic admission:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/ready
curl -fsS -H 'X-API-Key: …' http://127.0.0.1:8000/metrics
```

Metrics include request outcomes, upload rejections, ingestion stage totals, queue depth,
retrieval latency and abstentions. Alerts should distinguish sustained queue saturation, elevated
retrieval latency, provider unavailability and rising ingestion errors. Logs keep detailed
dependency errors server-side; public health responses intentionally expose stable states rather
than credentials or internal endpoints.

For Compose deployments, also inspect:

```bash
docker compose ps
docker compose logs --no-color api qdrant
```

Do not start a second Reed API process against the same data directory as a recovery technique.
Single-node means one process owns the registry, queue and embedded-store lock.
