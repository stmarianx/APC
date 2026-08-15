# APC portability and release policy

Status: development policy
Applies to: source, datasets, checkpoints, player profiles and deployment

## Deployment objective

APC must be reproducible on this 8 GB development laptop and transferable to a
stronger Windows or Linux training machine without copying a Python
installation or relying on machine-specific absolute paths.

The portable unit is:

1. versioned source code and schemas;
2. small, auditable manifests and model cards;
3. a declared dependency profile;
4. externally stored, fingerprinted datasets and checkpoints;
5. a machine-readable run configuration and evaluation report.

Downloaded dependencies and model caches are never part of the source
repository. Large datasets and trained checkpoints move through fingerprinted
artifact bundles instead of being copied as an unverified runtime tree.

## Runtime profiles

| Profile | Intended machine | Contents |
| --- | --- | --- |
| `core` | 8 GB laptop | coaching engine, hand review and exact tests |
| `vision` | 8 GB laptop | core plus Pillow/NumPy and APC perception tests |
| `transcription` | temporary research use | Faster-Whisper and its downloaded model cache |
| `all` | stronger training machine | every development dependency |

Create an isolated environment with:

```powershell
.\scripts\setup_apc.ps1 -Profile vision
```

The default development path is CPU-first and uses bounded batches. Training
runs must record peak memory, elapsed time, dependency versions, input manifest
fingerprints and checkpoint fingerprints. A run that terminates from memory
pressure is evidence, not a promotable checkpoint.

## Artifact classes

### Versioned in Git

- source, tests, schemas and deterministic generators;
- dataset manifests without private local paths;
- aggregate evaluation reports and model cards;
- small synthetic fixtures that are explicitly approved for publication;
- documentation and reproducible dependency declarations.

### Stored outside Git

- downloaded third-party models such as Faster-Whisper `model.bin`;
- raw/processed visible-table frames and videos;
- APC checkpoints, optimizer state and run caches;
- course media and any redistribution-restricted source material.

External artifacts must be copied with a manifest containing relative path,
byte length, SHA-256 digest, producer/version and source dataset fingerprint.
This lets another laptop verify an artifact without putting it in Git.

Create and verify that transfer manifest with:

```powershell
python -m apc.tools.artifact_manifest create . .\transfer\manifest.json `
  .\apc\checkpoints\run-v1 .\apc\runs\run-v1 `
  --producer apc-run-v1 --artifact-class checkpoint_bundle `
  --source-fingerprint dataset=REPLACE_WITH_DATASET_SHA256

python -m apc.tools.artifact_manifest verify . .\transfer\manifest.json
```

## Publication lifecycle

During development the source repository may remain publicly visible. Before a
final release or deployment, APC must pass a release review that:

1. chooses an explicit code license or makes the repository private;
2. verifies rights to publish course-derived text, images and other assets;
3. removes downloaded/vendor binaries and rebuildable environments;
4. publishes only approved checkpoints with their model cards and licenses;
5. tests a clean clone on a second machine.

Rewriting already-published Git history or changing GitHub visibility is a
separate, explicit operation. `.gitignore` prevents future additions but does
not remove files that are already present in a commit.
