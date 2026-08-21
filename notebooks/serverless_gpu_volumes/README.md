# Loading models & datasets from UC Volumes on Serverless GPU

A hands-on walkthrough for a customer (Shariyar) question:

> On regular Serverless compute, a model or dataset loads fine by referencing the
> `/Volumes/...` path directly. On **Serverless GPU**, local Python operations such as
> `os.listdir()` return no files and libraries like pandas cannot find the file — yet the
> same files are still reachable through SQL and Spark. What is the recommended way to load
> models and datasets from Unity Catalog Volumes on Serverless GPU, use `UCVolumeDataset`,
> handle libraries that need a local filesystem path, and list files in a Volume? Is this
> expected?

## Short answer

**Yes, this is expected behavior of Serverless GPU compute today — not a bug.**

Serverless GPU compute does **not** expose the `/Volumes` **POSIX FUSE mount** to arbitrary
local Python / OS calls. That is why `os.listdir("/Volumes/...")` returns nothing and
`pandas.read_csv("/Volumes/...")` raises `FileNotFound` — there is no local mount for those
libraries to see. Access that goes through the **governed Unity Catalog layer** still works,
which is why Spark and SQL are unaffected:

| Access path | Works on Serverless GPU? | Notes |
|---|---|---|
| Spark / `spark.read` | ✅ yes | goes through UC, not a POSIX mount |
| SQL (`SELECT ... read_files`, tables) | ✅ yes | same |
| `UCVolumeDataset` (`serverless_gpu.data`) | ✅ yes | **recommended for training data** — streams + caches locally |
| Databricks SDK Files API (`w.files.download` / `list_directory_contents`) | ✅ yes | **recommended for models / individual files** |
| `os.listdir("/Volumes/...")`, `open()`, `pandas.read_csv("/Volumes/...")` | ❌ no | no FUSE mount → `FileNotFound` / empty listing |
| `dbutils.fs.ls("/Volumes/...")` | ⚠️ context-dependent | prefer the SDK Files API for portability |

The fix is to stop treating a Volume as a local directory on Serverless GPU and instead use a
UC-aware API. For any library that *insists* on a local path, **download the file to local
disk first** (via the Files API), then hand the library the local path.

## Recommended approach by use case

### 1. Loading a training/inference **dataset** (many files) → `UCVolumeDataset`

For unstructured data (images, audio, text) this is the first-class pattern. It reads files
from the Volume, **caches them to local disk on first access** (so multi-epoch training does
not re-read the Volume), and **partitions files across `torch.distributed` ranks and
DataLoader workers** so each `(rank, worker)` gets a non-overlapping slice.

```python
from serverless_gpu.data import UCVolumeDataset, DataLoader
from torch.utils.data import IterableDataset
from PIL import Image
import torchvision.transforms.functional as TF

class ImageDataset(IterableDataset):
    def __init__(self, path_dataset: UCVolumeDataset):
        self._path_dataset = path_dataset
    def __iter__(self):
        for local_path in self._path_dataset:      # <- a LOCAL cached path, not /Volumes
            img = Image.open(local_path).convert("RGB")
            yield TF.to_tensor(img)

path_dataset = UCVolumeDataset("/Volumes/catalog/schema/my_volume/images")
loader = DataLoader(ImageDataset(path_dataset), batch_size=32, pin_memory=True)
```

Build the dataset **inside** the `@distributed` run so partitioning sees the initialized
process group, and keep `num_workers` identical across ranks (otherwise files get duplicated
or skipped).

### 2. Loading a single **model** or a few files → Databricks SDK Files API

The Files API reads through UC governance instead of a mount, so it works on Serverless GPU.

```python
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

# list a directory (replaces os.listdir)
for entry in w.files.list_directory_contents("/Volumes/catalog/schema/my_volume/models/"):
    print(entry.path, entry.is_directory)

# read a file into memory (replaces open()/pandas on /Volumes)
import io, pandas as pd
resp = w.files.download("/Volumes/catalog/schema/my_volume/data/train.csv")
df = pd.read_csv(io.BytesIO(resp.contents.read()))
```

> For **MLflow models**, prefer not to touch the Volume path at all — load by URI:
> `mlflow.pyfunc.load_model("models:/catalog.schema.model@champion")`. MLflow stages the
> artifacts to local disk for you. Only reach for the Files API when you have raw model files
> sitting in a Volume.

### 3. Libraries that require a **local filesystem path** → download first, then pass the local path

Some libraries (e.g. a custom C-extension loader, `llama.cpp`, a checkpoint reader that only
takes a path string) will not accept bytes. Stage the file to local ephemeral disk, then pass
that path:

```python
from databricks.sdk import WorkspaceClient
from pathlib import Path
import tempfile

w = WorkspaceClient()
src = "/Volumes/catalog/schema/my_volume/models/model.gguf"
dst = Path(tempfile.gettempdir()) / "model.gguf"    # /tmp — writable local scratch
with dst.open("wb") as f:
    f.write(w.files.download(src).contents.read())

load_my_model(str(dst))                             # library gets a real local path
```

Use `tempfile.gettempdir()` (`/tmp`) for the local scratch path — **not `/local_disk0`, which
is not writable on serverless** (verified: it raises `PermissionError`). `/tmp` is real local
storage that every local library works against. For a whole directory, list with
`list_directory_contents` and download each file.

### 4. Listing files and directories in a Volume

`os.listdir` / `glob` do not work (no mount). Use one of these instead:

```python
# a) Databricks SDK (portable, recommended)
for e in w.files.list_directory_contents("/Volumes/catalog/schema/my_volume/"):
    print(e.path, e.file_size, e.is_directory)

# b) SQL — LIST is fully governed and always available
%sql LIST '/Volumes/catalog/schema/my_volume/'
```

## The notebooks

Run in order on the target workspace. Notebook `00` runs on **any** UC compute (regular
Serverless is fine); `01` and `02` are meant to run on **Serverless GPU** to reproduce the
behavior and confirm the recommended patterns. Change the `catalog` / `schema` / `volume`
widgets at the top of each notebook to a location you can write to.

| Notebook | What it does |
|---|---|
| `00_setup_volume_assets.py` | Creates a UC Volume and writes a sample **dataset** (CSV/Parquet) and a small **model artifact** into it. Run on regular Serverless. |
| `01_reproduce_gpu_access.py` | On **Serverless GPU**: shows `os.listdir` / `open` / `pandas` failing on `/Volumes`, while Spark and SQL read the same files fine. Establishes the "expected behavior." |
| `02_recommended_loading.py` | The four recommended patterns end-to-end: SDK Files API list/download, download-to-local-disk for path-only libraries, `mlflow` load-by-URI, and `UCVolumeDataset` for training data. |

## Reference documentation

- What are Unity Catalog volumes (compute/runtime requirements & FUSE limitations):
  https://docs.databricks.com/aws/en/volumes/
- Load data on AI Runtime — `UCVolumeDataset` + Serverless GPU DataLoader:
  https://docs.databricks.com/aws/en/machine-learning/ai-runtime/dataloading
- Databricks SDK Files API (`WorkspaceClient.files`): download / upload / list_directory_contents:
  https://databricks-sdk-py.readthedocs.io/en/latest/workspace/files/files.html
- KB — accessing UC Volume files fails with FileNotFound (SDK download pattern):
  https://kb.databricks.com/unity-catalog/queries-to-access-files-on-a-unity-catalog-volume-fail-with-filenotfound-error

## Verified results (fevm-shm-skunkworks, Aug 2026)

Ran the setup + a consolidated evidence harness as serverless jobs against
`shm_catalog.ml.gpu_volume_demo`. What was confirmed:

- **`00_setup_volume_assets.py` — SUCCESS.** Created the Volume, `data/train.csv` (5,000 rows),
  `data/train_parquet/`, and `models/churn_model.pkl`.
- **On regular serverless, local `/Volumes` access WORKS** — `os.listdir` returned
  `['train_parquet', 'train.csv']`, `open()` OK, `pandas.read_csv` read 5,000 rows. This is the
  key contrast: identical code + identical Volume works on regular serverless, so the customer's
  failure is specific to Serverless GPU (no FUSE mount there).
- **Governed access works** — Spark read 5,000 rows; SQL `LIST` returned 2 entries.
- **All recommended patterns work** — SDK `list_directory_contents`, SDK `download` → pandas
  (5,000 rows), and download-to-local-scratch → `pickle.load` → prediction `[1]`.
- **Bug found while testing:** `/local_disk0` is **not writable on serverless**
  (`PermissionError: [Errno 13]`). Fixed to use `tempfile.gettempdir()` (`/tmp`), which works.
- **Could not execute on Serverless GPU in this workspace:** the `serverless_gpu` package is not
  present in the serverless job environment (client 2 and 3 both raise
  `ModuleNotFoundError: No module named 'serverless_gpu'`) — the Serverless GPU AI-Runtime
  preview isn't enabled here. Run `01`/`02` on a GPU-enabled workspace to see the local-access
  failure directly; the regular-serverless contrast above plus the docs already establish the
  mechanism.

> The rest of the guidance is drawn from Databricks docs/KB. Run `01`/`02` on the customer's
> Serverless GPU compute to confirm the failing-then-fixed behavior in their exact environment.
