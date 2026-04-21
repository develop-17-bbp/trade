# LightGBM CUDA tree learner — optional 2–3x training speedup

`pip install lightgbm` ships a prebuilt wheel with the **OpenCL** GPU backend
(`device='gpu'`) but NOT the native **CUDA** backend. The OpenCL path is what
`src/ml/gpu.py` auto-detects; it works and is the default on this machine.

Native CUDA (`device='cuda'`) is faster — 2–3x on large training sets, more
VRAM-efficient. Enabling it requires rebuilding LightGBM from source.

## When to bother

Worth the build effort if **any** of these:
- Retrain cycle runs ≥ every 30 minutes (continuous_adapt tier)
- Training data exceeds ~500k rows (multi-TF + multi-asset Optuna pass)
- You want to trial CUDA-only LightGBM features (e.g. large categoricals)

Skip if training completes in < 2 minutes today — the speedup doesn't pay back
the build fiddling.

## Prerequisites on the GPU box

- Visual Studio 2022 Build Tools with C++ workload
- CUDA Toolkit ≥ 11.8 (check `nvcc --version`)
- CMake ≥ 3.28
- Git

## Build steps

```powershell
# 1. Pick a fresh directory outside the repo
cd C:\Users\admin
git clone --recursive https://github.com/microsoft/LightGBM.git
cd LightGBM

# 2. Configure + build with CUDA enabled
mkdir build; cd build
cmake -A x64 -DUSE_CUDA=1 ..
cmake --build . --target ALL_BUILD --config Release

# 3. Install the Python wrapper from the just-built binaries
cd ..\python-package
python setup.py install --precompile

# 4. Verify
python -c "
import lightgbm as lgb, numpy as np
X = np.random.rand(1024, 8).astype('float32')
y = (X[:,0] > 0.5).astype(int)
ds = lgb.Dataset(X, label=y)
try:
    lgb.train({'objective':'binary','device':'cuda','verbose':-1}, ds, num_boost_round=5)
    print('CUDA tree learner OK')
except Exception as e:
    print('CUDA failed:', e)
"
```

## After it works

Tell the ACT trainers to use `cuda` instead of `gpu`:

```cmd
setx ACT_LGBM_DEVICE cuda
```

Then open a new shell and re-run `START_ALL.ps1`. `src/ml/gpu.py` honours the env
override without an auto-probe.

## Rollback

If CUDA build breaks something, reinstall the prebuilt wheel:

```powershell
pip install --force-reinstall lightgbm
setx ACT_LGBM_DEVICE gpu   # back to OpenCL
```

No ACT code needs to change — the device-selection helper keeps working.
