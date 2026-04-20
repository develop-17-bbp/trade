"""Learning Mesh — Phase 4.5 package.

  - experience.py:        Pydantic Experience extending Decision   (4.5a §2.1)
  - credit_assigner.py:   rolling-500 regression credit weights    (4.5a §3)
  - meta_coordinator.py:  supervisor that fans out experience      (4.5a §6)
  - signal_bus.py:        cross-learner pub/sub on model.signal.*  (4.5b §4)
  - coevolution.py:       Genetic↔RL, LoRA↔Calibrator transfers    (4.5b §5)
  - safety.py:            delta caps + quarantine + authority gate (4.5c §7)

All imports are deferred so the package can be imported on a minimal
environment (no sklearn, no redis) and still expose the module surface.
"""
