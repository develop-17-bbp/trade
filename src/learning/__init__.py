"""Learning Mesh — Phase 4.5 package.

  - experience.py:        Pydantic Experience extending Decision (§2.1)
  - credit_assigner.py:   rolling-500 regression credit weights (§3.2)
  - meta_coordinator.py:  supervisor that fans out enriched experience (§6)

All imports are deferred so the package can be imported on a minimal
environment (no sklearn, no redis) and still expose the module surface.
"""
