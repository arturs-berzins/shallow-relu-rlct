# Provenance

Arturs Berzins developed the underlying solver and shared-hub certificate in the `research_automation_relu` research collection. This package extracts six modules from its `topics/shallow-relu` directory at revision `c881b808f1d72fadccedd5dca795d4224718528a`. The extraction adapts imports for installation as a standalone library.

| Material | Source in the research collection |
|---|---|
| Symbolic solver | `rlct_pipeline.py`, `rlct_decision.py`, `rlct_certificates.py`, `rlct_ideal_reduction.py`, `rlct_monomial.py` |
| Shared-hub checker | `shared_hub_certificate.py` |
| Biased fold | `verify_rlct_decision.py` and `verify_biased_pair_reduction.py` |
| Benchmark family | `verify_shared_hub_edge_core.py` and `verify_shared_hub_criterion.py` |
| Mathematical derivations | `finite_data_rlct_decision.md`, `shared_hub_geometric_criterion.md`, and `shared_hub_edge_core_rlct.md` |

The scalar rank entry point, packaging, and regression checks were added for this release. The MIT license applies to this repository.
