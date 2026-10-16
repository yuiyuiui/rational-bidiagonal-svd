# 0.1.1 readiness evidence

| Gate | Status | Evidence |
|---|---|---|
| Scope | READY | Existing numerical contract and mathematical assumptions preserved |
| Reliability | READY | Independent reference checks; all 27 tests passed |
| Performance | READY | Paired source hashes, raw samples, profiles, time and memory in bench/ |
| Usability/distribution | READY | Fresh isolated wheel consumer outside checkout; full suite passed; wheel rebuilt from sdist; Twine passed |
| Community/credit | READY | License, real contributor metadata, citation, contribution guidance |
| Hosted checks | NEEDS_WORK | Actual GitHub checks reported separately; configuration is not a passing run |
| DOI / software paper | NOT_APPLICABLE | Neither requested nor registered |

Local validation: macOS arm64, CPython 3.13.3. Configured CI broadens the
platform matrix only when its actual jobs pass. No theorem or proof certification
is inferred from these software checks. No timing target weakened accuracy.
