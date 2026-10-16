# Contributing

Report bugs with the smallest input, effective result configuration, Python,
compiler and Boost versions, and the full exception. Do not include private data.
Use GitHub issues for support and pull requests for changes.

Install `.[dev]`, run `python -m pytest`, `python -m ruff check src tests bench`,
and build with `python -m build`. Tests need Boost >=1.89 and C++17. Never reduce
proof precision or relax error bounds to pass a timing target. Include independent
residual/value checks for numerical changes and a matched benchmark for performance
claims. Keep diagnostic and proof profiles distinct. A fresh wheel consumer must
exercise the packaged C++ resource. Supported development platforms are POSIX
Linux/macOS; Windows is currently unsupported because the build lock uses fcntl.

No public API removal in a patch release. Changes to precision policy or arithmetic
assumptions require an explicit contract review. Version numbers follow semantic
versioning while retaining the research-software limitations in the README.
