#!/usr/bin/env python3
"""Check dependencies for gPAC and Tensorpac comparison."""

try:
    import gpac
    print(f"gpac is available, version: {getattr(gpac, '__version__', 'unknown')}")
except ImportError:
    print("gpac is NOT available")

try:
    import tensorpac
    print(f"tensorpac is available, version: {getattr(tensorpac, '__version__', 'unknown')}")
except ImportError:
    print("tensorpac is NOT available")

try:
    import mngs
    print(f"mngs is available")
except ImportError:
    print("mngs is NOT available")