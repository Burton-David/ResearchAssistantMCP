"""research-mcp: an MCP server for research workflows."""

import os

__version__ = "0.1.0"

# Mac-specific guardrail: faiss-cpu and torch (under sentence-transformers)
# each ship their own libomp.dylib. Intel's OpenMP runtime aborts the
# process when it sees two registrations — manifests as macOS "Python
# quit unexpectedly" SIGABRT crashes during the first ingest after both
# libraries are imported. Setting this env before either C extension
# loads tells OpenMP to tolerate the duplicate. The Intel docs admit it
# can mask real bugs, but in our case the duplicate registration IS the
# entire issue.
#
# Set at package import so it's in place before any of our entry points
# (`research-mcp serve`, `repl`, `cli`) reaches a `from research_mcp.index
# import FaissIndex` (which triggers `import faiss`).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
