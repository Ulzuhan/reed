"""Version of the pipeline that turns a stored file into embedding input.

Bump this whenever the same file would now produce different text to embed: a
parser that extracts differently or names sections differently, a change to
``split_sections``, or a change to how ``_embedding_text`` composes the title
and section. The version is part of the collection fingerprint, so a bump makes
every existing index mismatch, and `reed index reindex` is the documented
remedy — the same path a model or chunking change already takes.

Do not bump for anything that cannot change the embedded text: refactors,
logging, error messages, or changes to displayed and cited output, which comes
from a separate field.

History:

1. Section-aware chunking with embedding input composed as
   ``Title: <filename>\\nSection: <section>\\n\\n<text>``, over the PDF, DOCX,
   HTML, Markdown and plain-text parsers. Reed 0.3 shipped this scheme without
   recording it; Reed 0.2 embedded the raw chunk text and is therefore *not*
   this version. Adding a format does not change what an already-ingestible
   file produces, so a new parser alone does not bump this number.
"""

from __future__ import annotations

EXTRACTION_VERSION = 1
