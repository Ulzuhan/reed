"""Build a tiny text-bearing PDF, so PDF tests need no binary fixture.

pypdf can only write blank pages, and committing a generated PDF would hide
what the tests actually exercise. This emits the minimal structure a real
extractor accepts: one content stream per page with a single Tj operator.
"""

from __future__ import annotations

from pathlib import Path


def write_pdf(path: Path, pages: list[str]) -> Path:
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)  # 1-based object number

    catalog_num = add(b"")  # placeholder, filled once Pages is known
    pages_num = add(b"")
    font_num = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_nums: list[int] = []
    for text in pages:
        escaped = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1")
        content_num = add(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
        page_nums.append(
            add(
                b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 595 842] "
                b"/Contents %d 0 R /Resources << /Font << /F1 %d 0 R >> >> >>"
                % (pages_num, content_num, font_num)
            )
        )

    kids = b" ".join(b"%d 0 R" % n for n in page_nums)
    objects[pages_num - 1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (kids, len(page_nums))
    objects[catalog_num - 1] = b"<< /Type /Catalog /Pages %d 0 R >>" % pages_num

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (number, body)

    xref_at = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        catalog_num,
        xref_at,
    )

    path.write_bytes(bytes(out))
    return path
