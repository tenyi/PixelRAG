"""PDF directory source."""

from pathlib import Path
from typing import Iterator

from .base import Document, Source


class PDFSource(Source):
    def __init__(self, path: str, **kwargs):
        self.path = Path(path)
        self._files = sorted(self.path.glob("**/*.pdf"))

    def __iter__(self) -> Iterator[Document]:
        for pdf in self._files:
            yield Document(id=pdf.stem, path=str(pdf), metadata={"type": "pdf"})

    def __len__(self) -> int:
        return len(self._files)
