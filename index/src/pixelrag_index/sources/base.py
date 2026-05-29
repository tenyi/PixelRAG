"""Base class for document sources."""

from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class Document:
    id: str
    url: str | None = None
    path: str | None = None
    metadata: dict = field(default_factory=dict)


class Source:
    def __iter__(self) -> Iterator[Document]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
