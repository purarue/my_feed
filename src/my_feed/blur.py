from __future__ import annotations
import re
import fnmatch

from pathlib import Path
from enum import Enum
from typing import NamedTuple, TextIO

from .sources.model import FeedItem


class Attr(str, Enum):
    ID_FNMATCH = "id"
    TITLE_FNMATCH = "title"
    IMAGE_FNMATCH = "image_url"
    ID_REGEX = "id_regex"
    TITLE_REGEX = "title_regex"
    IMAGE_REGEX = "image_url_regex"

    @classmethod
    def from_str(cls, data: str) -> Attr:
        match data:
            case "id":
                return cls.ID_FNMATCH
            case "title":
                return cls.TITLE_FNMATCH
            case "image_url":
                return cls.IMAGE_FNMATCH
            case "id_regex":
                return cls.ID_REGEX
            case "title_regex":
                return cls.TITLE_REGEX
            case "image_url_regex":
                return cls.IMAGE_REGEX
            case _:
                raise ValueError(f"Unknown attribute: {data}")


class Blur(NamedTuple):
    attr: Attr
    pattern: str

    def __str__(self) -> str:
        return f"{self.attr}: {self.pattern}"

    @classmethod
    def from_str(cls, line: str) -> Blur:
        attr, _, value = line.partition(":")
        val = value.strip()
        if not val:
            raise ValueError(f"Empty value for attribute {attr}, line {line}")
        return cls(Attr.from_str(attr.strip().lower()), val)


class Blurred(NamedTuple):
    items: set[Blur]

    @classmethod
    def parse_blob(cls, f: TextIO, /) -> Blurred:
        return cls({Blur.from_str(line) for line in f if line.strip()})

    @classmethod
    def parse_file(cls, path: Path, /) -> Blurred:
        with path.open("r") as f:
            return cls.parse_blob(f)

    def should_be_blurred(self, *, feed_item: FeedItem) -> bool:
        for blur in self.items:
            if blur.attr == Attr.ID_FNMATCH:
                if fnmatch.fnmatch(feed_item.id, blur.pattern):
                    return True
            elif blur.attr == Attr.ID_REGEX:
                if re.search(blur.pattern, feed_item.id):
                    return True
            elif blur.attr == Attr.TITLE_FNMATCH:
                if fnmatch.fnmatch(feed_item.title, blur.pattern):
                    return True
            elif blur.attr == Attr.TITLE_REGEX:
                if re.search(blur.pattern, feed_item.title):
                    return True
            elif blur.attr in (Attr.IMAGE_FNMATCH, Attr.IMAGE_REGEX):
                if not feed_item.image_url:
                    continue
                if blur.attr == Attr.IMAGE_REGEX:
                    if re.search(blur.pattern, feed_item.image_url):
                        return True
                else:
                    if fnmatch.fnmatch(feed_item.image_url, blur.pattern):
                        return True
        return False
