from __future__ import annotations
import json
from typing import Any
from datetime import datetime, date

from dataclasses import dataclass, field


@dataclass
class FeedItem:
    id: str  # unique id, namespaced by module
    # if it has one, parent entity (e.g. scrobble -> album, or episode -> tv show)
    title: str  # name of entry, track, episode name, or 'Episode {}'
    ftype: str  # scrobble, episode, movie, book
    when: datetime  # when I finished this
    creator: str | None = None  # artist, or person who created this
    # any additional data to attach to this
    data: dict[str, Any] = field(default_factory=dict)
    release_date: date | None = None  # when this entry was released
    part: int | None = None  # e.g. season
    subpart: int | None = None  # e.g. episode, or track number
    # if these are episodes/parts, a collection under which to group these
    collection: str | None = None
    # parent_id: Optional[str] = None
    subtitle: str | None = None  # show name, or album name (for scrobble)
    url: str | None = None
    image_url: str | None = None
    # e.g. 'blur', for passing flags for displaying client side
    flags: list[str] = field(default_factory=list)
    score: float | None = None  # normalized to out of 10

    def check(self) -> None:
        """
        Make sure there are no empty values which should be nulls and do some bounds checking
        """
        if self.image_url is not None and self.image_url.strip() == "":
            self.image_url = None
        if self.url is not None and self.url.strip() == "":
            self.url = None
        if self.score is not None and not (0.0 <= self.score <= 10.0):
            raise ValueError(f"Score for {self} is not within 0-10")
        if isinstance(self.when, datetime):
            assert self.when.tzinfo is not None, str(self)

    def blur(self) -> None:
        if self.image_url is not None:
            self.flags.append("i_blur")

    def to_json(self) -> str:
        return json.dumps(
            {
                "id": self.id,
                "title": self.title,
                "ftype": self.ftype,
                "when": int(self.when.timestamp()),
                "creator": self.creator,
                "data": self.data if self.data else {},
                "release_date": (
                    str(self.release_date) if self.release_date is not None else None
                ),
                "part": self.part,
                "subpart": self.subpart,
                "collection": self.collection,
                "subtitle": self.subtitle,
                "url": self.url,
                "image_url": self.image_url,
                "flags": self.flags if self.flags else [],
                "score": float(self.score) if self.score else None,
            },
            separators=(",", ":"),
        )
