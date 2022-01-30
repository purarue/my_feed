from typing import Optional, List, Dict, Any
from datetime import datetime, date

from dataclasses import dataclass, field

Tags = List[str]
# datetime, date, or year


@dataclass
class FeedItem:
    id: str  # unique id, namespaced by module
    # if it has one, parent entity (e.g. scrobble -> album, or episode -> tv show)
    title: str  # name of entry, track, episode name, or 'Episode {}'
    ftype: str  # scrobble, episode, movie, book
    when: datetime  # when I finished this
    creator: Optional[str] = None  # artist, or person who created this
    tags: Tags = field(default_factory=list)  # extra information/tags for this item
    # any additional data to attach to this
    data: Dict[str, Any] = field(default_factory=dict)
    release_date: Optional[date] = None  # when this entry was released
    part: Optional[int] = None  # e.g. season
    subpart: Optional[int] = None  # e.g. episode, or track number
    # if these are episodes/parts, a collection under which to group these
    collection: Optional[str] = None
    # parent_id: Optional[str] = None
    subtitle: Optional[str] = None  # show name, or album name (for scrobble)
    url: Optional[str] = None
    image_url: Optional[str] = None
    score: Optional[float] = None  # normalized to out of 10

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
