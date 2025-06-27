#!/usr/bin/env python3

"""
Listens (music), with some manual fixes because of lacking metadata

All of the metadata *should* be correct now because of
https://github.com/purarue/plaintext_playlist_py/blob/master/bin/id3stuff
but some older entries have bad metadata
"""

import os
import json
from pathlib import Path
from typing import Iterator, Tuple, Dict, cast, Optional

from my.listenbrainz.export import history as lb_history, Listen

from .model import FeedItem
from ..log import logger
from .common import click, FeedBackgroundError

# defaults from listenbrainz/media player, when the artist was unknown
# tags are either the artist or the release name
BROKEN_TAGS = {"unknown artist", "<unknown>"}

try:
    from my.config.feed import broken_tags  # type: ignore[import]

    BROKEN_TAGS.update({s.lower() for s in broken_tags})
except ImportError as e:
    logger.warning("Could not import feed configuration", exc_info=e)


def _manual_listen_datafile() -> Path:
    return Path(os.path.join(os.environ["HPIDATA"], "feed_listen_fixes.json"))


Metadata = Tuple[str, str, str]


def _manually_fix_listen(ls: Listen) -> Metadata:
    """Fix broken metadata on listens, and save my responses to a cache file"""

    # load data
    datafile = _manual_listen_datafile()
    data: Dict[str, Metadata] = {}
    if datafile.exists():
        data = cast(Dict[str, Metadata], json.loads(datafile.read_text()))

    # use timestamp to uniquely identify a single fix
    assert ls.listened_at is not None
    ts = str(int(ls.listened_at.timestamp()))
    if ts in data:
        logger.debug(f"Replacing manual listen fix {data[ts]}")
        return data[ts]

    # prompt me to manually type in the correct data
    click().echo(f"broken: {ls}", err=True)
    title = click().prompt("title").strip()
    subtitle = click().prompt("album name").strip()
    creator = click().prompt("artist name").strip()

    new_data = (
        title,
        subtitle,
        creator,
    )
    # write data
    data[ts] = new_data
    datafile.write_text(json.dumps(data, separators=(",", ":")))
    return new_data


def history() -> Iterator[FeedItem]:
    for listen in lb_history():
        if listen.listened_at is None:
            logger.debug(f"ignoring listen with no datetime {listen}")
            continue

        title: str = listen.track_name
        subtitle: Optional[str] = listen.release_name
        creator: str = listen.artist_name
        # some unique filename part like (Album Version (Explicit))
        tag_matches_title_substring = any(
            [b.lower() in listen.track_name.lower() for b in BROKEN_TAGS]
        )
        # if I've marked this as broken
        if (
            tag_matches_title_substring
            or listen.artist_name.lower() in BROKEN_TAGS
            or listen.track_name.lower() in BROKEN_TAGS
            or (
                listen.release_name is not None
                and listen.release_name.lower() in BROKEN_TAGS
            )
        ):
            try:
                title, subtitle, creator = _manually_fix_listen(listen)
            except FeedBackgroundError as e:
                logger.warning(
                    f"Running in the background, cannot prompt for {listen}", exc_info=e
                )

        # it seems this happens when im listening to internet radio through foobar?
        # this does break if I listen to an album name that is literally '?', but I'm
        # okay with that tradeoff
        if (
            subtitle == "?"
            and listen.metadata.get("submission_client") == "Pano Scrobbler"
        ):
            subtitle = None

        ts: int = int(listen.listened_at.timestamp())
        # TODO: attach to album somehow (parent_id/collection)?
        yield FeedItem(
            id=f"listen_{ts}",
            ftype="listen",
            title=title,
            creator=creator,
            subtitle=subtitle,
            when=listen.listened_at,
        )
