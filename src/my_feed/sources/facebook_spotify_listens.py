"""
When I had my spotify connected to Facebook, it would send action events
every time I listened to a song

This parses that info from the facebook GDPR export, using
https://github.com/purarue/HPI/blob/master/my/facebook/gdpr.py
"""

import re
from typing import Iterator

from my.facebook.gdpr import events, Action

from .model import FeedItem

listened_regex = re.compile(r".*?listened to ([\w\s]+) by ([\w\s]+) on Spotify.*")


def history() -> Iterator[FeedItem]:
    for e in events():
        if not isinstance(e, Action):
            continue
        match = re.match(listened_regex, e.description)
        if match:
            ts = int(e.dt.timestamp())
            yield FeedItem(
                id=f"facebook_spotify_listen_{ts}",
                ftype="listen",
                title=str(match.group(1)),
                creator=str(match.group(2)),
                when=e.dt,
            )
