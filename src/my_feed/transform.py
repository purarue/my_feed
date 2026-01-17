"""
Allows custom transformations to be applied to feed items

After creating a list called TRANSFORMS in my.config.feed, you can, you can
wrap some source with transform(source), e.g.:

def _fix_artist_name(item: FeedItem) -> Optional[FeedItem]:
    if item.ftype != 'mpv':
        return item

    if item.creator = "something I want to ignore":
        return None  # drop item

    # e.g. fix a misspelled artist name or one that's slightly
    # different from the one in the musicbrainz database
    if item.creator = "...":
        data = dataclasses.asdict(item)
        data['creator'] = "something"
        return FeedItem(**data)
    return item  # otherwise return as is


TRANSFORMS = [_fix_artist_name]

def sources() -> Iterator[Callable[[], Iterator["FeedItem"]]]:
    from my_feed.transform import transform

    yield transform(mpv.history)
"""

import functools
from typing import Optional
from collections.abc import Callable, Iterator

from .sources.model import FeedItem
from .log import logger


# if this returns nothing, the item is dropped
# if you don't want to transform the item, yield the item itself
TransformFunction = Callable[[FeedItem], Optional[FeedItem]]

TRANSFORMS: list[TransformFunction] = []

try:
    from my.config.feed import TRANSFORMS as MY_TRANSFORMS  # type: ignore[import]

    assert isinstance(MY_TRANSFORMS, list)
    for tr in MY_TRANSFORMS:
        assert callable(tr)
    TRANSFORMS.extend(MY_TRANSFORMS)

except ImportError as e:
    logger.debug(e, exc_info=True)


def transform(
    feed: Callable[[], Iterator[FeedItem]],
    transforms: list[TransformFunction] = TRANSFORMS,
) -> Callable[[], Iterator[FeedItem]]:
    """
    Receives a callable source as input, and wraps it, returning a callable
    This is the entrypoint to this module
    """

    @functools.wraps(feed)
    def _tr() -> Iterator[FeedItem]:
        yield from _transform(feed(), transforms)

    return _tr


def _transform(
    feed: Iterator[FeedItem], transforms: list[TransformFunction] = TRANSFORMS
) -> Iterator[FeedItem]:
    for item in feed:
        # update scope with item
        updated: FeedItem = item
        for transform in transforms:
            if transformed := transform(item):
                updated = transformed
            else:
                # drop item, was none
                break
        else:
            # we hit this block if we exhausted the transforms without breaking
            # so we yield the item here
            yield updated
