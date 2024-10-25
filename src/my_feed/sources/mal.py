"""
Gets anime/manga from using
https://github.com/purarue/malexport
"""

import os
from datetime import datetime, timezone
from typing import Iterator, Optional, Union

import my.mal.export as mal

from .model import FeedItem
from ..log import logger


def _image_url(data: Union[mal.AnimeData, mal.MangaData]) -> Optional[str]:
    if data.APIList is None:
        # TODO: fetch from dbsentinel?
        # https://dbsentinel.sean.fish/
        return None
    api_images = data.APIList.main_picture
    for k in (
        "medium",
        "large",
    ):
        if img_url := api_images.get(k):
            return img_url
    return None


def _completed_datetime(
    data: Union[mal.AnimeData, mal.MangaData]
) -> Optional[datetime]:
    dt: Optional[datetime] = None
    total_count: int
    watched: int
    if isinstance(data, mal.AnimeData):
        total_count = data.XMLData.episodes
        watched = data.XMLData.watched_episodes
    else:
        total_count = data.XMLData.chapters
        watched = data.XMLData.read_chapters

    # if there's only one episode, find the first time I watched this
    if watched > 0 and len(data.history) > 0:
        # its sorted from newest to oldest, so iterate from the beginning
        # this is the datetime when I completed the last epsisode the first
        # time (could be multiple times because of rewatches)
        completed_last_ep_at = [
            ep for ep in reversed(data.history) if ep.number == total_count
        ]
        if completed_last_ep_at:
            dt = completed_last_ep_at[0].at
    if dt is None:
        # use finish date
        if data.XMLData.finish_date is not None:
            dt = datetime.combine(
                data.XMLData.finish_date, datetime.min.time(), tzinfo=timezone.utc
            )
        elif len(data.history) > 0:
            # use history entry
            dt = data.history[0].at

    return dt


def _anime() -> Iterator[FeedItem]:
    for an in mal.anime():
        if an.username != os.environ["MAL_USERNAME"]:
            continue

        if an.APIList is None:
            logger.warning(f"No API info for anime {an.XMLData.id}")
            continue

        url = f"https://myanimelist.net/anime/{an.id}"
        score = float(an.XMLData.score) if an.XMLData.score is not None else None

        for hist in an.history:
            yield FeedItem(
                id=f"anime_episode_{an.id}_{hist.number}_{int(hist.at.timestamp())}",
                ftype="anime_episode",
                when=hist.at,
                url=url,
                image_url=_image_url(an),
                subtitle=f"Episode {hist.number}",
                collection=str(an.id),
                part=hist.number,  # no reliable season data for anime data
                release_date=an.APIList.start_date,
                title=an.APIList.title,
            )
        if an.XMLData.status.casefold() == "completed":
            if dt := _completed_datetime(an):
                yield FeedItem(
                    id=f"anime_entry_{an.id}",
                    ftype="anime",
                    when=dt,
                    url=url,
                    image_url=_image_url(an),
                    title=an.APIList.title,
                    release_date=an.APIList.start_date,
                    score=score,
                )


def _manga() -> Iterator[FeedItem]:
    for mn in mal.manga():
        if mn.username != os.environ["MAL_USERNAME"]:
            continue

        if mn.APIList is None:
            logger.warning(f"No API info for manga {mn.XMLData.id}")
            continue

        url = f"https://myanimelist.net/manga/{mn.id}"
        score = float(mn.XMLData.score) if mn.XMLData.score is not None else None

        for hist in mn.history:
            yield FeedItem(
                id=f"manga_chapter_{mn.id}_{hist.number}_{int(hist.at.timestamp())}",
                ftype="manga_chapter",
                when=hist.at,
                url=url,
                collection=str(mn.id),
                image_url=_image_url(mn),
                subtitle=f"Chapter {hist.number}",
                part=hist.number,  # no reliable volume data for manga data
                release_date=mn.APIList.start_date,
                title=mn.APIList.title,
            )
        if mn.XMLData.status.casefold() == "completed":
            if dt := _completed_datetime(mn):
                yield FeedItem(
                    id=f"manga_entry_{mn.id}",
                    ftype="manga",
                    when=dt,
                    url=url,
                    image_url=_image_url(mn),
                    title=mn.APIList.title,
                    release_date=mn.APIList.start_date,
                    score=score,
                )


def history() -> Iterator[FeedItem]:
    yield from _anime()
    yield from _manga()


def _deleted_anime() -> Iterator[FeedItem]:
    from my.mal.export import deleted_anime

    for an in deleted_anime():
        if an.username != os.environ["MAL_USERNAME"]:
            continue

        url = f"https://myanimelist.net/anime/{an.id}"
        score = float(an.XMLData.score) if an.XMLData.score is not None else None

        release_date = None
        if an.APIList is not None:
            release_date = an.APIList.start_date

        for hist in an.history:
            yield FeedItem(
                id=f"anime_episode_{an.id}_{hist.number}_{int(hist.at.timestamp())}",
                ftype="anime_episode",
                when=hist.at,
                url=url,
                collection=str(an.id),
                image_url=_image_url(an),
                subtitle=f"Episode {hist.number}",
                part=hist.number,  # no reliable season data for anime data
                title=an.XMLData.title,
                release_date=release_date,
            )

        if an.XMLData.status.casefold() == "completed":
            if dt := _completed_datetime(an):
                yield FeedItem(
                    id=f"anime_entry_{an.id}",
                    ftype="anime",
                    when=dt,
                    url=url,
                    image_url=_image_url(an),
                    title=an.XMLData.title,
                    release_date=release_date,
                    score=score,
                )


def _deleted_manga() -> Iterator[FeedItem]:
    from my.mal.export import deleted_manga

    for mn in deleted_manga():
        if mn.username != os.environ["MAL_USERNAME"]:
            continue

        url = f"https://myanimelist.net/manga/{mn.id}"
        score = float(mn.XMLData.score) if mn.XMLData.score is not None else None

        release_date = None
        if mn.APIList is not None:
            release_date = mn.APIList.start_date

        for hist in mn.history:
            yield FeedItem(
                id=f"manga_chapter_{mn.id}_{hist.number}_{int(hist.at.timestamp())}",
                ftype="manga_chapter",
                when=hist.at,
                url=url,
                collection=str(mn.id),
                image_url=_image_url(mn),
                subtitle=f"Chapter {hist.number}",
                part=hist.number,  # no reliable volume data for manga data
                title=mn.XMLData.title,
                release_date=release_date,
            )

        if mn.XMLData.status.casefold() == "completed":
            if dt := _completed_datetime(mn):
                yield FeedItem(
                    id=f"manga_entry_{mn.id}",
                    ftype="manga",
                    when=dt,
                    url=url,
                    image_url=_image_url(mn),
                    title=mn.XMLData.title,
                    release_date=release_date,
                    score=score,
                )


# items which have been deleted from MAL
# https://github.com/purarue/malexport/#recover_deleted
def deleted_history() -> Iterator[FeedItem]:
    yield from _deleted_anime()
    yield from _deleted_manga()
