"""
Gets anime/manga from using
https://github.com/purarue/malexport
"""

import os
import typing
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Iterator, Optional, Union

import my.mal.export as mal

from .model import FeedItem
from ..log import logger


def _image_url(data: Union[mal.AnimeData, mal.MangaData]) -> Optional[str]:
    if data.APIList is None:
        # TODO: fetch from dbsentinel?
        # https://dbsentinel.purarue.xyz/
        return None
    api_images = data.APIList.main_picture
    for k in (
        "medium",
        "large",
    ):
        if img_url := api_images.get(k):
            return img_url
    return None


from .trakt.tmdb import tmdb_urlcache, TMDBCache

TMDB_CACHE: TMDBCache | None = None


class SeasonInfo(typing.TypedDict):
    num: int
    ep_count: int


class TMDBInfo(typing.TypedDict):
    trakt_id: int
    tmdb_id: int
    title: str
    media_type: typing.Literal["movie", "tv"]
    season: int | None
    episode_offset: int | None
    season_info: list[SeasonInfo]


TMDBMapping = typing.Dict[str, TMDBInfo | None]


def load_mal_tmdb_mapping() -> TMDBMapping:
    if "MAL_TMDB_MAPPING" not in os.environ:
        return {}
    pth = Path(os.environ["MAL_TMDB_MAPPING"])
    if pth.exists():
        dat: TMDBMapping = json.loads(pth.read_text())
        return dat
    return {}


EPISODE_MAPPING: TMDBMapping = load_mal_tmdb_mapping()
TMDB_CACHE = tmdb_urlcache()


def _anime_episode_url(
    data: mal.AnimeData, episode: int
) -> typing.Tuple[Optional[str], typing.List[str]]:
    from .trakt import fetch_image_by_params, _destructure_img_result

    if TMDB_CACHE is None:
        return _image_url(data), ["i_poster"]
    if str(data.id) in EPISODE_MAPPING:
        assert episode >= 1
        tmdb_data = EPISODE_MAPPING[str(data.id)]
        if tmdb_data is None:
            pass
        elif tmdb_data["media_type"] == "tv" and tmdb_data["season_info"]:
            # offset the MAL episode number to the correct TMDB season/episode
            season = tmdb_data["season"] if tmdb_data["season"] is not None else 1
            offset = (
                tmdb_data["episode_offset"]
                if tmdb_data["episode_offset"] is not None
                else 0
            )
            season_info = tmdb_data["season_info"]
            seasons = [s for s in season_info if s["num"] >= season]
            trakt_episodes: typing.List[typing.Tuple[int, int]] = []
            for ssn in seasons:
                for ep in range(1, ssn["ep_count"] + 1):
                    trakt_episodes.append((ssn["num"], ep))
            # this may not always be true if there is an offset, need to account for that
            assert len(trakt_episodes) == sum(s["ep_count"] for s in seasons)
            try:
                season_episode = trakt_episodes[episode + offset - 1]
            except IndexError:
                # this typically meant there was a mismatch between MAL and TMDB on specials/something else
                logger.warning(
                    f"Failed to index episode {episode} offset {offset} in trakt episodes data {len(trakt_episodes)}"
                )
                return _image_url(data), ["i_poster"]
            else:
                img, _ = _destructure_img_result(
                    fetch_image_by_params(
                        tv_id=tmdb_data["tmdb_id"],
                        season=season_episode[0],
                        episode=season_episode[1],
                    )
                )
                # if no image from tmdb, then just default to MAL
                if img is not None:
                    return img, ["i_still"]
        elif tmdb_data["media_type"] == "movie":
            img, flags = _destructure_img_result(
                fetch_image_by_params(movie_id=tmdb_data["tmdb_id"])
            )
            if img is not None:
                return img, flags
    return _image_url(data), ["i_poster"]


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
            img, flags = _anime_episode_url(an, hist.number)
            yield FeedItem(
                id=f"anime_episode_{an.id}_{hist.number}_{int(hist.at.timestamp())}",
                ftype="anime_episode",
                flags=flags,
                when=hist.at,
                url=url,
                image_url=img,
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
                    flags=["i_poster"],
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
                flags=["i_poster"],
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
                    flags=["i_poster"],
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
                flags=["i_poster"],
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
                    flags=["i_poster"],
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
                flags=["i_poster"],
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
                    flags=["i_poster"],
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
