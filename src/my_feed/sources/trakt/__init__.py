"""
Use https://github.com/seanbreckenridge/traktexport
(through my https://github.com/seanbreckenridge/HPI trakt.py file)
to get feed data for movies and TV episodes

Use TMDB to fetch image and release date information for feed items
"""

from typing import Iterator, Dict, Optional, Union, List, Tuple
from datetime import date, datetime

import traktexport.dal as D
from my.trakt.export import ratings, history as trakt_history

from .tmdb import fetch_tmdb_data, BASE_URL
from ..model import FeedItem


def _fetch_image(url: str, width: int) -> Optional[str]:
    """
    Request or grab TMDB data for the URL from cache, returning the poster path if it exists
    """
    if summary := fetch_tmdb_data(url):
        # TODO: if metadata is returned was an error (tmdb api returned no data)
        # and this was requested more than 3 months ago, try requesting
        # again and overwriting the data in the summary cache
        #
        # even if the fetch_tmdb_data failed (so it wrote an error), 'poster_path'
        # doesn't exist on errors, so episodes will continue to check the season and
        # then the show poster
        if still_path := summary.metadata.get("still_path"):
            assert still_path.startswith("/")
            return f"https://image.tmdb.org/t/p/w{400}{still_path}?s"
        if poster_path := summary.metadata.get("poster_path"):
            assert poster_path.startswith("/")
            return f"https://image.tmdb.org/t/p/w{400}{poster_path}?p"
    return None


def get_image(
    media_data: Union[D.Movie, D.Episode, D.Show], *, width: int = 400
) -> Optional[str]:
    """
    Get an image for particular media data
    For movies, uses the movies endpoint

    For episodes, if there is no image for the episode, uses the season poster. If theres
    no season poster, uses the show poster
    """
    if isinstance(media_data, D.Movie):
        if movie_id := media_data.ids.tmdb_id:
            if poster := _fetch_image(f"{BASE_URL}/movie/{movie_id}", width=width):
                return poster
    elif isinstance(media_data, D.Episode):
        # try episode, then season, then tv show
        if tv_id := media_data.show.ids.tmdb_id:
            if poster := _fetch_image(
                f"{BASE_URL}/tv/{tv_id}/season/{media_data.season}/episode/{media_data.episode}",
                width,
            ):
                return poster
            elif season_poster := _fetch_image(
                f"{BASE_URL}/tv/{tv_id}/season/{media_data.season}", width
            ):
                return season_poster
            elif show_poster := _fetch_image(f"{BASE_URL}/tv/{tv_id}", width):
                return show_poster
    elif isinstance(media_data, D.Show):
        if tv_id := media_data.ids.tmdb_id:
            if show_poster := _fetch_image(f"{BASE_URL}/tv/{tv_id}", width):
                return show_poster
    return None


def get_release_date(media_data: Union[D.Movie, D.Episode, D.Show]) -> Optional[date]:
    """
    Get the release date of the movie/episode
    """
    if isinstance(media_data, D.Movie):
        if movie_id := media_data.ids.tmdb_id:
            if summary := fetch_tmdb_data(f"{BASE_URL}/movie/{movie_id}"):
                if dt_raw := summary.metadata.get("release_date"):
                    if dt := dt_raw.strip():
                        return date.fromisoformat(dt)
    elif isinstance(media_data, D.Episode):
        if tv_id := media_data.show.ids.tmdb_id:
            if summary := fetch_tmdb_data(
                f"{BASE_URL}/tv/{tv_id}/season/{media_data.season}/episode/{media_data.episode}"
            ):
                if dt_raw := summary.metadata.get("air_date"):
                    if dt := dt_raw.strip():
                        return date.fromisoformat(dt)
    elif isinstance(media_data, D.Show):
        if tv_id := media_data.ids.tmdb_id:
            if summary := fetch_tmdb_data(f"{BASE_URL}/tv/{tv_id}"):
                if dt_raw := summary.metadata.get("first_air_date"):
                    if dt := dt_raw.strip():
                        return date.fromisoformat(dt)
    return None


def get_genres(media_data: Union[D.Movie, D.Episode, D.Show]) -> List[str]:
    """
    Get genres for the movie/show
    """
    if isinstance(media_data, D.Movie):
        if movie_id := media_data.ids.tmdb_id:
            if summary := fetch_tmdb_data(f"{BASE_URL}/movie/{movie_id}"):
                if genre_list := summary.metadata.get("genres"):
                    return [genre["name"].casefold() for genre in genre_list]
    elif isinstance(media_data, (D.Episode, D.Show)):
        tv_id = (
            media_data.show.ids.tmdb_id
            if isinstance(media_data, D.Episode)
            else media_data.ids.tmdb_id
        )
        if tv_id:
            if summary := fetch_tmdb_data(f"{BASE_URL}/tv/{tv_id}"):
                if genre_list := summary.metadata.get("genres"):
                    return [genre["name"].casefold() for genre in genre_list]
    return []


def get_rating(
    media_data: Union[D.Movie, D.Episode, D.Show], *, rating_map: Dict[str, D.Rating]
) -> Optional[float]:
    """
    get rating for movie/episode. If that doesn't exist,
    use the rating for the show, then the season, if those exists
    """
    if isinstance(media_data, D.Movie):
        if rt := rating_map.get(media_data.url):
            return float(rt.rating)
    elif isinstance(media_data, (D.Show, D.Episode)):
        url = (
            media_data.show.url if isinstance(media_data, D.Episode) else media_data.url
        )
        # use rating for the show, if that exists
        if rt := rating_map.get(url):
            return float(rt.rating)
    return None


# create mapping from most recent time I watched a movie/episode url (movie/show URL) -> datetime
def _history_mapping(hst: List[D.HistoryEntry]) -> Dict[str, datetime]:
    hst_mapping: Dict[str, datetime] = {}

    for h in hst:
        if h.action != "watch":
            continue
        if isinstance(h.media_data, (D.Movie, D.Episode)):
            url: str
            if isinstance(h.media_data, D.Movie):
                url = h.media_data.url
            else:
                url = h.media_data.show.url
            if url not in hst_mapping:
                hst_mapping[url] = h.watched_at

    return hst_mapping


def history() -> Iterator[FeedItem]:

    emitted: set[Tuple[str, datetime]] = set()

    hst = list(trakt_history())

    # url to datetime mapping
    hst_mapping: Dict[str, datetime] = _history_mapping(hst)

    # url to rating mapping
    rm: Dict[str, D.Rating] = {r.media_data.url: r for r in ratings()}

    for rt in rm.values():
        m = rt.media_data
        if not isinstance(m, (D.Movie, D.Show)):
            continue

        dt: datetime
        if m.url in hst_mapping:
            dt = hst_mapping[m.url]
        else:
            dt = rt.rated_at

        title: str = m.title
        assert m.ids.trakt_slug is not None

        # add this rating to emitted, so we don't have movies right next to each other
        key: Tuple[str, datetime] = (m.ids.trakt_slug, dt)
        emitted.add(key)

        yield FeedItem(
            id=f"trakt_{m.__class__.__name__.lower()}_{m.ids.trakt_slug}",
            title=m.title,
            ftype="trakt_movie" if isinstance(m, D.Movie) else "trakt_show",
            # TODO: date-shift items at account creation
            when=dt,
            url=m.url,
            image_url=get_image(m),
            score=get_rating(m, rating_map=rm),
            release_date=get_release_date(m),
            tags=get_genres(m),
        )

    # iterate through individual history/episode entries
    for h in hst:
        if h.action in {"checkin", "scrobble"}:
            continue
        assert h.action == "watch", f"Unexpected action {h.action} {h}"
        m = h.media_data

        assert isinstance(m, (D.Episode, D.Movie))

        # set default (for movie) and episode metadata
        title = m.title
        part: Optional[int] = None
        subpart: Optional[int] = None
        subtitle: Optional[str] = None
        collection: str
        if isinstance(m, D.Episode):
            part = m.season
            subpart = m.episode
            subtitle = m.show.title
            assert m.show.ids.trakt_slug is not None
            collection = m.show.ids.trakt_slug
        else:
            assert m.ids.trakt_slug is not None
            collection = m.ids.trakt_slug

        # this was already added while adding ratings; ignore it
        key = (collection, h.watched_at)
        if key in emitted:
            continue

        yield FeedItem(
            id=f"trakt_{h.history_id}",
            title=title,
            ftype="trakt_history_episode"
            if isinstance(m, D.Episode)
            else "trakt_history_movie",
            when=h.watched_at,
            part=part,
            subpart=subpart,
            subtitle=subtitle,
            url=m.url,
            image_url=get_image(m),
            collection=collection,
            score=get_rating(m, rating_map=rm),
            release_date=get_release_date(m),
            tags=get_genres(m),
        )
