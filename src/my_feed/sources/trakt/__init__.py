"""
Use https://github.com/purarue/traktexport
(through my https://github.com/purarue/HPI trakt.py file)
to get feed data for movies and TV episodes

Use TMDB to fetch image and release date information for feed items
"""

from typing import Iterator, Dict, Optional, Union, List, Tuple
from datetime import date, datetime

import traktexport.dal as D
from my.trakt.export import ratings, history as trakt_history

from .tmdb import fetch_tmdb_data, BASE_URL
from ..model import FeedItem


def _fetch_image(url: str, width: int) -> Optional[Tuple[str, List[str]]]:
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
            return f"https://image.tmdb.org/t/p/w{width}{still_path}", ["i_still"]
        if poster_path := summary.metadata.get("poster_path"):
            assert poster_path.startswith("/")
            return f"https://image.tmdb.org/t/p/w{width}{poster_path}", ["i_poster"]
    return None


def fetch_image_by_params(
    *,
    movie_id: Optional[int] = None,
    tv_id: Optional[int] = None,
    season: Optional[int] = None,
    episode: Optional[int] = None,
    width: int = 400,
) -> Optional[Tuple[str, List[str]]]:
    """
    - Movie image if movie_id is provided
    - Episode image if tv_id, season, and episode are provided
    - Season image if tv_id and season are provided
    - Show image if only tv_id is provided

    For episodes, if there is no image for the episode, uses the season poster. If there's
    no season poster, uses the show poster
    """
    if movie_id:
        return _fetch_image(f"{BASE_URL}/movie/{movie_id}", width=width)
    elif tv_id:
        if episode is not None and season is not None:
            if episode_poster := _fetch_image(
                f"{BASE_URL}/tv/{tv_id}/season/{season}/episode/{episode}", width
            ):
                return episode_poster
        if season is not None:
            if season_poster := _fetch_image(
                f"{BASE_URL}/tv/{tv_id}/season/{season}", width
            ):
                return season_poster
        return _fetch_image(f"{BASE_URL}/tv/{tv_id}", width)
    return None


def get_image(
    media_data: Union[D.Movie, D.Episode, D.Show], *, width: int = 400
) -> Optional[Tuple[str, List[str]]]:
    """
    Wrapper function to maintain compatibility with the old get_image signature.
    Extracts necessary parameters from media_data and calls fetch_image_by_params.
    """
    if isinstance(media_data, D.Movie):
        movie_id = media_data.ids.tmdb_id
        return fetch_image_by_params(movie_id=movie_id, width=width)
    elif isinstance(media_data, D.Episode):
        tv_id = media_data.show.ids.tmdb_id
        season = media_data.season
        episode = media_data.episode
        return fetch_image_by_params(
            tv_id=tv_id, season=season, episode=episode, width=width
        )
    elif isinstance(media_data, D.Show):
        tv_id = media_data.ids.tmdb_id
        return fetch_image_by_params(tv_id=tv_id, width=width)


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


def _get_genres(media_data: Union[D.Movie, D.Episode, D.Show]) -> List[str]:
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


def _destructure_img_result(
    res: Optional[Tuple[str, List[str]]]
) -> Tuple[Optional[str], List[str]]:
    match res:
        case None:
            return None, []
        case (img, flags):
            return img, flags
        case _:
            return None, []


def history() -> Iterator[FeedItem]:
    # emitted: set[Tuple[str, str, datetime]] = set()

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
        # key: Tuple[str, str, datetime] = (m.ids.trakt_slug, type(m).__name__, dt)
        # emitted.add(key)
        img_url, flags = _destructure_img_result(get_image(m))

        yield FeedItem(
            id=f"trakt_{m.__class__.__name__.lower()}_{m.ids.trakt_slug}",
            title=m.title,
            ftype="trakt_movie" if isinstance(m, D.Movie) else "trakt_show",
            # TODO: date-shift items at account creation
            when=dt,
            url=m.url,
            image_url=img_url,
            flags=flags,
            score=get_rating(m, rating_map=rm),
            release_date=get_release_date(m),
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
            title = m.show.title
            subtitle = m.title
            part = m.season
            subpart = m.episode
            assert m.show.ids.trakt_slug is not None
            collection = m.show.ids.trakt_slug
        else:
            assert m.ids.trakt_slug is not None
            collection = m.ids.trakt_slug

        # if this was already added while adding ratings, and this is a movie, ignore it
        # this is to prevent duplicate movie/rating entries from appearing when I've only watched
        # the movie once. if its an episode of a show, still want the episode and the season/show
        # to appear on the feed multiple times
        # key = (collection, type(m).__name__, h.watched_at)
        # if key in emitted and isinstance(m, D.Movie):
        #     continue

        img_url, flags = _destructure_img_result(get_image(m))
        yield FeedItem(
            id=f"trakt_{h.history_id}",
            title=title,
            ftype=(
                "trakt_history_episode"
                if isinstance(m, D.Episode)
                else "trakt_history_movie"
            ),
            when=h.watched_at,
            part=part,
            subpart=subpart,
            subtitle=subtitle,
            url=m.url,
            image_url=img_url,
            flags=flags,
            collection=collection,
            # score=get_rating(m, rating_map=rm),
            release_date=get_release_date(m),
        )
