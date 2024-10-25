import os
import string
import warnings
from typing import Iterator, Optional, Dict, Any, cast, Literal
from datetime import datetime, date
from functools import cache

import requests
from url_cache.core import URLCache, Summary

from .model import FeedItem
from ..log import logger


def game_center() -> Iterator[FeedItem]:
    from my.apple.privacy_export import events, GameAchievement

    for e in events():
        if isinstance(e, GameAchievement):
            yield FeedItem(
                id=f"game_center_{int(e.dt.timestamp())}_{_slugify(e.title)}_{e.percentage}",
                ftype="game_achievement",
                title=e.title,
                subtitle=e.game_name,
                when=e.dt,
            )


def _slugify(st: str) -> str:
    ss = "".join(s for s in st.casefold() if s in string.ascii_lowercase + " ")
    return ss.replace(" ", "_")


def steam() -> Iterator[FeedItem]:
    from my.steam.scraper import achievements

    for ac in achievements():
        if isinstance(ac, Exception):
            logger.debug(f"Ignoring exception: {ac}")
            continue
        if ac.achieved_on is None:
            logger.debug(f"steam, no datetime on achievement: {ac}")
            continue
        yield FeedItem(
            id=f"steam_{str(ac.achieved_on.date())}_{_slugify(ac.title)}",
            ftype="game_achievement",
            when=ac.achieved_on,
            title=ac.title,
            subtitle=ac.game_name,
            image_url=ac.icon,
        )


class GiantBombCache(URLCache):
    def request_data(self, url: str) -> Summary:  # type: ignore[override]
        uurl = self.preprocess_url(url)
        logger.info(f"Caching info for grouvee: {url}")
        self.sleep()
        r = requests.get(
            uurl,
            params={"api_key": os.environ["GIANTBOMB_API_KEY"], "format": "json"},
            headers={"User-Agent": f"{os.environ['USER']}_my_feed"},
        )
        r.raise_for_status()
        return Summary(url=uurl, data={}, metadata=r.json(), timestamp=datetime.now())


@cache
def gb_cache() -> GiantBombCache:
    default_local = os.path.expanduser("~/.local/share")
    cache_dir = os.path.join(default_local, "giantbomb_cache")
    if "GIANTBOMB_CACHE_DIR" in os.environ:
        cache_dir = os.environ["GIANTBOMB_CACHE_DIR"]
    return GiantBombCache(cache_dir=cache_dir, sleep_time=5)


def fetch_giantbomb_data(giantbomb_id: int) -> Optional[Summary]:
    url = f"http://www.giantbomb.com/api/game/{giantbomb_id}/"
    try:
        return gb_cache().get(url)
    except requests.RequestException as r:
        warnings.warn(f"Could not cache {url}: {str(r)}")
        return None


def _grouvee_img(res: dict[str, Any]) -> Optional[str]:
    """traverse API resp and grab the thumbnail/medium image"""
    if img := res.get("image"):
        assert isinstance(img, dict)
        for key in ("medium_url", "thumb_url"):
            if img_url := img.get(key):
                return cast(str, img_url)
    return None


def grouvee() -> Iterator[FeedItem]:
    from my.grouvee.export import played

    for g in played():
        assert g.giantbomb_id is not None
        gb_data: Optional[Summary] = fetch_giantbomb_data(g.giantbomb_id)
        assert gb_data is not None, f"No summary returned for {g.giantbomb_id}"

        # handle error
        res: dict[str, Any]
        if gb_data.metadata["error"] == "OK":
            res = gb_data.metadata["results"]
            assert isinstance(res, dict), str(gb_data.metadata)
        else:
            res = {}

        rel: Optional[date] = g.release_date

        # update with more accurate date, if possible
        if rel_str := res.get("original_release_date"):
            try:
                rel = date.fromisoformat(rel_str.strip())
            except ValueError:
                pass

        dt: datetime | None = None
        for s in g.shelves:
            dt = s.added
            break
        score = None
        if g.rating is not None:
            score = g.rating * 2
        assert dt is not None
        yield FeedItem(
            id=f"grouvee_{g.grouvee_id}",
            ftype="game",
            title=g.name,
            url=g.url,
            release_date=rel,
            image_url=_grouvee_img(res),
            when=dt,
            score=score,
        )


def osrs() -> Iterator[FeedItem]:
    from my.runelite.screenshots import screenshots, Level
    from my.time.tz.via_location import localize

    IGNORED_SCREENSHOTS = {
        "Kingdom Rewards",
        "Collection Log",
    }

    for sc in screenshots():
        if sc.screenshot_type in IGNORED_SCREENSHOTS:
            continue
        id_: str
        desc: str
        img: Optional[str] = None
        dt = localize(sc.dt)
        if "HPIDATA" in os.environ:
            if prefix := os.getenv("RUNELITE_PHOTOS_PREFIX"):
                img = os.path.join(prefix, str(sc.path).lstrip(os.environ["HPIDATA"]))
        if isinstance(sc.description, Level):
            id_ = f"osrs_level_{sc.description.skill.casefold()}_{sc.description.level}_{int(dt.timestamp())}"
            desc = f"{sc.description.skill} Level {sc.description.level}"
        else:
            id_ = f"osrs_{_slugify(sc.description)}_{int(dt.timestamp())}"
            assert isinstance(sc.description, str)
            desc = sc.description
        yield FeedItem(
            id=id_,
            ftype="osrs_achievement",
            title=f"OSRS - {desc}",
            image_url=img,
            subtitle=sc.screenshot_type,
            when=dt,
        )


# TODO: make configurable through config file?
CHESS_USERNAME = os.environ.get("CHESS_USERNAME", "purarue")


def chess() -> Iterator[FeedItem]:
    from io import StringIO

    import chess.pgn
    import chess.svg

    from my.chess.export import history
    from chess_export.chessdotcom.model import ChessDotComGame
    from chess_export.lichess.model import LichessGame

    for game in history():
        if game.pgn is None:
            logger.debug(f"Ignoring chess game with no PGN: {game}")
            continue
        assert isinstance(
            game, (ChessDotComGame, LichessGame)
        ), f"Unexpected game type {type(game)}"
        result: Literal["won", "loss", "draw", "unknown"] = "unknown"
        if has_result := game.result(CHESS_USERNAME):
            result = has_result.value

        url: str
        if isinstance(game, ChessDotComGame):
            url = game.url
        else:
            # 9999 skips to the last move of the match
            url = f"https://lichess.org/{game.game_id}#9999"

        me = "white" if game.white.username == CHESS_USERNAME else "black"
        pgn = chess.pgn.read_game(StringIO(game.pgn))
        if pgn is None:
            logger.warning(f"Could not parse PGN: {game.pgn}")
            continue
        # iterate through mainline moves to create svg
        board = pgn.board()
        for move in pgn.mainline_moves():
            board.push(move)
        data: Dict[str, Any] = {}
        data["svg"] = str(chess.svg.board(board))
        yield FeedItem(
            id=f"chess_{int(game.end_time.timestamp())}",
            title=f"Chess ({me}) - {result.capitalize()}",
            ftype="chess",
            when=game.end_time,
            data=data,
            url=url,
        )
