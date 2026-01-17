from collections.abc import Iterator

from my.offline.listens import history as of_history

from .model import FeedItem


def history() -> Iterator[FeedItem]:
    for listen in of_history():
        yield FeedItem(
            id=f"offline_listen_{int(listen.when.timestamp())}",
            ftype="listen",
            when=listen.when,
            title=listen.track,
            creator=listen.artist,
            subtitle=listen.album,
        )
