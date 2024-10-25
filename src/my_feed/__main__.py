import time
import json
from pathlib import Path
from typing import Iterator, Callable, Optional, List, TypeGuard, Any, Set

import click

from .log import logger
from .sources.model import FeedItem
from .blur import Blurred


@click.group()
def main():
    """My Personal Media Feed"""


BLUE = (83, 158, 206)


def _check_source(source: Any) -> TypeGuard[Callable[[], Iterator[FeedItem]]]:
    if not callable(source):
        click.echo(f"{source} is not callable, ignoring source", err=True)
        return False
    return True


def _sources() -> Iterator[Callable[[], Iterator[FeedItem]]]:
    try:
        from my.config.feed import sources  # type: ignore[import]
    except Exception:
        click.echo(
            "Could not import sources from my.config.feed, see docs or https://github.com/purarue/dotfiles/blob/master/.config/my/my/config/feed.py as an example",
            err=True,
        )
        return

    assert callable(sources), "sources imported from my.config.feed is not a function"
    for src in iter(sources()):
        if _check_source(src):
            yield src


def data(
    *, allow: List[str], deny: List[str], blurred: Blurred | None, echo: bool = False
) -> Iterator[FeedItem]:
    for producer in _sources():
        func = f"{producer.__module__}.{producer.__qualname__}"
        if len(allow) > 0 and not any(substr in func for substr in allow):
            continue
        if len(deny) > 0 and any(substr in func for substr in deny):
            continue
        emitted: set[str] = set()
        start_time = time.time()
        func = f"{producer.__module__}.{producer.__qualname__}"
        ext = f"Extracting {click.style(func, fg='green')}"
        click.echo(f"{ext}...")
        for item in producer():
            assert isinstance(item, FeedItem)
            item.check()
            if item.id in emitted:
                logger.warning(f"Duplicate id: {item.id} {item}")
                continue
            emitted.add(item.id)
            if echo:
                print(item)
            if blurred and blurred.should_be_blurred(feed_item=item):
                item.blur()
                click.echo(f"Blurred image: {item.id=} {item.title=} {item.image_url=}")
            yield item
        took = time.time() - start_time
        click.echo(
            f"{ext}: {click.style(str(len(emitted)), fg=BLUE)} items (took {click.style(round(took, 2), fg=BLUE)} seconds)",
            err=True,
        )


# TODO: this could allow either passing the ID or the URL
def _parse_blur_file(
    ctx: click.Context, param: click.Parameter, value: Optional[Path]
) -> Optional[Blurred]:
    if value is not None:
        return Blurred.parse_file(value)
    return None


def _parse_sources(
    ctx: click.Context, param: click.Parameter, value: Optional[str]
) -> List[str]:
    if value is not None:
        return [p.strip() for p in value.strip().split(",")]
    return []


@main.command(name="index", short_help="recompute feed data")
@click.option(
    "--echo/--no-echo",
    default=False,
    is_flag=True,
    help="Print feed items as they're computed",
)
@click.option(
    "-i",
    "--include-sources",
    default=None,
    help="A comma delimited list of substrings of sources to include. e.g. 'mpv,trakt,listens'",
    callback=_parse_sources,
    envvar="MY_FEED_INCLUDE_SOURCES",
)
@click.option(
    "-e",
    "--exclude-sources",
    default=None,
    envvar="MY_FEED_EXCLUDE_SOURCES",
    help="A comma delimited list of substrings of sources to exclude. e.g. 'mpv,trakt,listens'",
    callback=_parse_sources,
)
@click.option(
    "-E",
    "--exclude-id-file",
    default=None,
    help="A json file containing a list of IDs to exclude, from the /data/ids endpoint. reduces amount of data to sync to the server",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "-C",
    "--write-count-to",
    type=click.Path(writable=True, path_type=Path),
    default=None,
    help="Write the number of items to this file",
)
@click.option(
    "-B",
    "--blur-images-file",
    "blurred",
    default=None,
    help="A file containing a list of image URLs to blur, one per line",
    type=click.Path(exists=True, path_type=Path),
    callback=_parse_blur_file,
)
@click.argument(
    "OUTPUT", type=click.Path(writable=True, path_type=Path), required=False
)
def index(
    echo: bool,
    include_sources: List[str],
    exclude_sources: List[str],
    write_count_to: Optional[Path],
    output: Optional[Path],
    blurred: Optional[Blurred],
    exclude_id_file: Optional[Path],
) -> None:
    if blurred:
        click.echo("Blurred matchers:")
        click.echo("\n".join(map(str, blurred.items)))

    exclude_ids: Set[str] = set()
    if exclude_id_file is not None:
        click.echo(f"Reading exclude IDs from '{exclude_id_file}'")
        exclude_ids = set(json.loads(exclude_id_file.read_text()))
    all_items = list(
        data(allow=include_sources, deny=exclude_sources, blurred=blurred, echo=echo)
    )
    items = [i for i in all_items if i.id not in exclude_ids]

    if exclude_ids:
        click.echo(f"Excluded {click.style(len(all_items) - len(items), BLUE)} items")
    click.echo(f"Total: {click.style(len(items), BLUE)} items")
    if output is not None:
        click.echo(f"Writing to '{output}'")
        with output.open("w") as f:
            for item in items:
                f.write(item.to_json())
                f.write("\n")
        if write_count_to:
            write_count_to.write_text(str(len(items)))

    # hm: consume the rest of the generator so that cachew db closes...?
    #
    # happens when I use the albums module since it uses via_location.localize
    #
    # so, I suppose when trying to exit, since we never finished using the generator,
    # its still trying to write, but we exit so it fails to write to the closed db
    #
    # for posterity: https://gist.github.com/purarue/58f0b9f9dbcfeb970270e7fba99604d7
    # try:
    #     from my.time.tz.via_location import loc_tz_getter
    #
    #     list(loc_tz_getter())  # type: ignore
    # except Exception as e:
    #     logger.exception(e)


if __name__ == "__main__":
    main(prog_name="my_feed")
