A personal feed/website using [`HPI`](https://github.com/purarue/HPI)

Live at <https://purarue.xyz/feed/>

<img src="https://github.com/purarue/my_feed/blob/master/.github/my_feed.png" width=500/>

This uses:

- `python`: to get my data using [HPI](https://github.com/purarue/HPI), and to cleanup/enrich it with some local data/cached API requests. `my_feed index` is called in the [`feed_index`](./scripts/feed_index) script, which syncs a JSON file up to the server which `backend` can combine into the sqlite database
- `golang`: basic REST API to let frontend paginate through the data, authenticated endpoints for updating the sqlite database
- `typescript`: public-facing [frontend](https://purarue.xyz/feed/); requests to the backend, lets user filter/order/search the data

### Data Sources:

- Music
  - [listenbrainz_export](https://github.com/purarue/listenbrainz_export) for scrobbles, from [listenbrainz](https://listenbrainz.org/) (similar to last.fm)
  - [mpv_history_daemon](https://github.com/purarue/mpv-history-daemon) for [mpv](https://github.com/mpv-player/mpv) history
- Movies/TV Shows
  - [traktexport](https://github.com/purarue/traktexport), grabbing data from [Trakt](https://trakt.tv/). Trakt provides [TMDB](http://themoviedb.org/) IDs, so I can fetch images for each episode
- Games
  - [grouvee_export](https://github.com/purarue/grouvee_export) to parse the CSV export from [Grouvee](https://www.grouvee.com/) with images from [GiantBomb](https://www.giantbomb.com/)
  - [steamscraper](https://github.com/purarue/steamscraper) to scrape my [steam](https://steamcommunity.com/) achievements
  - [chess_export](https://github.com/purarue/chess_export) for chess games, the [python-chess.svg](https://python-chess.readthedocs.io/en/latest/) package to parse the PGNs into SVGs
- Albums
  - [albums](https://github.com/purarue/albums) which requests out to [discogs](https://www.discogs.com/)
- Anime/Manga
  - [malexport](https://github.com/purarue/malexport/), saving my data from [MAL](https://myanimelist.net/)

If not mentioned its likely a module in [HPI](https://github.com/purarue/HPI)

I periodically index all my data [in the background](https://purarue.xyz/d/my_feed_index_bg.job?redirect):

```
Extracting my_feed.sources.listens.history...
Extracting my_feed.sources.listens.history: 5388 items (took 0.14 seconds)
Extracting my_feed.sources.games.steam...
Extracting my_feed.sources.games.steam: 285 items (took 0.01 seconds)
Extracting my_feed.sources.games.osrs...
Extracting my_feed.sources.games.osrs: 924 items (took 0.03 seconds)
Extracting my_feed.sources.games.game_center...
Extracting my_feed.sources.games.game_center: 141 items (took 0.02 seconds)
Extracting my_feed.sources.games.grouvee...
Extracting my_feed.sources.games.grouvee: 243 items (took 0.15 seconds)
Extracting my_feed.sources.games.chess...
Extracting my_feed.sources.games.chess: 681 items (took 2.98 seconds)
Extracting my_feed.sources.trakt.history...
Extracting my_feed.sources.trakt.history: 15327 items (took 11.51 seconds)
Extracting my_feed.sources.mpv.history...
Extracting my_feed.sources.mpv.history: 13807 items (took 13.67 seconds)
Extracting my_feed.sources.nextalbums.history...
Extracting my_feed.sources.nextalbums.history: 1938 items (took 2.36 seconds)
Extracting my_feed.sources.mal.history...
Extracting my_feed.sources.mal.history: 20865 items (took 3.58 seconds)
Total: 59599 items
Writing to 'backend/data/1644267551.json'
```

... which then gets synced up and combined into the `sqlite` database on the [`backend`](./backend/); all handled by [`feed_index`](./scripts/feed_index)

That has a [front-end](https://purarue.xyz/feed/) so I can view/filter/sort stuff and view the data as an infinite scrollable list

Served with `nginx` in prod, like:

```
location /feed/ {
  proxy_pass http://127.0.0.1:4500/feed;
}

location /feed/_next/ {
  # required since the above proxy pass doesn't end with '/'
  proxy_pass http://127.0.0.1:4500/feed/_next/;
}

location /feed_api/ {
  proxy_pass http://127.0.0.1:5100/;
}
```

### Install/Config:

For the python library:

```bash
git clone https://github.com/purarue/my_feed
pip install -e ./my_feed
```

... installs `my_feed` (or `python3 -m my_feed`)

This uses the `HPI` config structure (which you'd probably already have setup if you're using this)

To install dependencies for the servers, check the [frontend](./frontend/) and [backend](./backend/) directories.

So, in `~/.config/my/my/config/feed.py`, create a top-level `sources` function, which returns each _function_:

```python
from typing import Iterator, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from my_feed.sources.model import FeedItem


def sources() -> Iterator[Callable[[], Iterator["FeedItem"]]]:
    # yields functions, when which called yield FeedItem
    from my_feed.sources import games

    yield games.steam
    yield games.osrs
    yield games.game_center
    yield games.grouvee
    yield games.chess

    from my_feed.sources import (
        trakt,
        listens,
        nextalbums,
        mal,
        mpv,
        facebook_spotify_listens,
    )

    yield trakt.history
    yield listens.history
    yield nextalbums.history
    yield mal.history
    yield mpv.history
    yield facebook_spotify_listens.history
```

The [`feed_index`](./scripts/feed_index) script in this repo:

- warms the `my.time.tz.via_location` cache, so that timezones can be estimated for some of the data sources here
- does an `rsync` for some images hosted here
- requests the `/data/ids` endpoint on the server, which returns a list of known IDs (those are used to filter out duplicates before syncing)
- runs an `my_feed index` to save json objects to a local file
- Syncs the json up to my server with `scp`
- Server is pinged (at `/check`), which makes the server process the json files, updating the local sqlite database

To blur images, `my_feed index` accepts a `-B` flag, which lets you match against the `id`, `title`, or `image_url` with an [`fnmatch`](https://docs.python.org/3/library/fnmatch.html#module-fnmatch) or a `regex`. Those are placed in a file, one per line, for example:

```
id:*up_2009_*
title:*up_2009_*
image_url:*up_2009_*
id_regex:.*up_2009_.*
title_regex:.*up_2009_.*
image_url_regex:.*up_2009_.*
```

`my_feed` has a couple options that have developed over time, to let me ignore specific IDs (if I know they're already in the database), ignore sources which take a while to process (only do those once a week or so):

```
Usage: my_feed index [OPTIONS] [OUTPUT]

Options:
  --echo / --no-echo           Print feed items as they're computed
  -i, --include-sources TEXT   A comma delimited list of substrings of sources
                               to include. e.g. 'mpv,trakt,listens'
  -e, --exclude-sources TEXT   A comma delimited list of substrings of sources
                               to exclude. e.g. 'mpv,trakt,listens'
  -E, --exclude-id-file PATH   A json file containing a list of IDs to
                               exclude, from the /data/ids endpoint. reduces
                               amount of data to sync to the server
  -C, --write-count-to PATH    Write the number of items to this file
  -B, --blur-images-file PATH  A file containing a list of image URLs to blur,
                               one per line
  --help                       Show this message and exit.
```

### feed_check

`feed_check` updates some of my data which is updated more often (music (both mpv and listenbrainz), tv shows (trakt), chess, albums), by comparing the IDs of the latest items in the remote database to the corresponding live data sources.

This is pretty personal as it relies on my `anacron`-like [bgproc](https://github.com/purarue/bgproc) tool to handle updating data periodically.

So all of these follow some pattern like (e.g. for `chess`)

- get the `end_time` of the last couple items from the `my_feed` database (using the same `JSON` endpoints the frontend uses)
- get the first page of my chess games from the `chess.com` API using [chess_export](https://github.com/purarue/chess_export)
- if there's new data (the last `end_time` is not in the first page of the API), then:
  - remove the `evry tag` for the [job that updates my chess games](https://github.com/purarue/HPI-personal/blob/master/jobs/linux/backup_chess.job)
  - print 'chess'
- If anything was printed by the script:
  - I know at least one thing has expired, so I run `bgproc_on_machine` to update all the expired data
  - Run [scripts/feed_index](./scripts/feed_index) to update the `my_feed` database on my server

`feed_check` runs [once every 15 minutes](https://github.com/purarue/dotfiles/blob/df69db98e0256e7d9eb5f77cd1af9a354d782eaf/.local/scripts/supervisor_jobs/linux/my_feed_index_bg.job#L21-L27), so my data is never more than 15 minutes out of date.

Example output:

```
[I 230921 15:44:15 feed_check:213] Checking 'check_albums'
[I 230921 15:44:18 feed_check:42] Requesting https://purarue.xyz/feed_api/data/?offset=0&order_by=when&sort=desc&limit=500&ftype=album
[I 230921 15:44:18 feed_check:213] Checking 'check_trakt'
[D 230921 15:44:18 export:32] Requesting 'https://api-v2launch.trakt.tv/users/purplepinapples/history?limit=100&page=1'...
[D 230921 15:44:20 export:46] First item: {'id': 9230963378, 'watched_at': '2023-09-21T08:03:23.000Z', 'action': 'watch', 'type': 'episode', 'episode': {'season': 1, 'number': 1, 'title': 'ROMANCE DAWN', 'ids': {'trakt': 5437335, 'tvdb': 8651297, 'imdb': 'tt11748904', 'tmdb': 2454621, 'tvrage': None}}, 'show': {'title': 'ONE PIECE', 'year': 2023, 'ids': {'trakt': 184618, 'slug': 'one-piece-2023', 'tvdb': 392276, 'imdb': 'tt11737520', 'tmdb': 111110, 'tvrage': None}}}
[I 230921 15:44:20 feed_check:42] Requesting https://purarue.xyz/feed_api/data/?offset=0&order_by=when&sort=desc&limit=10&ftype=trakt_history_movie,trakt_history_episode
[I 230921 15:44:21 feed_check:213] Checking 'check_chess'
[I 230921 15:44:21 feed_check:42] Requesting https://purarue.xyz/feed_api/data/?offset=0&order_by=when&sort=desc&limit=10&ftype=chess
Requesting https://api.chess.com/pub/player/purarue/games/archives
Requesting https://api.chess.com/pub/player/purarue/games/2023/09
[I 230921 15:44:22 feed_check:213] Checking 'check_mpv'
[I 230921 15:44:23 feed_check:42] Requesting https://purarue.xyz/feed_api/data/?offset=0&order_by=when&sort=desc&limit=500&ftype=listen
[I 230921 15:44:23 feed_check:213] Checking 'check_listens'
[I 230921 15:44:23 feed_check:42] Requesting https://purarue.xyz/feed_api/data/?offset=0&order_by=when&sort=desc&limit=500&ftype=listen
[D 230921 15:44:25 export:62] Requesting https://api.listenbrainz.org/1/user/purarue/listens?count=100
[D 230921 15:44:25 export:84] Have 100, now searching for listens before 2023-09-11 04:39:08...
[I 230921 15:44:25 feed_check:213] Checking 'check_mal'
[I 230921 15:44:25 feed_check:42] Requesting https://purarue.xyz/feed_api/data/?offset=0&order_by=when&sort=desc&limit=50&ftype=anime,anime_episode
Expired: mpv.history
removed '/home/username/.local/share/evry/data/my-feed-index-bg'
2023-09-21T15-44-35:bg-feed-index:running my_feed index...
Indexing...
```

This also has the upside of updating my local data whenever there are any changes to the data sources, which means any scripts using the corresponding [`HPI`](https://github.com/purarue/HPI) modules also stay up to date.
