To build, this requires `go` (I set minimum version to `v1.18`, but you can change the `go.mod` file and try with something lower as well).

To build `sqlite3-go` requires C, so you need `gcc` on your `$PATH`

```bash
./build  # this takes a while to build
```

This will create a `./backend/main` executable, which has the `backend` folder path embedded in it for configuration. Hence, additional flags to configure should not be needed, but you can always customize if needed:

```
Usage of ./main:
  -data-dir string
    	Data directory for JSON files (default "/home/username/Repos/my_feed/backend/data")
  -db-path string
    	Path to sqlite database file (default "/home/username/Repos/my_feed/backend/feeddata.sqlite")
  -db-uri string
    	Database URI (overrides db-path)
  -echo
    	Echo SQL queries
  -ftypes-file string
    	Path to feedtypes.json file (default "/home/username/Repos/my_feed/backend/feedtypes.json")
  -log-requests
    	Log info from HTTP requests to stderr
  -port int
    	Port to listen on (default 5100)
  -root-dir string
    	Root dir for backend (same dir as 'build' script) (default "/home/username/Repos/my_feed/backend")
```

You can check the [`index`](./index) script for how I use this, but generally:

To index:

1. Hit the `/data/ids` endpoint to get a list of all currently known feed ids
2. `my_feed index -E ./file/ids.json -o /tmp/tmpfile.json` to compute any new feed ids and write to `/tmp/tmpfile.json`
3. `scp /tmp/tmpfile.json <server>:code/my_feed/backend/data/tmpfile.json`
4. `curl -H "token: <token>" server.com/check` to check for new files and update the database

Sometimes I change how a feed item works, or there are errors syncing, so once a week I just reindex everything

Also, I choose to exclude some data sources which take longer to update, so those just update once a week. That makes the regular indexing much faster

To reindex, same as above, just dont request the `/data/ids/` file

Before `scp`ing the JSON file up, `curl ... server.com/clear-data-dir` which removes all the old files in the data directory

And then `curl ... server.com/recheck` to reindex everything (reindexing deletes every item in the database and then re-adds them from the JSON file)
