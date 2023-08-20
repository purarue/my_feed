package main

import (
	"bufio"
	"bytes"
	"database/sql"
	"encoding/json"
	"flag"
	"fmt"
	sqlbuilder "github.com/huandu/go-sqlbuilder"
	_ "github.com/mattn/go-sqlite3"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path"
	"strconv"
	"strings"
	"time"
)

// This is the table structure for the feeddata.sqlite database
//
// CREATE TABLE feedmodel (
// 	model_id VARCHAR NOT NULL,
// 	ftype VARCHAR NOT NULL,
// 	title VARCHAR NOT NULL,
// 	score FLOAT,
// 	subtitle VARCHAR,
// 	creator VARCHAR,
// 	part INTEGER,
// 	subpart INTEGER,
// 	collection VARCHAR,
// 	"when" INTEGER NOT NULL,
// 	release_date DATE,
// 	image_url VARCHAR,
// 	url VARCHAR,
// 	id INTEGER NOT NULL,
// 	data VARCHAR,
// 	flags VARCHAR,
// 	PRIMARY KEY (id)
// )

// rows to use for /data/ select:
// SELECT model_id, ftype, title, score, subtitle, creator, part, subpart, collection, when, release_date, image_url, url, data, flags

var RootDir string

type FeedTypes struct {
	All []string `json:"all"`
}

func ParseFeedTypes(file string) (*FeedTypes, error) {
	f, err := os.Open(file)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var ftypes FeedTypes
	err = json.NewDecoder(f).Decode(&ftypes)
	if err != nil {
		return nil, err
	}
	if len(ftypes.All) == 0 {
		return nil, fmt.Errorf("Feedtypes file %s has no 'all' field", file)
	}
	return &ftypes, nil
}

func contains(slice []string, s string) bool {
	for _, item := range slice {
		if item == s {
			return true
		}
	}
	return false
}

func sliceDifference(a, b []string) []string {
	mb := make(map[string]struct{}, len(b))
	for _, x := range b {
		mb[x] = struct{}{}
	}
	var diff []string
	for _, x := range a {
		if _, found := mb[x]; !found {
			diff = append(diff, x)
		}
	}
	return diff
}

type Config struct {
	RootDir      string
	DatabaseUri  string
	BearerSecret string
	FeedTypes    *FeedTypes
	SQLEcho      bool
	Port         int
	LogRequests  bool
}

func databaseUri() *Config {
	dbName := os.Getenv("FEED_DB_NAME")
	if dbName == "" {
		dbName = "feeddata.sqlite"
	}

	ftypesFile := os.Getenv("FEEDTYPES_FILE_NAME")
	if ftypesFile == "" {
		ftypesFile = "feedtypes.json"
	}

	var root string
	var dbpath string
	var dburi string
	var echo bool
	var port int
	var logrequests bool

	flag.StringVar(&root, "root-dir", RootDir, "Root dir for backend (where Pipfile lives)")
	flag.StringVar(&dbpath, "db-path", path.Join(RootDir, dbName), "Path to sqlite database file")
	flag.StringVar(&dburi, "db-uri", "", "Database URI (overrides db-path)")
	flag.BoolVar(&logrequests, "log-requests", false, "Log info from HTTP requests to stderr")
	flag.StringVar(&ftypesFile, "ftypes-file", path.Join(RootDir, ftypesFile), "Path to feedtypes.json file")
	flag.BoolVar(&echo, "echo", false, "Echo SQL queries")
	flag.IntVar(&port, "port", 5100, "Port to listen on")

	flag.Parse()
	if _, err := os.Stat(dbpath); os.IsNotExist(err) {
		log.Fatalf("Database file %s does not exist", dbpath)
	}

	if dburi == "" {
		dburi = "file:" + dbpath + "?cache=shared&mode=ro"
	}

	if _, err := os.Stat(ftypesFile); os.IsNotExist(err) {
		log.Fatalf("Feedtypes file %s does not exist", ftypesFile)
	}

	secret := os.Getenv("BEARER_SECRET")
	if secret == "" {
		log.Fatal("BEARER_SECRET environment variable not set. This is required to authenticate the check/recheck endpoints.")
	}

	ftypes, err := ParseFeedTypes(path.Join(RootDir, "feedtypes.json"))
	if err != nil {
		log.Fatal(err)
	}

	return &Config{
		RootDir:      root,
		DatabaseUri:  dburi,
		BearerSecret: secret,
		FeedTypes:    ftypes,
		SQLEcho:      echo,
		LogRequests:  logrequests,
		Port:         port,
	}
}

func rowCount(db *sql.DB) int {
	rows, err := db.Query("SELECT COUNT(*) FROM feedmodel")
	if err != nil {
		log.Fatal(err)
	}
	defer rows.Close()
	var count int
	for rows.Next() {
		err := rows.Scan(&count)
		if err != nil {
			log.Fatal(err)
		}
	}
	err = rows.Err()
	if err != nil {
		log.Fatal(err)
	}
	return count
}

func stringQuery(db *sql.DB, query string) []string {
	rows, err := db.Query(query)
	if err != nil {
		log.Fatal(err)
	}
	defer rows.Close()
	var strings []string
	for rows.Next() {
		var s string
		err := rows.Scan(&s)
		if err != nil {
			log.Fatal(err)
		}
		strings = append(strings, s)
	}
	err = rows.Err()
	if err != nil {
		log.Fatal(err)
	}
	return strings
}

func modelIds(db *sql.DB) []string {
	return stringQuery(db, "SELECT model_id FROM feedmodel")
}

func feedTypes(db *sql.DB) []string {
	return stringQuery(db, "SELECT DISTINCT(ftype) FROM feedmodel")
}

// - `pipenv run cli update-db` to update the database whenever pinged to do so
// - `pipenv run cli update-db --delete-db` to delete the database and create a new one (the equivalent of FEED_REINDEX=1 from the [`index`](../index) script)
func shellPipenv(deleteDatabase bool, rootDir string) string {
	// change to the root directory
	currentDir, err := os.Getwd()
	if err != nil {
		log.Fatal(err)
	}
	os.Chdir(rootDir)
	defer os.Chdir(currentDir)

	tempfile, err := os.CreateTemp("", "feed-count-")
	if err != nil {
		log.Fatal(err)
	}

	cmd := exec.Command("pipenv", "run", "cli", "update-db", "-C", tempfile.Name())
	if deleteDatabase {
		cmd.Args = append(cmd.Args, "--delete-db")
	}

	stderr, err := cmd.StderrPipe()
	if err != nil {
		log.Fatal(err)
	}

	err = cmd.Start()
	go func() {
		for {
			errScanner := bufio.NewScanner(stderr)
			for errScanner.Scan() {
				log.Println(errScanner.Text())
			}
		}
	}()

	log.Println("Waiting for database update to finish...")
	err = cmd.Wait()

	if err != nil {
		log.Fatal(err)
	}

	log.Println("Finished updating database")
	// read number of rows from the tempfile
	tf, err := os.Open(tempfile.Name())
	if err != nil {
		log.Fatal(err)
	}
	tempfileReader := bufio.NewReader(tf)
	// read entire file
	b, err := io.ReadAll(tempfileReader)
	if err != nil {
		log.Fatal(err)
	}
	// convert to string
	s := string(b)
	return strings.TrimSpace(s)
}

func auth(w *http.ResponseWriter, r *http.Request, bearerSecret string) bool {
	bearer := r.Header.Get("token")
	if bearer == "" {
		http.Error(*w, "token header missing", http.StatusUnauthorized)
		return false
	}
	if bearer != bearerSecret {
		http.Error(*w, "Invalid bearer token", http.StatusUnauthorized)
		return false
	}
	return true
}

// function which takes a min and max and makes sure user input is between them
func clamp(min int, max *int, val int) int {
	if val < min {
		return min
	}
	if max != nil && val > *max {
		return *max
	}
	return val
}

// https://go.dev/doc/faq#convert_slice_of_interface
func stringToInterface(s []string) []interface{} {
	i := make([]interface{}, len(s))
	for j, v := range s {
		i[j] = v
	}
	return i
}

type OrderBy string

const (
	When    OrderBy = "when"
	Score           = "score"
	Release         = "release"
)

type Sort string

const (
	Ascending  Sort = "asc"
	Descending      = "desc"
)

func parseEnumQueryParam(name string, query *url.Values, defaultValue string, allowed []string) (string, error) {
	val := query.Get(name)
	if val == "" {
		return defaultValue, nil
	}
	if contains(allowed, val) {
		return val, nil
	}
	return "", fmt.Errorf("Invalid %s: %s", name, val)
}

func parseIntegerQueryParam(name string, query *url.Values, defaultValue int, min int, max *int) (int, error) {
	val := query.Get(name)
	if val == "" {
		return defaultValue, nil
	}
	i, err := strconv.Atoi(val)
	if err != nil {
		return 0, fmt.Errorf("Invalid %s: %s", name, val)
	}
	return clamp(min, max, i), nil
}

var maxLimit int = 500

type FeedItem struct {
	ModelId     string                 `json:"model_id"`
	FeedType    string                 `json:"ftype"`
	When        int64                  `json:"when"`
	Title       string                 `json:"title"`
	Score       *float32               `json:"score"`
	Subtitle    *string                `json:"subtitle"`
	Creator     *string                `json:"creator"`
	Part        *int                   `json:"part"`
	Subpart     *int                   `json:"subpart"`
	Collection  *string                `json:"collection"`
	ReleaseDate *string                `json:"release_date"` // this is a date, but it gets converted to a string by sqlite3 lib
	ImageUrl    *string                `json:"image_url"`
	Url         *string                `json:"url"`
	Data        map[string]interface{} `json:"data"`
	Flags       []string               `json:"flags"`
}

func main() {
	config := databaseUri()

	isUpdating := false

	var db *sql.DB
	var err error

	db, err = sql.Open("sqlite3", config.DatabaseUri)
	if err != nil {
		log.Fatal(err)
	}
	defer func() {
		err := db.Close()
		if err != nil {
			log.Fatal(err)
		}
	}()

	reconnectToDb := func() {
		log.Println("Reconnecting to database...")
		if db != nil {
			log.Println("Closing existing connection...")
			err := db.Close()
			if err != nil {
				log.Fatal(err)
			}
		}
		db, err = sql.Open("sqlite3", config.DatabaseUri)
		if err != nil {
			log.Fatal(err)
		}
		log.Println("Reconnected to database")
	}

	waitTillDbIsReady := func() {
		for {
			if isUpdating {
				log.Println("Waiting for update to finish...")
				time.Sleep(1 * time.Second)
				continue
			}

			err := db.Ping()
			if err != nil {
				log.Println("Waiting for database to be ready...")
				time.Sleep(1 * time.Second)
			} else {
				break
			}
		}
	}

	// Get the feed data
	count := rowCount(db)
	log.Printf("feedmodel table contains %d rows\n", count)

	// Start the web server
	http.HandleFunc("/check", func(w http.ResponseWriter, r *http.Request) {
		if !auth(&w, r, config.BearerSecret) {
			return
		}
		if isUpdating {
			log.Fatalf("Already updating, ignoring request")
			return
		}
		isUpdating = true

		if config.LogRequests {
			log.Println("Running check...")
		}

		added := shellPipenv(false, config.RootDir)
		fmt.Fprintf(w, "Added %s\n", added)
		isUpdating = false
		reconnectToDb()
	})
	http.HandleFunc("/recheck", func(w http.ResponseWriter, r *http.Request) {
		if !auth(&w, r, config.BearerSecret) {
			return
		}
		if isUpdating {
			log.Fatalf("Already updating, ignoring request")
			return
		}

		isUpdating = true

		if config.LogRequests {
			log.Println("Running recheck...")
		}

		added := shellPipenv(true, config.RootDir)
		fmt.Fprintf(w, "Added %s\n", added)

		fmt.Fprintf(w, "Recreated database...\n")
		isUpdating = false
		reconnectToDb()
	})

	http.HandleFunc("/data/ids", func(w http.ResponseWriter, r *http.Request) {
		waitTillDbIsReady()
		if config.LogRequests {
			log.Println("Running data/ids...")
		}
		ids := modelIds(db)
		if config.LogRequests {
			log.Printf("Found %d ids\n", len(ids))
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ids)
	})

	http.HandleFunc("/data/types", func(w http.ResponseWriter, r *http.Request) {
		waitTillDbIsReady()
		types := feedTypes(db)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(types)
	})

	http.HandleFunc("/data/", func(w http.ResponseWriter, r *http.Request) {
		waitTillDbIsReady()

		// parse query params
		qrParams := r.URL.Query()
		offset, err := parseIntegerQueryParam("offset", &qrParams, 0, 0, nil)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		limit, err := parseIntegerQueryParam("limit", &qrParams, 100, 1, &maxLimit)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		orderBy, err := parseEnumQueryParam("order_by", &qrParams, string(When), []string{string(When), string(Score), string(Release)})
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		sort, err := parseEnumQueryParam("sort", &qrParams, string(Descending), []string{string(Ascending), string(Descending)})
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		var filterFtypes []string
		ftypeRaw := qrParams.Get("ftype")
		if ftypeRaw != "" {
			if strings.Contains(ftypeRaw, ",") {
				filterFtypes = strings.Split(ftypeRaw, ",")
			} else {
				filterFtypes = []string{ftypeRaw}
			}
		}

		// validate to make sure all ftypes are valid
		for _, ftype := range filterFtypes {
			if !contains(config.FeedTypes.All, ftype) {
				http.Error(w, fmt.Sprintf("Invalid ftype value %s", ftype), http.StatusBadRequest)
				return
			}
		}

		query := r.URL.Query().Get("query")
		if strings.TrimSpace(query) == "" {
			query = ""
		}

		if config.LogRequests {
			log.Printf("Running data/ with offset '%d', limit '%d', orderBy '%s', sort '%s', ftype filter %+v, query '%s'\n", offset, limit, orderBy, sort, filterFtypes, query)
		}

		sb := sqlbuilder.NewSelectBuilder()
		sb.Select("model_id, ftype, title, score, subtitle, creator, part, subpart, collection, `when`, release_date, image_url, url, data, flags")
		sb.From("feedmodel")
		if len(filterFtypes) > 0 {
			sb.Where(sb.In("ftype", stringToInterface(filterFtypes)...))
		}

		if query != "" {
			queryWild := "%" + query + "%"
			// by default, sqlite is case insensitive for ascii characters, but case sensitive for unicode characters
			// probably good enough unless I find some really weird edge case
			sb.Where(sb.Or(
				sb.Like("title", queryWild),
				sb.Like("subtitle", queryWild),
				sb.Like("creator", queryWild),
				sb.Like("collection", queryWild),
				sb.Like("model_id", queryWild),
			))
		}

		// Note: 'when' is a reserved keyword in sqlite, so we have to use backticks

		if orderBy == string(Score) {
			sb.Where(sb.IsNotNull("score"))
			sb.Where("score > 0")
			if sort == string(Descending) {
				sb.OrderBy("score").Desc()
			} else {
				sb.OrderBy("score").Asc()
			}
			// querybuilder only supports one sort sort order
			// so we have to do this manually incase order_by=score&sort=asc
			sb.SQL(", `when` DESC")
		} else if orderBy == string(When) {
			if sort == string(Descending) {
				sb.OrderBy("`when`").Desc()
			} else {
				sb.OrderBy("`when`").Asc()
			}
		} else {
			sb.Where(sb.IsNotNull("release_date"))
			if sort == string(Descending) {
				sb.OrderBy("release_date").Desc()
			} else {
				sb.OrderBy("release_date").Asc()
			}
		}

		sb.Limit(limit).Offset(offset)

		sql, args := sb.Build()
		if config.SQLEcho {
			log.Printf("QUERY: %s\n", sql)
			// json stringify args
			var buf bytes.Buffer
			json.NewEncoder(&buf).Encode(args)
			log.Printf("VARS: %s", buf.String())
		}

		rows, err := db.Query(sql, args...)
		if err != nil {
			log.Printf("Error querying database: %s\n", err)
			http.Error(w, "Error querying database", http.StatusInternalServerError)
			return
		}
		defer rows.Close()

		var models []FeedItem = []FeedItem{}
		for rows.Next() {
			var model FeedItem
			var rawFlags *[]byte
			var rawData *[]byte
			var releaseDate *string
			// model_id, ftype, title, score, subtitle, creator, part, subpart, collection, when, release_date, image_url, url, data, flags
			err := rows.Scan(&model.ModelId, &model.FeedType, &model.Title, &model.Score, &model.Subtitle, &model.Creator, &model.Part, &model.Subpart, &model.Collection, &model.When, &releaseDate, &model.ImageUrl, &model.Url, &rawData, &rawFlags)
			if err != nil {
				log.Printf("Error scanning row: %s\n", err)
				http.Error(w, "Error retrieving row from database", http.StatusInternalServerError)
				return
			}

			if releaseDate != nil {
				// only retain the date portion, not the time
				model.ReleaseDate = &strings.Split(*releaseDate, "T")[0]
			}

			// data is stored as nil if not present, else a stringified json object
			// parse it back into a map
			if rawData != nil {
				err = json.Unmarshal(*rawData, &model.Data)
				if err != nil {
					log.Printf("Error unmarshalling data: %s\n", err)
					http.Error(w, "Error unmarshalling data field", http.StatusInternalServerError)
					return
				}
			} else {
				// set to empty map
				model.Data = make(map[string]interface{})
			}

			// flags is stored as nil if not present, else a stringified json array
			// parse it back into an array
			if rawFlags != nil {
				err = json.Unmarshal(*rawFlags, &model.Flags)
				if err != nil {
					log.Printf("Error unmarshalling flags: %s\n", err)
					http.Error(w, "Error unmarshalling flags field", http.StatusInternalServerError)
					return
				}
			} else {
				// set to empty array
				model.Flags = []string{}
			}

			models = append(models, model)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(models)
	})

	log.Fatal(http.ListenAndServe(fmt.Sprintf(":%d", config.Port), nil))
}
