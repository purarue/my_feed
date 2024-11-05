package main

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	sqlbuilder "github.com/huandu/go-sqlbuilder"
	"io"
	"log"
	"os"
	"path"
	"sort"
	"strings"
	"time"
)

// serialized to json and sent to the client
type FeedItem struct {
	Id          string                 `json:"id"`
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

func (f *FeedItem) validate() error {
	if f.Id == "" {
		return fmt.Errorf("id is required")
	}
	if f.FeedType == "" {
		return fmt.Errorf("ftype is required")
	}
	if f.When == 0 {
		return fmt.Errorf("when is required")
	}
	if f.Title == "" {
		return fmt.Errorf("title is required")
	}
	return nil
}

type ModelSet map[string]bool

func (s ModelSet) add(modelId string) {
	s[modelId] = true
}

func (s ModelSet) remove(modelId string) {
	delete(s, modelId)
}

func (s ModelSet) has(modelId string) bool {
	_, ok := s[modelId]
	return ok
}

func (s ModelSet) missing(modelId string) bool {
	return !s.has(modelId)
}

func modelSet(db *sql.DB) ModelSet {
	s := make(ModelSet)
	for _, modelId := range modelIds(db) {
		s.add(modelId)
	}
	return s
}

func initDb(db *sql.DB) {
	// check if table exists
	// if it does, return
	if _, err := db.Exec("SELECT id FROM feedmodel LIMIT 1"); err == nil {
		log.Println("Database already initialized")
		return
	}

	log.Println("Initializing database")

	// otherwise, create table
	_, err := db.Exec(`CREATE TABLE feedmodel (
	id VARCHAR NOT NULL,
	ftype VARCHAR NOT NULL, 
	title VARCHAR NOT NULL, 
	score FLOAT, 
	subtitle VARCHAR, 
	creator VARCHAR, 
	part INTEGER, 
	subpart INTEGER, 
	collection VARCHAR, 
	"when" INTEGER NOT NULL, 
	release_date DATE, 
	image_url VARCHAR, 
	url VARCHAR, 
	data VARCHAR, 
	flags VARCHAR, 
	PRIMARY KEY (id));
	CREATE INDEX ix_feedmodel_id ON feedmodel (id);
	CREATE INDEX ix_feedmodel_when ON feedmodel ("when");
	CREATE INDEX ix_feedmodel_ftype ON feedmodel (ftype);
	CREATE INDEX ix_feedmodel_score ON feedmodel (score);
	`)
	if err != nil {
		log.Fatal(err)
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
	var strings []string = make([]string, 0)
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
	return stringQuery(db, "SELECT id FROM feedmodel")
}

func feedTypes(db *sql.DB) []string {
	return stringQuery(db, "SELECT DISTINCT(ftype) FROM feedmodel")
}

func clearDatabase(db *sql.DB) (int, error) {
	log.Println("Clearing database")
	resp, err := db.Exec("DELETE FROM feedmodel")
	if err != nil {
		return 0, err
	}
	affected, err := resp.RowsAffected()
	if err != nil {
		return 0, err
	}
	return int(affected), nil
}

func listJsonFiles(config *Config) []string {
	files, err := os.ReadDir(config.DataDir)
	if err != nil {
		log.Fatal(err)
	}
	var jsonFiles []string
	for _, f := range files {
		if strings.HasSuffix(f.Name(), ".json") {
			jsonFiles = append(jsonFiles, path.Join(config.DataDir, f.Name()))
		}
	}
	return jsonFiles
}

func clearDataDir(config *Config) error {
	files, err := os.ReadDir(config.DataDir)
	if err != nil {
		return err
	}
	for _, f := range files {
		if strings.HasSuffix(f.Name(), ".json") {
			file := path.Join(config.DataDir, f.Name())
			log.Printf("removing %s\n", file)
			err := os.Remove(file)
			if err != nil {
				return err
			}
		}
	}
	return nil
}

// https://www.sqlite.org/pragma.html#pragma_wal_checkpoint
func truncateWal(db *sql.DB) error {
	log.Println("Truncating WAL...")
	_, err := db.Exec("PRAGMA wal_checkpoint(TRUNCATE)")
	return err
}

func updateDatabaseFromJsonFiles(db *sql.DB, config *Config) (int, error) {
	// load all json files
	jsonFiles := listJsonFiles(config)

	modelSet := modelSet(db)
	var funcErr error

	totalAdded := 0
	for len(jsonFiles) > 0 {
		log.Printf("loading data from %s\n", jsonFiles[0])
		added, err := loadFeedItemsFromFile(db, jsonFiles[0], &modelSet)
		if err != nil {
			funcErr = err
			// unlink file, since we couldn't load it
			// if we don't do this, we'll keep trying to load it
			log.Printf("error loading %s, removing file: %s\n", jsonFiles[0], err.Error())
			os.Remove(jsonFiles[0])
		}
		totalAdded += added
		jsonFiles = jsonFiles[1:]
	}

	// list JSON files and sort by name
	// remove all files but the newest one

	jsonFiles = listJsonFiles(config)
	sort.Sort(sort.StringSlice(jsonFiles))
	if len(jsonFiles) > 1 {
		for _, jsonFile := range jsonFiles[:len(jsonFiles)-1] {
			log.Printf("Pruning old file %s\n", jsonFile)
			os.Remove(jsonFile)
		}
	}
	return totalAdded, funcErr
}

func serializeFlags(flags []string) (*string, error) {
	if len(flags) == 0 {
		return nil, nil
	}
	jsonFlags, _ := json.Marshal(flags)
	s := string(jsonFlags)
	return &s, nil
}

func serializeData(data map[string]interface{}) (*string, error) {
	if len(data) == 0 {
		return nil, nil
	}
	jsonData, _ := json.Marshal(data)
	s := string(jsonData)
	return &s, nil
}

func loadFeedItemsFromFile(db *sql.DB, filename string, modelSet *ModelSet) (int, error) {
	added := 0

	// open file
	file, err := os.Open(filename)
	if err != nil {
		return 0, err
	}
	defer file.Close()

	lines := 0
	tx, err := db.Begin()
	if err != nil {
		return 0, err
	}
	defer tx.Rollback() // The rollback will be ignored if the tx has been committed later in the function.

	dc := json.NewDecoder(file)
	for {
		var item FeedItem
		// decodes a single JSON value (object, array, string, number, etc)
		// so stops at the end of the line when the object ends
		if err := dc.Decode(&item); err == io.EOF {
			break
		} else if err != nil {
			return 0, err
		}
		lines += 1

		if err = item.validate(); err != nil {
			return 0, err
		}

		// this modelId is already in the db, so skip it
		if modelSet.has(item.Id) {
			continue
		}

		flag, err := serializeFlags(item.Flags)
		if err != nil {
			return 0, err
		}
		data, err := serializeData(item.Data)
		if err != nil {
			return 0, err
		}
		// parse the release date into a time.Time (currently they are like 2023-01-01)
		var releaseDate *time.Time
		if item.ReleaseDate != nil {
			rd, err := time.Parse(time.DateOnly, *item.ReleaseDate)
			if err != nil {
				return 0, err
			}
			releaseDate = &rd
		}

		_, err = tx.Exec("INSERT INTO feedmodel (id, ftype, title, score, subtitle, creator, part, subpart, collection, `when`, release_date, image_url, url, data, flags) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);", item.Id, item.FeedType, item.Title, item.Score, item.Subtitle, item.Creator, item.Part, item.Subpart, item.Collection, item.When, releaseDate, item.ImageUrl, item.Url, data, flag)
		if err != nil {
			return 0, err
		}
		added += 1
		modelSet.add(item.Id)
	}
	err = tx.Commit()
	if err != nil {
		return 0, err
	}
	log.Printf("Checked %d lines from %s\n", lines, filename)
	return added, nil
}

func queryData(
	db *sql.DB,
	config *Config,
	query string,
	filterFtypes []string,
	orderBy OrderBy,
	sort Sort,
	limit int,
	offset int,
) ([]FeedItem, error) {
	sb := sqlbuilder.NewSelectBuilder()
	sb.Select("id, ftype, title, score, subtitle, creator, part, subpart, collection, `when`, release_date, image_url, url, data, flags")
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
			sb.Like("id", queryWild),
		))
	}

	// Note: 'when' is a reserved keyword in sqlite, so we have to use backticks

	if orderBy == Score {
		sb.Where(sb.IsNotNull("score"))
		sb.Where("score > 0")
		if sort == Descending {
			sb.OrderBy("score").Desc()
		} else {
			sb.OrderBy("score").Asc()
		}
		// querybuilder only supports one sort sort order
		// so we have to do this manually in case order_by=score&sort=asc
		sb.SQL(", `when` DESC")
	} else if orderBy == When {
		if sort == Descending {
			sb.OrderBy("`when`").Desc()
		} else {
			sb.OrderBy("`when`").Asc()
		}
	} else {
		sb.Where(sb.IsNotNull("release_date"))
		if sort == Descending {
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
		return nil, errors.New("Error querying database")
	}
	defer rows.Close()

	var models []FeedItem = []FeedItem{}

	for rows.Next() {
		var model FeedItem
		var rawFlags *[]byte
		var rawData *[]byte
		var releaseDate *string
		// id, ftype, title, score, subtitle, creator, part, subpart, collection, when, release_date, image_url, url, data, flags
		err := rows.Scan(&model.Id, &model.FeedType, &model.Title, &model.Score, &model.Subtitle, &model.Creator, &model.Part, &model.Subpart, &model.Collection, &model.When, &releaseDate, &model.ImageUrl, &model.Url, &rawData, &rawFlags)
		if err != nil {
			log.Printf("Error scanning row: %s\n", err)
			return nil, errors.New("Error scanning row")
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
				return nil, errors.New("Error unmarshalling data field")
			}
		} else {
			// set to empty map so frontend doesn't have to check for nil
			model.Data = make(map[string]interface{})
		}

		// flags is stored as nil if not present, else a stringified json array
		// parse it back into an array
		if rawFlags != nil {
			err = json.Unmarshal(*rawFlags, &model.Flags)
			if err != nil {
				log.Printf("Error unmarshalling flags: %s\n", err)
				return nil, errors.New("Error unmarshalling flags field")
			}
		} else {
			// set to empty array
			model.Flags = []string{}
		}

		models = append(models, model)
	}
	return models, nil
}
