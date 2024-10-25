import styles from "../styles/Index.module.css"

const About: React.FC = () => {
  return (
    <div className={styles.about}>
      <a
        style={{ marginLeft: "20px" }}
        className={styles.link}
        href="https://github.com/purarue/my_feed"
      >
        Source Code
      </a>
      <p>Any Images here are owned by the respective services:</p>
      <ul>
        <li>
          Listens (Songs), using{" "}
          <a className={styles.link} href="https://listenbrainz.org/">
            ListenBrainz
          </a>
        </li>
        <li>
          Game Achievements from{" "}
          <a href="https://steamcommunity.com/" className={styles.link}>
            Steam
          </a>
        </li>
        <li>
          Album art from{" "}
          <a href="https://discogs.com/" className={styles.link}>
            Discogs
          </a>
        </li>
        <li>
          Anime/Manga from{" "}
          <a href="https://myanimelist.net/" className={styles.link}>
            MyAnimeList
          </a>
        </li>
        <li>
          Games from{" "}
          <a href="https://www.grouvee.com/" className={styles.link}>
            Grouvee
          </a>
          , with images from{" "}
          <a href="https://www.giantbomb.com/" className={styles.link}>
            GiantBomb
          </a>
        </li>
        <li>
          Movies/TV Shows/Episodes -{" "}
          <a className={styles.link} href="https://trakt.tv/">
            Trakt
          </a>
          , using{" "}
          <a className={styles.link} href="https://www.themoviedb.org/">
            TMDB
          </a>{" "}
          (This product uses the TMDB API but is not endorsed or certified by TMDB)
        </li>
      </ul>
    </div>
  )
}

export default About
