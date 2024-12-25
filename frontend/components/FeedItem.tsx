import { IconProp } from "@fortawesome/fontawesome-svg-core"
import {
  faBook,
  faChessKnight,
  faFilm,
  faGamepad,
  faLink,
  faMusic,
  faRecordVinyl,
} from "@fortawesome/free-solid-svg-icons"
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome"
import dayjs, { unix } from "dayjs"
import relativeTime from "dayjs/plugin/relativeTime"
import React, { useState } from "react"

import PrefsConsumer, { Prefs } from "../lib/prefs"
import styles from "../styles/Index.module.css"
import Image from "./Image"
dayjs.extend(relativeTime)

export type FeedItemStruct = {
  id: string
  ftype: string
  title: string
  score: number | null
  subtitle: string | null
  creator: string | null
  part: number | null
  subpart: number | null
  collection: string | null
  when: number
  release_date: string | null
  image_url: string | null
  url: string | null
  data: any
  flags: string[]
}
interface FeedGridProps {
  data: FeedItemStruct[]
}

export const FeedGrid: React.FC<FeedGridProps> = ({ data }: FeedGridProps) => {
  return (
    <div className={styles.grid}>
      {data.map((feedItem: FeedItemStruct) => {
        return (
          <div key={feedItem.id} data-id={feedItem.id} className={styles.card}>
            <FeedBody item={feedItem} />
          </div>
        )
      })}
    </div>
  )
}

interface CardFooterProps {
  dt: number
  score?: number | null
  releaseDate?: string | null
}

const CardFooter: React.FC<CardFooterProps> = ({ dt, score, releaseDate }: CardFooterProps) => {
  const sc = score ?? null
  return (
    <PrefsConsumer>
      {(prefs: Prefs) => {
        const d = unix(dt)
        const ds = prefs.dateAbsolute ? d.format("YYYY-MM-DD hh:mm A") : d.fromNow()
        return (
          <>
            {prefs.showReleaseDate && releaseDate && <div>Release Date: {releaseDate}</div>}
            <div className={styles.cardFooter}>
              <div>{ds}</div>
              {score && prefs.hideScore === false && (
                <div className={styles.footerScore}>{`${sc}/10`}</div>
              )}
            </div>
          </>
        )
      }}
    </PrefsConsumer>
  )
}

CardFooter.displayName = "Card Footer"

type CardImageProps = {
  src: string | null
  alt: string
  minWidth?: string
  minHeight?: string
  flags: string[]
}

const CardImage: React.FC<CardImageProps> = ({
  src,
  alt,
  flags,
  minWidth,
  minHeight,
}: CardImageProps) => {
  const [isBlurred, setIsBlurred] = useState(true)
  if (!src) {
    return <></>
  }
  let uMinWidth = minWidth ?? "15"
  let uMinHeight = minHeight ?? "12"
  let shouldBeBlurred = false

  for (const flag of flags) {
    if (flag === "i_poster") {
      uMinHeight = "24"
    } else if (flag === "i_still") {
      uMinWidth = "20"
    } else if (flag === "i_blur") {
      shouldBeBlurred = true
    }
  }

  return (
    <div
      style={{
        minWidth: `${uMinWidth}rem`,
        minHeight: `${uMinHeight}rem`,
        marginTop: "2px",
        marginBottom: "2px",
        width: "100%",
        height: "100%",
        position: "relative",
        filter: shouldBeBlurred && isBlurred ? "blur(30px)" : undefined,
        cursor: shouldBeBlurred ? "pointer" : undefined,
      }}
      title={shouldBeBlurred ? (isBlurred ? "Unblur" : "Blur") : undefined}
      onClick={() => setIsBlurred((prev) => !prev)}
    >
      <Image src={src} alt={alt} layout="fill" objectFit="contain" unoptimized />
    </div>
  )
}

const padPart = (part: number): string => {
  return String(part).padStart(2, "0")
}

type CardHeaderProps = {
  title: string
  icon: IconProp
  link: string | null
}

const CardHeader: React.FC<CardHeaderProps> = ({ title, icon, link }) => {
  return (
    <div className={styles.cardHeader}>
      <p className={styles.title}>
        <FontAwesomeIcon className={styles.logo} icon={icon} />
        {title}
      </p>
      {link && (
        <a
          className={styles.iconLink}
          href={link}
          aria-label="link"
          title="link"
          target="_blank"
          rel="noreferrer"
        >
          <FontAwesomeIcon icon={faLink} />
        </a>
      )}
    </div>
  )
}

CardHeader.displayName = "Card Header"

interface FeedBodyProps {
  item: FeedItemStruct
}

export const FeedBody: React.FC<FeedBodyProps> = React.memo(({ item }: FeedBodyProps) => {
  if (item.ftype === "listen") {
    return (
      <div className={styles.cardFlexBody}>
        <CardHeader title={item.title} icon={faMusic} link={item.url} />
        <p className={styles.subtitle}>{item.creator}</p>
        <p className={styles.subtitle}>{item.subtitle}</p>
        <CardFooter dt={item.when} releaseDate={item.release_date} />
      </div>
    )
  } else if (item.ftype === "game_achievement") {
    return (
      <div className={styles.cardFlexBody}>
        <CardHeader title={item.title} icon={faGamepad} link={item.url} />
        <CardImage src={item.image_url} alt={item.title} flags={item.flags} />
        <p className={styles.subtitle}>{item.subtitle}</p>
        <CardFooter dt={item.when} releaseDate={item.release_date} />
      </div>
    )
  } else if (item.ftype === "osrs_achievement") {
    return (
      <div className={styles.cardFlexBody}>
        <CardHeader title={item.title} icon={faGamepad} link={item.url} />
        <CardImage src={item.image_url} alt={item.title} flags={item.flags} />
        <p className={styles.subtitle}>{item.subtitle}</p>
        <CardFooter dt={item.when} />
      </div>
    )
  } else if (item.ftype === "game") {
    return (
      <div className={styles.cardFlexBody}>
        <CardHeader title={item.title} icon={faGamepad} link={item.url} />
        <CardImage src={item.image_url} alt={item.title} flags={item.flags} />
        <p className={styles.subtitle}>{item.subtitle}</p>
        <CardFooter dt={item.when} score={item.score} releaseDate={item.release_date} />
      </div>
    )
  } else if (item.ftype === "chess") {
    return (
      <div className={styles.cardFlexBody}>
        <CardHeader title={item.title} icon={faChessKnight} link={item.url} />
        <div className={styles.chessSvg} dangerouslySetInnerHTML={{ __html: item.data.svg }}></div>
        <p className={styles.subtitle}>{item.subtitle}</p>
        <CardFooter dt={item.when} />
      </div>
    )
  } else if (
    item.ftype === "trakt_history_episode" ||
    item.ftype == "trakt_history_movie" ||
    item.ftype == "trakt_show" ||
    item.ftype == "trakt_movie"
  ) {
    let seasonData = ""
    if (item.part && item.subpart) {
      seasonData = `S${padPart(item.part)}E${padPart(item.subpart)}`
    }
    const sc = item.ftype.includes("history") ? null : item.score
    return (
      <div className={styles.cardFlexBody}>
        <CardHeader title={item.title} icon={faFilm} link={item.url} />
        <CardImage src={item.image_url} alt={item.title} flags={item.flags} />
        <p className={styles.subtitle}>{item.subtitle}</p>
        {seasonData.length ? <p className={styles.subtitle}>{seasonData}</p> : null}
        <CardFooter dt={item.when} score={sc} releaseDate={item.release_date} />
      </div>
    )
  } else if (item.ftype === "album") {
    let release_year = ""
    if (item.release_date) {
      release_year = ` (${item.release_date.split("-")[0]})`
    }
    return (
      <div className={styles.cardFlexBody}>
        <CardHeader title={`${item.title}${release_year}`} icon={faRecordVinyl} link={item.url} />
        <CardImage src={item.image_url} alt={item.title} minHeight="15" flags={item.flags} />
        <p className={styles.subtitle}>{item.subtitle}</p>
        <CardFooter dt={item.when} score={item.score} releaseDate={item.release_date} />
      </div>
    )
  } else if (
    item.ftype === "anime_episode" ||
    item.ftype == "anime" ||
    item.ftype == "manga_chapter" ||
    item.ftype == "manga"
  ) {
    const sc = item.ftype == "anime" || item.ftype == "manga" ? item.score : null
    const icon = item.ftype.startsWith("anime") ? faFilm : faBook
    return (
      <div className={styles.cardFlexBody}>
        <CardHeader title={item.title} icon={icon} link={item.url} />
        <CardImage src={item.image_url} alt={item.title} flags={item.flags} />
        {item.subtitle && <p className={styles.subtitle}>{item.subtitle}</p>}
        <CardFooter dt={item.when} score={sc} releaseDate={item.release_date} />
      </div>
    )
  } else {
    console.log("Unknown type in switch", item)
    return <div>{`Unknown type in switch: ${item.ftype}`}</div>
  }
})

FeedBody.displayName = "FeedItem"
