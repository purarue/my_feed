import {
  faCalendar,
  faClock,
  faCopy,
  faHeart,
  faHistory,
  faListOl,
  faSyncAlt,
  faTimes,
} from "@fortawesome/free-solid-svg-icons"
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome"
import type { NextPage } from "next"
import Head from "next/head"
import { useRouter } from "next/router"
import { Dispatch, SetStateAction, useEffect, useRef, useState } from "react"
import { DebounceInput } from "react-debounce-input"
import Select from "react-select"
import useSWRInfinite from "swr/infinite"

import About from "../components/About"
import { FeedGrid, FeedItemStruct } from "../components/FeedItem"
import useOnScreen from "../hooks/useOnScreen"
import { FeedItemOptions, LabelOption, OrderByOptions } from "../lib/enums"
import PrefsConsumer, { Prefs } from "../lib/prefs"
import styles from "../styles/Index.module.css"

async function fetcher(...args: any[]) {
  try {
    // @ts-ignore
    const res = await fetch(...args)
    if (!res.ok) {
      const error = new Error("An error occurred while fetching the data.")
      throw error
    }
    return res.json()
  } catch (e) {
    console.error(e)
    const error = new Error("An error occurred while fetching the data.")
    throw error
  }
}

const createQuery = (obj: any) => {
  let str = ""
  for (const key in obj) {
    if (str != "") {
      str += "&"
    }
    str += key + "=" + encodeURIComponent(obj[key])
  }
  return str
}

const attachParams = (
  baseParams: any,
  query: string,
  selectedTypes: string[],
  selectedOrder: string,
  sort: string
): any => {
  const qt = query.trim()
  if (qt.length > 0) {
    baseParams.query = qt
  }

  if (selectedOrder.length > 0) {
    baseParams.order_by = selectedOrder
    if (sort) {
      baseParams.sort = sort
    }
  }

  if (selectedTypes.length > 0) {
    baseParams.ftype = selectedTypes.join(",")
  }
  return baseParams
}

const getKey = (
  pageIndex: number,
  previousPageData: any,
  baseUrl: string,
  query: string,
  selectedTypes: string[],
  selectedOrder: string,
  sort: string,
  limit: number,
  setAtEnd: Dispatch<SetStateAction<boolean>>
) => {
  if (previousPageData && !previousPageData.length) {
    setAtEnd(true)
    return null // reached the end
  }

  let offset = 0
  if (pageIndex > 0) {
    offset = pageIndex * 100
  }

  let params: any = {
    offset: offset,
    limit: limit,
  }

  params = attachParams(params, query, selectedTypes, selectedOrder, sort)

  return `${baseUrl}?${createQuery(params)}`
}

const generateLink = (
  query: string,
  selectedTypes: string[],
  selectedOrder: string,
  sort: string
): string => {
  const baseUrl = window.location.href.split("?")[0]
  const params = attachParams({}, query, selectedTypes, selectedOrder, sort)
  return `${baseUrl}?${createQuery(params)}`
}

const baseUrl = process.env.NEXT_PUBLIC_API_URL
if (!baseUrl) {
  throw new Error(`No base URL: ${baseUrl}`)
}

const dataBase = `${baseUrl}/data/`
const paginationLimit = 100
const defaultSelectedOrder = OrderByOptions[0]

interface IndexProps {
  prefs: Prefs
}

const Page: NextPage<IndexProps> = ({ prefs }: IndexProps) => {
  const scrollRef = useRef(null)
  const isVisible = useOnScreen(scrollRef)
  const { query } = useRouter()

  const [showAttribution, setShowAttribution] = useState<boolean>(false)

  // query text
  const [queryText, setQueryText] = useState<string>("")

  // valid type labels, used in Select, sent to API
  const [selectedTypeLabels, setSelectedTypeLabels] = useState<LabelOption[]>([])

  // valid order by, used in select and sent to API
  const [selectedOrderLabel, setSelectedOrderLabel] = useState<LabelOption>(defaultSelectedOrder)

  // order sort
  const [sort, setSort] = useState<string>("desc")

  // hit the end of the data for this query
  const [atEnd, setAtEnd] = useState<boolean>(false)

  const { data, error, size, setSize, isValidating } = useSWRInfinite(
    (...args) =>
      getKey(
        ...args,
        dataBase,
        queryText,
        selectedTypeLabels.map((e) => e.value),
        selectedOrderLabel.value,
        sort,
        paginationLimit,
        setAtEnd
      ),
    fetcher
  )

  const feedItems = data ? [].concat(...data) : []
  const isLoadingInitialData = !data && !error
  const isLoadingMore =
    isLoadingInitialData || (size > 0 && data && typeof data[size - 1] === "undefined")
  const isEmpty = data?.[0]?.length === 0
  const isRefreshing = isValidating && data && data.length === size

  useEffect(() => {
    if (isVisible && !atEnd && !isRefreshing && !isLoadingInitialData && !isLoadingMore) {
      setSize(size + 1)
    }
  }, [
    isVisible,
    isRefreshing,
    atEnd,
    isLoadingMore,
    isLoadingInitialData,
    setSize,
    size,
    selectedOrderLabel,
    selectedTypeLabels,
    queryText,
  ])

  // parse query params
  useEffect(() => {
    // feed item type
    if (query.ftype) {
      const selectedTypes: string[] = (query.ftype as string).split(",")
      const validTypes: LabelOption[] = selectedTypes
        .filter((e) => FeedItemOptions.some((l) => l.value === e))
        .map((e) => FeedItemOptions.find((l) => l.value === e))
        .filter((e) => e !== undefined) as LabelOption[]
      setSelectedTypeLabels(validTypes)
    }
    // query
    if (query.query) {
      const qs = query.query.toString()
      if (qs.length > 0) {
        setQueryText(qs)
      }
    }
    // order by
    if (query.order_by) {
      const sort = OrderByOptions.find((e) => e.value === query.order_by)
      if (sort) {
        setSelectedOrderLabel(sort)
      }
    }
    // sort
    if (query.sort) {
      if (query.sort === "asc") {
        setSort("asc")
      } else if (query.sort === "desc") {
        setSort("desc")
      }
    }
  }, [query])

  // reset 'atEnd' (so new items load when you scroll down) after user changes any of the inputs
  useEffect(() => {
    setAtEnd(false)
  }, [queryText, selectedOrderLabel, selectedTypeLabels, sort])

  useEffect(() => {
    if (selectedOrderLabel.value == "release") {
      prefs.setPrefs((oldPrefs: Prefs): Prefs => {
        return {
          ...oldPrefs,
          showReleaseDate: true,
        }
      })
    }
  }, [selectedOrderLabel, prefs])

  const clear = () => {
    setQueryText("")
    setSelectedOrderLabel(defaultSelectedOrder)
    setSelectedTypeLabels([])
    setSize(0)
    setSort("desc")
  }

  const swapSort = () => {
    if (sort === "desc") {
      setSort("asc")
    } else {
      setSort("desc")
    }
  }

  const copyLink = () => {
    const url = generateLink(
      queryText,
      selectedTypeLabels.map((e) => e.value),
      selectedOrderLabel.value,
      sort
    )
    navigator.clipboard.writeText(url).then(() => {
      alert(`Copied ${url} to clipboard`)
    })
  }

  if (error) {
    return <div>{error}</div>
  }

  return (
    <>
      <Head>
        <title>feed</title>
        <meta name="description" content="my personal media feed" />
        <link rel="icon" href="https://purarue.xyz/favicon.ico" />
      </Head>
      <main className={styles.main}>
        <nav className={styles.nav}>
          <div className={styles.mainTitle}>- FEED -</div>
          <div className={styles.aboutLink}>
            <a href="#" onClick={() => setShowAttribution((toggle) => !toggle)}>
              About/Attribution
            </a>
          </div>
        </nav>
        {showAttribution && <About />}
        <div className={styles.filterBar}>
          <DebounceInput
            className={styles.queryInput}
            value={queryText}
            minLength={2}
            aria-label="Search"
            debounceTimeout={300}
            onChange={(e) => setQueryText(e.target.value)}
            placeholder="Search..."
          />
          <label>
            <Select
              value={selectedTypeLabels}
              isMulti
              instanceId="type_select"
              inputId="type_select"
              options={FeedItemOptions}
              placeholder="Filter Type..."
              aria-label="Filter Types"
              className={styles.typeSelect}
              onChange={(e) => {
                if (e) {
                  setSelectedTypeLabels(e as LabelOption[])
                }
              }}
            />
          </label>
          <label>
            <Select
              value={selectedOrderLabel}
              instanceId="order_select"
              inputId="order_select"
              aria-label="Select Order"
              options={OrderByOptions}
              className={styles.sortSelect}
              onChange={(e) => {
                if (e) {
                  setSelectedOrderLabel(e)
                }
              }}
            />
          </label>
          <div className={styles.filterButtons}>
            <div
              className={styles.filterIcon}
              title="Toggle Date Format"
              role="button"
              onClick={() => {
                // toggle on click
                prefs.setPrefs((oldPrefs: Prefs): Prefs => {
                  return {
                    ...oldPrefs,
                    dateAbsolute: !oldPrefs.dateAbsolute,
                  }
                })
              }}
            >
              <FontAwesomeIcon icon={prefs.dateAbsolute ? faClock : faHistory} />
            </div>
            <div
              role="button"
              className={styles.filterIcon}
              title="Swap Sort Order"
              onClick={swapSort}
            >
              <FontAwesomeIcon icon={faSyncAlt} />
            </div>
            <div
              role="button"
              className={styles.filterIcon}
              title="Toggle Score"
              onClick={() => {
                prefs.setPrefs((oldPrefs: Prefs): Prefs => {
                  return {
                    ...oldPrefs,
                    hideScore: !oldPrefs.hideScore,
                  }
                })
              }}
            >
              <FontAwesomeIcon icon={faListOl} />
            </div>
            <div
              role="button"
              className={styles.filterIcon}
              title="Copy Link To Clipboard"
              onClick={copyLink}
            >
              <FontAwesomeIcon icon={faCopy} />
            </div>
            <div
              role="button"
              className={styles.filterIcon}
              title="Show Release Dates"
              onClick={() => {
                prefs.setPrefs((oldPrefs: Prefs): Prefs => {
                  return {
                    ...oldPrefs,
                    showReleaseDate: !oldPrefs.showReleaseDate,
                  }
                })
              }}
            >
              <FontAwesomeIcon icon={faCalendar} />
            </div>
            <div
              role="button"
              className={styles.filterIcon}
              style={{
                color: "#f73e3e",
              }}
              title="Show Favorites"
              onClick={() => {
                setQueryText("")
                setSelectedTypeLabels([])
                setSize(0)
                setSort("desc")
                setSelectedOrderLabel(OrderByOptions[1])
              }}
            >
              <FontAwesomeIcon icon={faHeart} />
            </div>
            <div role="button" className={styles.filterIcon} title="Reset Filters" onClick={clear}>
              <FontAwesomeIcon icon={faTimes} />
            </div>
          </div>
        </div>
        <FeedGrid data={feedItems as FeedItemStruct[]} />
        <div ref={scrollRef} style={{ marginTop: "20vh" }}>
          {atEnd ? (
            <div>
              {"no more items" + (selectedOrderLabel.value === "score" ? " with scores, " : ", ")}
              <a className={styles.aboutLink} href="#" onClick={clear}>
                {"reset?"}
              </a>
            </div>
          ) : !isEmpty && isLoadingMore ? (
            "loading..."
          ) : (
            ""
          )}
        </div>
      </main>
    </>
  )
}

const Index = () => {
  return (
    <PrefsConsumer>
      {(prefs: Prefs) => {
        return <Page prefs={prefs} />
      }}
    </PrefsConsumer>
  )
}

export default Index
