import Head from "next/head"
import React, { Component, ErrorInfo, ReactNode } from "react"

import styles from "../styles/Index.module.css"

interface Props {
  children: ReactNode
}

interface State {
  hasError: boolean
  errors: Error[]
}

class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    errors: [],
  }

  public static getDerivedStateFromError(_: Error): State {
    // Update state so the next render will show the fallback UI.
    return { hasError: true, errors: [] }
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Uncaught error:", error, errorInfo)
    this.setState((oldState) => ({
      ...oldState,
      errors: [...oldState.errors, error],
    }))
  }

  public render() {
    if (this.state.hasError) {
      return (
        <>
          <Head>
            <meta httpEquiv="refresh" content="20" />
          </Head>
          <div className={styles.container}>
            <main className={styles.main}>
              <h3>{`Error: Website is currently being updated, this page will automatically refresh when its done`}</h3>
              {this.state.errors.map((error, index) => (
                <div key={index}>{error.message}</div>
              ))}
            </main>
          </div>
        </>
      )
    }

    return this.props.children
  }
}

export default ErrorBoundary
