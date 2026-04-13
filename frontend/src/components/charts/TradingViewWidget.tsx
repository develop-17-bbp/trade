import { useEffect, useRef, memo } from 'react'

interface Props {
  symbol?: string
  interval?: string
  theme?: 'dark' | 'light'
  height?: number
  showToolbar?: boolean
}

/**
 * TradingView Advanced Chart Widget — Real-time professional charts
 * Embeds TradingView's free chart widget with live data from their servers.
 * Shows the EXACT same data that Robinhood and all exchanges use.
 */
function TradingViewWidget({
  symbol = 'KRAKEN:BTCUSD',
  interval = '60',
  theme = 'dark',
  height = 500,
  showToolbar = true,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!containerRef.current) return

    // Clear previous widget
    containerRef.current.innerHTML = ''

    const script = document.createElement('script')
    script.src = 'https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js'
    script.type = 'text/javascript'
    script.async = true
    script.innerHTML = JSON.stringify({
      autosize: true,
      symbol: symbol,
      interval: interval,
      timezone: 'Etc/UTC',
      theme: theme,
      style: '1', // Candlestick
      locale: 'en',
      backgroundColor: 'rgba(5, 5, 15, 1)',
      gridColor: 'rgba(100, 80, 255, 0.06)',
      allow_symbol_change: true,
      calendar: false,
      support_host: 'https://www.tradingview.com',
      hide_side_toolbar: false,
      details: true,
      hotlist: false,
      show_popup_button: true,
      popup_width: '1000',
      popup_height: '650',
      withdateranges: true,
      hide_volume: false,
      studies: [
        'STD;EMA',                    // EMA
        'STD;RSI',                    // RSI
        'STD;MACD',                   // MACD
        'STD;Bollinger_Bands',        // Bollinger Bands
        'STD;Average_True_Range',     // ATR
      ],
      overrides: {
        'mainSeriesProperties.candleStyle.upColor': '#00ffaa',
        'mainSeriesProperties.candleStyle.downColor': '#ff2266',
        'mainSeriesProperties.candleStyle.borderUpColor': '#00ffaa',
        'mainSeriesProperties.candleStyle.borderDownColor': '#ff2266',
        'mainSeriesProperties.candleStyle.wickUpColor': '#00ffaa88',
        'mainSeriesProperties.candleStyle.wickDownColor': '#ff226688',
        'paneProperties.background': '#05050f',
        'paneProperties.vertGridProperties.color': 'rgba(100, 80, 255, 0.04)',
        'paneProperties.horzGridProperties.color': 'rgba(100, 80, 255, 0.04)',
        'scalesProperties.textColor': '#5a6080',
        'scalesProperties.lineColor': 'rgba(100, 80, 255, 0.1)',
      },
      enabled_features: [
        'header_symbol_search',
        'header_chart_type',
        'header_indicators',
        'header_compare',
        'header_screenshot',
        'header_fullscreen_button',
      ],
      disabled_features: [
        'header_saveload',
      ],
      toolbar_bg: '#05050f',
      loading_screen: {
        backgroundColor: '#05050f',
        foregroundColor: '#00fff0',
      },
    })

    containerRef.current.appendChild(script)

    return () => {
      if (containerRef.current) {
        containerRef.current.innerHTML = ''
      }
    }
  }, [symbol, interval, theme])

  return (
    <div
      className="tradingview-widget-container relative"
      ref={containerRef}
      style={{ height, width: '100%' }}
    >
      <div className="flex items-center justify-center h-full">
        <div className="w-8 h-8 border-2 border-[#00fff0] border-t-transparent rounded-full animate-spin" />
      </div>
    </div>
  )
}

export default memo(TradingViewWidget)
