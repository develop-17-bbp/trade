import { useEffect, useRef, memo } from 'react'

interface Props {
  symbol?: string
  interval?: string
  theme?: 'dark' | 'light'
  height?: number
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
      backgroundColor: 'rgba(0, 0, 0, 1)',
      gridColor: 'rgba(255, 255, 255, 0.04)',
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
        'mainSeriesProperties.candleStyle.upColor': '#22c55e',
        'mainSeriesProperties.candleStyle.downColor': '#ef4444',
        'mainSeriesProperties.candleStyle.borderUpColor': '#22c55e',
        'mainSeriesProperties.candleStyle.borderDownColor': '#ef4444',
        'mainSeriesProperties.candleStyle.wickUpColor': '#22c55e88',
        'mainSeriesProperties.candleStyle.wickDownColor': '#ef444488',
        'paneProperties.background': '#000000',
        'paneProperties.vertGridProperties.color': 'rgba(255, 255, 255, 0.03)',
        'paneProperties.horzGridProperties.color': 'rgba(255, 255, 255, 0.03)',
        'scalesProperties.textColor': '#666666',
        'scalesProperties.lineColor': 'rgba(255, 255, 255, 0.06)',
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
      toolbar_bg: '#000000',
      loading_screen: {
        backgroundColor: '#000000',
        foregroundColor: '#666666',
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
        <div className="w-8 h-8 border-2 border-[#333] border-t-transparent rounded-full animate-spin" />
      </div>
    </div>
  )
}

export default memo(TradingViewWidget)
