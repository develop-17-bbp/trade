param(
    [string]$JournalPath = "logs/trading_journal.json",
    [string]$OutputDir = "authority_trade_summaries",
    [datetime]$StartDate = [datetime]"2026-03-10",
    [datetime]$EndDate = [datetime]::Today
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Escape-Xml {
    param([string]$Text)
    if ($null -eq $Text) { return "" }
    return [System.Security.SecurityElement]::Escape($Text)
}

function New-MinimalDocx {
    param(
        [string]$Path,
        [string[]]$Lines
    )

    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem

    $contentTypes = @'
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
'@

    $rels = @'
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
'@

    $paragraphs = foreach ($line in $Lines) {
        $escaped = Escape-Xml $line
        if ([string]::IsNullOrWhiteSpace($escaped)) {
            '<w:p/>'
        } else {
            "<w:p><w:r><w:t xml:space=`"preserve`">$escaped</w:t></w:r></w:p>"
        }
    }

    $document = @"
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas"
 xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"
 xmlns:o="urn:schemas-microsoft-com:office:office"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"
 xmlns:v="urn:schemas-microsoft-com:vml"
 xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing"
 xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
 xmlns:w10="urn:schemas-microsoft-com:office:word"
 xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
 xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml"
 xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup"
 xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk"
 xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml"
 xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape"
 mc:Ignorable="w14 wp14">
  <w:body>
    $($paragraphs -join "`n    ")
    <w:sectPr>
      <w:pgSz w:w="12240" w:h="15840"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/>
    </w:sectPr>
  </w:body>
</w:document>
"@

    if (Test-Path $Path) {
        Remove-Item $Path -Force
    }

    $zip = [System.IO.Compression.ZipFile]::Open($Path, [System.IO.Compression.ZipArchiveMode]::Create)
    try {
        $entry = $zip.CreateEntry("[Content_Types].xml")
        $writer = New-Object System.IO.StreamWriter($entry.Open())
        $writer.Write($contentTypes)
        $writer.Dispose()

        $entry = $zip.CreateEntry("_rels/.rels")
        $writer = New-Object System.IO.StreamWriter($entry.Open())
        $writer.Write($rels)
        $writer.Dispose()

        $entry = $zip.CreateEntry("word/document.xml")
        $writer = New-Object System.IO.StreamWriter($entry.Open())
        $writer.Write($document)
        $writer.Dispose()
    }
    finally {
        $zip.Dispose()
    }
}

function Get-DayStats {
    param([object[]]$Rows)

    if (-not $Rows -or $Rows.Count -eq 0) {
        return [pscustomobject]@{
            TotalTrades = 0
            BuyTrades = 0
            SellTrades = 0
            OpenTrades = 0
            ClosedTrades = 0
            AvgConfidence = 0.0
            TotalNotional = 0.0
            AssetCounts = @{}
            FirstTimestamp = $null
            LastTimestamp = $null
            ClosedPnl = 0.0
        }
    }

    $buyTrades = @($Rows | Where-Object side -eq "buy").Count
    $sellTrades = @($Rows | Where-Object side -eq "sell").Count
    $openTrades = @($Rows | Where-Object status -eq "OPEN").Count
    $closedTrades = @($Rows | Where-Object status -eq "CLOSED").Count
    $avgConfidence = [math]::Round((($Rows | Measure-Object -Property confidence -Average).Average), 4)
    $totalNotional = 0.0
    $closedPnl = 0.0
    foreach ($row in $Rows) {
        $totalNotional += ([double]$row.quantity * [double]$row.price)
        $closedPnl += [double]($row.pnl | ForEach-Object { if ($_ -eq $null) { 0 } else { $_ } })
    }

    $assetCounts = @{}
    foreach ($group in ($Rows | Group-Object asset)) {
        $assetCounts[$group.Name] = $group.Count
    }

    return [pscustomobject]@{
        TotalTrades = $Rows.Count
        BuyTrades = $buyTrades
        SellTrades = $sellTrades
        OpenTrades = $openTrades
        ClosedTrades = $closedTrades
        AvgConfidence = $avgConfidence
        TotalNotional = [math]::Round($totalNotional, 2)
        AssetCounts = $assetCounts
        FirstTimestamp = ($Rows | Sort-Object timestamp | Select-Object -First 1).timestamp
        LastTimestamp = ($Rows | Sort-Object timestamp | Select-Object -Last 1).timestamp
        ClosedPnl = [math]::Round($closedPnl, 2)
    }
}

if (-not (Test-Path $JournalPath)) {
    throw "Journal file not found: $JournalPath"
}

$journal = Get-Content -Raw $JournalPath | ConvertFrom-Json
$dateLookup = @{}
foreach ($row in $journal) {
    $day = ([datetime]$row.timestamp).ToString("yyyy-MM-dd")
    if (-not $dateLookup.ContainsKey($day)) {
        $dateLookup[$day] = New-Object System.Collections.ArrayList
    }
    [void]$dateLookup[$day].Add($row)
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$current = $StartDate.Date
$previousStats = $null
while ($current -le $EndDate.Date) {
    $dayKey = $current.ToString("yyyy-MM-dd")
    $rows = @()
    if ($dateLookup.ContainsKey($dayKey)) {
        $rows = @($dateLookup[$dayKey])
    }

    $stats = Get-DayStats -Rows $rows

    $changes = @()
    if ($null -eq $previousStats) {
        $changes += "This is the first day in the requested reporting period, so there is no prior day for comparison."
    } else {
        $tradeDelta = $stats.TotalTrades - $previousStats.TotalTrades
        if ($tradeDelta -gt 0) {
            $changes += "Trade count increased by $tradeDelta compared with the previous day."
        } elseif ($tradeDelta -lt 0) {
            $changes += "Trade count decreased by $([math]::Abs($tradeDelta)) compared with the previous day."
        } else {
            $changes += "Trade count was unchanged compared with the previous day."
        }

        $buyDelta = $stats.BuyTrades - $previousStats.BuyTrades
        $sellDelta = $stats.SellTrades - $previousStats.SellTrades
        $changes += "Buy-order change versus previous day: $buyDelta. Sell-order change versus previous day: $sellDelta."

        $notionalDelta = [math]::Round($stats.TotalNotional - $previousStats.TotalNotional, 2)
        if ($notionalDelta -gt 0) {
            $changes += "Approximate total order value increased by USD $notionalDelta compared with the previous day."
        } elseif ($notionalDelta -lt 0) {
            $changes += "Approximate total order value decreased by USD $([math]::Abs($notionalDelta)) compared with the previous day."
        } else {
            $changes += "Approximate total order value was unchanged compared with the previous day."
        }
    }

    $assetLines = @()
    if ($stats.AssetCounts.Count -eq 0) {
        $assetLines += "- No recorded trades for BTC, ETH, or AAVE on this date."
    } else {
        foreach ($assetName in ($stats.AssetCounts.Keys | Sort-Object)) {
            $assetLines += "- ${assetName}: $($stats.AssetCounts[$assetName]) trades"
        }
    }

    $bodyLines = @(
        "# Trade Summary for $($current.ToString('MMMM d, yyyy'))",
        "",
        "This file is a plain-language summary of the trade activity recorded for this date in the system journal.",
        "",
        "Important note:",
        "- The primary source used is logs/trading_journal.json.",
        "- The current journal records SHADOW/testnet entries for this reporting period.",
        "- If the live system is currently trading but the journal has not been updated yet, those newer events are not part of this formal summary.",
        "",
        "Summary of activity:",
        "- Total recorded trades: $($stats.TotalTrades)",
        "- Buy orders: $($stats.BuyTrades)",
        "- Sell orders: $($stats.SellTrades)",
        "- Open trades: $($stats.OpenTrades)",
        "- Closed trades: $($stats.ClosedTrades)",
        "- Average recorded confidence: $([math]::Round($stats.AvgConfidence * 100, 1))%",
        "- Approximate total notional value of orders: USD $('{0:N2}' -f $stats.TotalNotional)",
        "- Recorded time window: $(if ($stats.FirstTimestamp) { $stats.FirstTimestamp } else { 'No entries recorded' }) to $(if ($stats.LastTimestamp) { $stats.LastTimestamp } else { 'No entries recorded' })",
        "- Profit achieved for this day based on journaled trade P&L: USD $('{0:N2}' -f $stats.ClosedPnl)",
        "- Realized P&L from closed trades recorded for this day: USD $('{0:N2}' -f $stats.ClosedPnl)",
        "",
        "Breakdown by asset:"
    ) + $assetLines + @(
        "",
        "Plain-language summary:",
        $(if ($stats.TotalTrades -eq 0) {
            "- No trade entries were found in the formal journal for this date."
        } else {
            "- The journal shows trading activity in testnet/shadow mode for this date."
        }),
        $(if ($stats.TotalTrades -gt 0 -and $stats.ClosedTrades -eq 0) {
            "- The recorded entries for this date do not show completed closures; all recorded trades remain open."
        } elseif ($stats.ClosedTrades -gt 0) {
            "- The journal records both open and closed trades for this date."
        } else {
            "- There are no recorded positions to describe for this date."
        }),
        $(if ($stats.ClosedPnl -gt 0) {
            "- The journal shows a positive profit of USD $('{0:N2}' -f $stats.ClosedPnl) achieved on this date."
        } elseif ($stats.ClosedPnl -lt 0) {
            "- The journal shows a net loss of USD $('{0:N2}' -f [math]::Abs($stats.ClosedPnl)) recorded on this date."
        } else {
            "- The journal does not yet show realized profit for this date."
        }),
        "",
        "What changed from the previous day:"
    ) + ($changes | ForEach-Object { "- $_" }) + @(
        "",
        "Source note:",
        "- Generated from $JournalPath on $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')."
    )

    $safeName = "$dayKey" + "_trade_summary_for_authorities"
    $mdPath = Join-Path $OutputDir ($safeName + ".md")
    $docxPath = Join-Path $OutputDir ($safeName + ".docx")

    Set-Content -Path $mdPath -Value ($bodyLines -join "`r`n") -Encoding UTF8
    New-MinimalDocx -Path $docxPath -Lines $bodyLines

    $previousStats = $stats
    $current = $current.AddDays(1)
}

Write-Output "Generated daily authority summaries in $OutputDir"
