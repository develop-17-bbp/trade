import pandas as pd
import numpy as np
from datetime import datetime

# Load data
btc = pd.read_csv('results/btc_10yr_score7.csv')
eth = pd.read_csv('results/eth_10yr_score7.csv')

# Tag and combine
btc['asset'] = 'BTC'
eth['asset'] = 'ETH'
df = pd.concat([btc, eth], ignore_index=True)
df['entry_time'] = pd.to_datetime(df['entry_time'])
df['exit_time'] = pd.to_datetime(df['exit_time'])
df['hour'] = df['entry_time'].dt.hour
df['is_win'] = df['pnl_pct'] > 0

print(f"Total trades: {len(df)} (BTC: {len(btc)}, ETH: {len(eth)})")
print(f"Date range: {df['entry_time'].min()} to {df['entry_time'].max()}")
print(f"Overall WR: {df['is_win'].mean()*100:.1f}%")
print(f"Avg Win: {df[df['is_win']]['pnl_pct'].mean():.4f}%")
print(f"Avg Loss: {df[~df['is_win']]['pnl_pct'].mean():.4f}%")
gross_profit = df[df['pnl_pct']>0]['pnl_pct'].sum()
gross_loss = abs(df[df['pnl_pct']<0]['pnl_pct'].sum())
pf = gross_profit / gross_loss if gross_loss > 0 else 0
print(f"Profit Factor: {pf:.4f}")
print(f"Total PnL%: {df['pnl_pct'].sum():.2f}%")
print()

# Helper
def calc_pf(data):
    gp = data[data['pnl_pct']>0]['pnl_pct'].sum()
    gl = abs(data[data['pnl_pct']<0]['pnl_pct'].sum())
    return gp/gl if gl > 0 else 0

def calc_stats(data):
    n = len(data)
    if n == 0:
        return {'count':0,'wr':0,'avg_pnl':0,'total_pnl':0,'pf':0}
    wr = data['is_win'].mean()*100
    avg = data['pnl_pct'].mean()
    total = data['pnl_pct'].sum()
    pf = calc_pf(data)
    return {'count':n,'wr':f'{wr:.1f}','avg_pnl':f'{avg:.4f}','total_pnl':f'{total:.2f}','pf':f'{pf:.3f}'}

# ============================================================
print("="*80)
print("1. HARD STOP ANALYSIS")
print("="*80)
hard_stop_mask = df['exit_reason'].str.contains('Hard stop', na=False)
print(f"Hard stop trades: {hard_stop_mask.sum()} ({hard_stop_mask.mean()*100:.1f}% of all trades)")
print(f"Avg hard stop loss: {df[hard_stop_mask]['pnl_pct'].mean():.4f}%")
print()

for cap in [1.5, 1.0, 0.75, 0.5]:
    capped = df.copy()
    capped.loc[hard_stop_mask & (capped['pnl_pct'] < -cap), 'pnl_pct'] = -cap
    capped['is_win'] = capped['pnl_pct'] > 0
    pf_new = calc_pf(capped)
    wr_new = capped['is_win'].mean()*100
    avg_loss_new = capped[capped['pnl_pct']<0]['pnl_pct'].mean()
    total_new = capped['pnl_pct'].sum()
    print(f"  Hard stop = -{cap:.1f}%: PF={pf_new:.4f}, WR={wr_new:.1f}%, AvgLoss={avg_loss_new:.4f}%, TotalPnL={total_new:.2f}%")

print()

# ============================================================
print("="*80)
print("2. EXIT REASON BREAKDOWN")
print("="*80)

def categorize_exit(reason):
    if 'Hard stop' in str(reason):
        return 'Hard Stop'
    elif 'SL L1' in str(reason):
        return 'SL L1'
    elif 'SL L2' in str(reason):
        return 'SL L2'
    elif 'SL L3' in str(reason):
        return 'SL L3'
    elif 'SL L4' in str(reason):
        return 'SL L4'
    elif 'SL L5' in str(reason):
        return 'SL L5'
    else:
        return 'EMA Exit'

df['exit_cat'] = df['exit_reason'].apply(categorize_exit)

exit_stats = []
for cat in sorted(df['exit_cat'].unique()):
    subset = df[df['exit_cat']==cat]
    s = calc_stats(subset)
    s['exit_reason'] = cat
    exit_stats.append(s)

exit_df = pd.DataFrame(exit_stats)[['exit_reason','count','wr','avg_pnl','total_pnl','pf']]
exit_df = exit_df.sort_values('total_pnl', key=lambda x: x.astype(float))
print(exit_df.to_string(index=False))
print()

# ============================================================
print("="*80)
print("3. SL L1 DEEP DIVE")
print("="*80)

l1 = df[df['exit_cat']=='SL L1']
l1_wins = l1[l1['is_win']]
l1_losses = l1[~l1['is_win']]
print(f"L1 trades: {len(l1)}")
print(f"L1 WR: {l1['is_win'].mean()*100:.1f}%")
print(f"L1 PF: {calc_pf(l1):.4f}")
print(f"L1 Winners - count: {len(l1_wins)}, avg: {l1_wins['pnl_pct'].mean():.4f}%, total: {l1_wins['pnl_pct'].sum():.2f}%")
print(f"L1 Losers  - count: {len(l1_losses)}, avg: {l1_losses['pnl_pct'].mean():.4f}%, total: {l1_losses['pnl_pct'].sum():.2f}%")
print()

print("Simulating L1 WR improvement:")
for target_wr in [0.58, 0.60, 0.65]:
    current_wr = l1['is_win'].mean()
    n_flip = int(len(l1) * (target_wr - current_wr))
    if n_flip <= 0:
        continue
    sim = df.copy()
    l1_loser_idx = sim[(sim['exit_cat']=='SL L1') & (~sim['is_win'])].index[:n_flip]
    avg_win = l1_wins['pnl_pct'].mean()
    sim.loc[l1_loser_idx, 'pnl_pct'] = avg_win
    sim['is_win'] = sim['pnl_pct'] > 0
    print(f"  L1 WR={target_wr*100:.0f}%: flip {n_flip} losers -> PF={calc_pf(sim):.4f}, TotalPnL={sim['pnl_pct'].sum():.2f}%")

print()

# ============================================================
print("="*80)
print("4. DURATION ANALYSIS")
print("="*80)

dur_bins = [0, 30, 60, 120, 180, 240, 360, 480, 720, 99999]
dur_labels = ['0-30', '30-60', '60-120', '120-180', '180-240', '240-360', '360-480', '480-720', '720+']
df['dur_bin'] = pd.cut(df['duration_min'], bins=dur_bins, labels=dur_labels, right=True)

dur_stats = []
for label in dur_labels:
    subset = df[df['dur_bin']==label]
    if len(subset) > 0:
        s = calc_stats(subset)
        s['duration'] = label
        dur_stats.append(s)

dur_df = pd.DataFrame(dur_stats)[['duration','count','wr','avg_pnl','total_pnl','pf']]
print(dur_df.to_string(index=False))
print()

print("Cumulative: keep only trades <= X minutes:")
for max_min in [60, 90, 120, 180, 240, 360, 480]:
    filtered = df[df['duration_min'] <= max_min]
    if len(filtered) > 10:
        print(f"  <= {max_min}min: N={len(filtered)}, WR={filtered['is_win'].mean()*100:.1f}%, PF={calc_pf(filtered):.4f}, TotalPnL={filtered['pnl_pct'].sum():.2f}%")

print()

# ============================================================
print("="*80)
print("5. ENTRY SCORE ANALYSIS")
print("="*80)

score_stats = []
for score in sorted(df['entry_score'].unique()):
    subset = df[df['entry_score']==score]
    s = calc_stats(subset)
    s['score'] = score
    score_stats.append(s)

score_df = pd.DataFrame(score_stats)[['score','count','wr','avg_pnl','total_pnl','pf']]
print(score_df.to_string(index=False))
print()

print("Cumulative score filter (score >= X):")
for min_score in range(7, 20):
    subset = df[df['entry_score'] >= min_score]
    if len(subset) >= 10:
        print(f"  score >= {min_score}: N={len(subset)}, WR={subset['is_win'].mean()*100:.1f}%, PF={calc_pf(subset):.4f}, TotalPnL={subset['pnl_pct'].sum():.2f}%")

print()

# ============================================================
print("="*80)
print("6. TIME-OF-DAY ANALYSIS (UTC)")
print("="*80)

hour_stats = []
for h in range(24):
    subset = df[df['hour']==h]
    if len(subset) > 0:
        s = calc_stats(subset)
        s['hour'] = h
        hour_stats.append(s)

hour_df = pd.DataFrame(hour_stats)[['hour','count','wr','avg_pnl','total_pnl','pf']]
print(hour_df.to_string(index=False))
print()

print("Hours with PF > 1.0:")
for row in hour_stats:
    if float(row['pf']) > 1.0:
        print(f"  Hour {row['hour']:02d}: PF={row['pf']}, WR={row['wr']}%, N={row['count']}")

print("\nHours with PF < 0.8 (candidates to skip):")
bad_hours = set()
for row in hour_stats:
    if float(row['pf']) < 0.8:
        bad_hours.add(row['hour'])
        print(f"  Hour {row['hour']:02d}: PF={row['pf']}, WR={row['wr']}%, N={row['count']}, TotalPnL={row['total_pnl']}%")

print()

# ============================================================
print("="*80)
print("7. DIRECTION ANALYSIS")
print("="*80)

for d in ['LONG', 'SHORT']:
    subset = df[df['direction']==d]
    print(f"{d}: N={len(subset)}, WR={subset['is_win'].mean()*100:.1f}%, PF={calc_pf(subset):.4f}, AvgPnL={subset['pnl_pct'].mean():.4f}%, TotalPnL={subset['pnl_pct'].sum():.2f}%")

print()
for asset in ['BTC', 'ETH']:
    for d in ['LONG', 'SHORT']:
        subset = df[(df['asset']==asset) & (df['direction']==d)]
        if len(subset) > 0:
            print(f"  {asset} {d}: N={len(subset)}, WR={subset['is_win'].mean()*100:.1f}%, PF={calc_pf(subset):.4f}, TotalPnL={subset['pnl_pct'].sum():.2f}%")

print()

# ============================================================
print("="*80)
print("8. WINNING FORMULA - FILTER COMBINATIONS")
print("="*80)

# Use bad hours with more lenient threshold for combo search
bad_hours_strict = set()
for row in hour_stats:
    if float(row['pf']) < 0.8 and int(row['count']) > 20:
        bad_hours_strict.add(row['hour'])

print(f"Bad hours (PF < 0.8, N>20): {sorted(bad_hours_strict)}")
print()

results = []

for min_score in [7, 8, 9, 10, 11]:
    for max_dur in [99999, 480, 360, 240, 180, 120]:
        for skip_bad_hours in [False, True]:
            for direction in ['BOTH', 'LONG', 'SHORT']:
                for hard_stop_cap in [2.5, 2.0, 1.5, 1.0]:
                    filt = df.copy()
                    filt = filt[filt['entry_score'] >= min_score]
                    if max_dur < 99999:
                        filt = filt[filt['duration_min'] <= max_dur]
                    if skip_bad_hours:
                        filt = filt[~filt['hour'].isin(bad_hours_strict)]
                    if direction != 'BOTH':
                        filt = filt[filt['direction'] == direction]
                    if hard_stop_cap < 2.5:
                        hs = filt['exit_reason'].str.contains('Hard stop', na=False)
                        filt.loc[hs & (filt['pnl_pct'] < -hard_stop_cap), 'pnl_pct'] = -hard_stop_cap

                    if len(filt) < 50:
                        continue

                    filt['is_win'] = filt['pnl_pct'] > 0
                    pf_val = calc_pf(filt)
                    wr_val = filt['is_win'].mean()*100
                    total = filt['pnl_pct'].sum()

                    results.append({
                        'min_score': min_score,
                        'max_dur': max_dur if max_dur < 99999 else 'none',
                        'skip_bad_hrs': skip_bad_hours,
                        'direction': direction,
                        'hard_stop': hard_stop_cap,
                        'n_trades': len(filt),
                        'wr': round(wr_val, 1),
                        'pf': round(pf_val, 4),
                        'total_pnl': round(total, 2)
                    })

res_df = pd.DataFrame(results)

winners = res_df[res_df['pf'] >= 1.0].sort_values('n_trades', ascending=False)
print(f"Combinations achieving PF >= 1.0: {len(winners)}")
print(f"\nTop 25 by trade count (most trades = least restrictive):")
cols = ['min_score','max_dur','skip_bad_hrs','direction','hard_stop','n_trades','wr','pf','total_pnl']
if len(winners) > 0:
    print(winners[cols].head(25).to_string(index=False))
else:
    print("  None found - showing closest:")
    closest = res_df.nlargest(10, 'pf')
    print(closest[cols].to_string(index=False))

print()
print("="*80)
print("MINIMUM CHANGES (least restrictive filters achieving PF >= 1.0):")
print("="*80)

if len(winners) > 0:
    best = winners.iloc[0]
    print(f"  Min score:      >= {best['min_score']}")
    print(f"  Max duration:   {best['max_dur']} min")
    print(f"  Skip bad hours: {best['skip_bad_hrs']} {('(' + str(sorted(bad_hours_strict)) + ')') if best['skip_bad_hrs'] else ''}")
    print(f"  Direction:      {best['direction']}")
    print(f"  Hard stop cap:  -{best['hard_stop']}%")
    print(f"  ---")
    print(f"  Trades:         {best['n_trades']}")
    print(f"  Win Rate:       {best['wr']:.1f}%")
    print(f"  Profit Factor:  {best['pf']:.4f}")
    print(f"  Total PnL:      {best['total_pnl']:.2f}%")
else:
    print("  No combination achieved PF >= 1.0 with >= 50 trades")

print()
print("="*80)
print("SINGLE-CHANGE IMPACT (only one filter changed at a time):")
print("="*80)

for cap in [2.0, 1.5, 1.0, 0.75]:
    capped = df.copy()
    hs = capped['exit_reason'].str.contains('Hard stop', na=False)
    capped.loc[hs & (capped['pnl_pct'] < -cap), 'pnl_pct'] = -cap
    capped['is_win'] = capped['pnl_pct'] > 0
    print(f"  Hard stop = -{cap}% only:     N={len(capped)}, PF={calc_pf(capped):.4f}")

print()
filt = df[~df['hour'].isin(bad_hours_strict)]
print(f"  Skip bad hours only:        N={len(filt)}, PF={calc_pf(filt):.4f}")

for s in [8, 9, 10, 11]:
    filt = df[df['entry_score'] >= s]
    print(f"  Score >= {s} only:            N={len(filt)}, PF={calc_pf(filt):.4f}")

for d in [120, 180, 240, 360]:
    filt = df[df['duration_min'] <= d]
    print(f"  Duration <= {d}min only:     N={len(filt)}, PF={calc_pf(filt):.4f}")

for d in ['LONG', 'SHORT']:
    filt = df[df['direction'] == d]
    print(f"  {d} only:                  N={len(filt)}, PF={calc_pf(filt):.4f}")

print()
print("="*80)
print("PER-ASSET RESULTS FOR BEST COMBO:")
print("="*80)
if len(winners) > 0:
    best = winners.iloc[0]
    for asset in ['BTC', 'ETH']:
        filt = df[df['asset']==asset].copy()
        filt = filt[filt['entry_score'] >= best['min_score']]
        if best['max_dur'] != 'none':
            filt = filt[filt['duration_min'] <= best['max_dur']]
        if best['skip_bad_hrs']:
            filt = filt[~filt['hour'].isin(bad_hours_strict)]
        if best['direction'] != 'BOTH':
            filt = filt[filt['direction'] == best['direction']]
        if best['hard_stop'] < 2.5:
            hs = filt['exit_reason'].str.contains('Hard stop', na=False)
            filt.loc[hs & (filt['pnl_pct'] < -best['hard_stop']), 'pnl_pct'] = -best['hard_stop']
        filt['is_win'] = filt['pnl_pct'] > 0
        print(f"  {asset}: N={len(filt)}, WR={filt['is_win'].mean()*100:.1f}%, PF={calc_pf(filt):.4f}, TotalPnL={filt['pnl_pct'].sum():.2f}%")
