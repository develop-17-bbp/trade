import sys

file_path = r'c:\Users\convo\trade\src\trading\executor.py'

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
target_found = False
for line in lines:
    if "ext_scraped = self.institutional.get_all_institutional(asset)" in line and not target_found:
        indent = line[:line.find("ext_scraped")]
        new_lines.append(f"{indent}# L2 Order Book Microstructure\n")
        new_lines.append(f"{indent}ob_data = self.price_source.exchange.fetch_order_book(symbol)\n")
        new_lines.append(f"{indent}l2_metrics = self.microstructure.analyze_order_book(ob_data)\n")
        new_lines.append(f"{indent}ext_feats.update(l2_metrics)\n\n")
        new_lines.append(line)
        target_found = True
    elif "ext_feats.update(ext_scraped)" in line and target_found:
        new_lines.append(line)
        indent = line[:line.find("ext_feats")]
        new_lines.append(f"\n{indent}# Volatility Regime Detection\n")
        new_lines.append(f"{indent}vol_metrics = self.regime_detector.detect_regime(closes, highs, lows)\n")
        new_lines.append(f"{indent}ext_feats.update(vol_metrics)\n")
        target_found = False # Reset for next asset in assets
    else:
        new_lines.append(line)

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Successfully injected microstructure and regime detection.")
