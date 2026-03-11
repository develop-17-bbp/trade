# 🔑 API Keys Setup Guide

To run the trading system, you need to configure the API keys. Here's how:

## Quick Start

### Option 1: Testnet Mode (Recommended for Testing)
Use **Binance Testnet** to trade with fake money (no risk).

**Steps:**
1. Go to: https://testnet.binance.vision/
2. Sign in with your Binance credentials or create account
3. Go to **API Management** (https://testnet.binance.vision/key/publicKey)
4. Create a new API key:
   - Label: `Trading Bot`
   - Restrictions: **Enable Spot Trading** ✓, **Enable Reading Account Trade History** ✓
5. Copy the **API Key** and **Secret Key**
6. Update `config.yaml`:
   ```yaml
   exchange:
     api_key: "YOUR_TESTNET_KEY_HERE"      # Paste API Key
     api_secret: "YOUR_TESTNET_SECRET_HERE" # Paste Secret Key
   ```
7. Restart the system

**Status after restart:**
```
✓ [TESTNET] Connected to binance Testnet (sandbox mode)
✓ [TESTNET] API key authenticated — order execution enabled
```

---

### Option 2: Live Trading Mode (Production)
Trade with **real money** on Binance.

⚠️ **Warning:** This carries real financial risk. Use testnet first to validate the system!

**Steps:**
1. Go to: https://www.binance.com/en/user/settings/api-management
2. Sign in to your Binance account
3. Create a new API key:
   - Label: `Trading Bot`
   - Restrictions: **Enable Spot Trading** ✓, **Enable Reading Account Trade History** ✓
4. Copy the **API Key** and **Secret Key**
5. Update `config.yaml`:
   ```yaml
   mode: live                    # ⚠️ Changes to LIVE TRADING
   exchange:
     api_key: "YOUR_LIVE_KEY_HERE"      # Paste API Key
     api_secret: "YOUR_LIVE_SECRET_HERE" # Paste Secret Key
   ```
6. Restart and verify balances show correctly before trading

---

## Troubleshooting

### "Invalid Api-Key ID" (Code -2008)

This means the API key is missing or invalid.

**Checklist:**
```
□ API key is NOT empty in config.yaml
□ API key is NOT wrapped in quotes incorrectly: ✓ api_key: "key" (correct)
□ You used **Testnet keys** (from testnet.binance.vision) if mode: testnet
□ You used **Live keys** (from binance.com) if mode: live
□ The keys are not expired or revoked
□ The IP address is whitelisted (if IP restriction is enabled on the key)
```

**Fix:**
1. Delete the old API key in Binance
2. Create a new API key with these restrictions:
   - ✅ Enable Spot Trading
   - ✅ Enable Reading Account Trade History
   - ⚠️ Do NOT restrict IP address (or whitelist your IP)
3. Copy the NEW key and secret into `config.yaml`
4. Restart

---

### No Balance Shown (Read-Only Mode)

If the system runs with "[TESTNET] No API key — read-only mode", it means:
- The API key field is empty in `config.yaml`
- OR the API key is invalid and the system is falling back to read-only

**Fix:**
1. Update `config.yaml` with your API key/secret (see above)
2. Restart the system

---

### Rate Limiting Warning

The system automatically respects Binance rate limits. If you see rate limit warnings:
- This is normal during backtesting or heavy data fetching
- The system will automatically retry after a short delay
- No action needed

---

## Security Best Practices

⚠️ **Never commit API keys to Git!**

```bash
# Keep your config.yaml out of version control:
git add config.yaml -u         # Remove it from tracking
git status                      # Verify it's now untracked
```

**Safe API Key Restrictions (Recommended):**
- ✅ **Enable Spot Trading** — allows buy/sell orders
- ✅ **Enable Reading Account Trade History** — allows fetching balances
- ❌ **Disable Withdrawal** — prevents accidental fund transfer
- ❌ **Disable Margin** — prevents leveraged trading (not supported)
- ⚠️ **IP Whitelist** — optional, but recommended if your IP is static

---

## Environment Variables (Advanced)

As an alternative to `config.yaml`, you can set environment variables:

```bash
# .env file (create in project root)
BINANCE_TESTNET_KEY=testnet_key_here
BINANCE_TESTNET_SECRET=testnet_secret_here
BINANCE_API_KEY=live_key_here
BINANCE_API_SECRET=live_secret_here
```

The system will check these if `config.yaml` is empty.

---

## Verification

After setting up API keys, run:

```bash
python -m src.main --mode testnet --symbol BTC
```

Expected output:
```
✓ [TESTNET] Connected to binance Testnet (sandbox mode)
✓ [TESTNET] API key authenticated — order execution enabled

📊 EXCHANGE WALLET BALANCES
──────────────────────────────────
Asset      Total           Available
──────────── ─────────────────────
USDT       10000.0000      10000.0000

💰 Reference Capital: $10,000.00 USDT
```

If you see balances, you're ✅ **all set!**

---

## Support

- **Binance Help:** https://www.binance.com/en/support
- **Testnet Guide:** https://academy.binance.com/en/articles/binance-testnet-guide
- **API Documentation:** https://binance-docs.github.io/apidocs/

