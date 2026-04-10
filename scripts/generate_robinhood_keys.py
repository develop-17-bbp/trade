"""
Generate ED25519 Key Pair for Robinhood Crypto API
===================================================
Run this ONCE, then:
  1. Copy the PUBLIC key → paste into Robinhood (robinhood.com/account/crypto → API Keys)
  2. Copy the PRIVATE key → paste into .env as ROBINHOOD_PRIVATE_KEY
  3. Copy the API key Robinhood gives you → paste into .env as ROBINHOOD_API_KEY
"""

import base64
from nacl.signing import SigningKey

# Generate key pair
private_key = SigningKey.generate()
public_key = private_key.verify_key

private_b64 = base64.b64encode(private_key.encode()).decode()
public_b64 = base64.b64encode(public_key.encode()).decode()

print("=" * 60)
print("  ROBINHOOD CRYPTO API — ED25519 KEY PAIR")
print("=" * 60)
print()
print("STEP 1: Copy this PUBLIC key and paste into Robinhood:")
print(f"  {public_b64}")
print()
print("STEP 2: Robinhood will give you an API KEY. Paste it in .env:")
print("  ROBINHOOD_API_KEY=<paste here>")
print()
print("STEP 3: Copy this PRIVATE key into .env:")
print(f"  ROBINHOOD_PRIVATE_KEY={private_b64}")
print()
print("=" * 60)
print("  KEEP THE PRIVATE KEY SECRET. NEVER SHARE IT.")
print("=" * 60)
