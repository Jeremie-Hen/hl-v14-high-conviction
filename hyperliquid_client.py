"""
Hyperliquid API Client — v14 High-Conviction Bot
Execution logic ported from hl-multistrat-shortonly-3x:
  - OID tracking for TP/SL orders
  - Fill price extraction from order responses
  - Precise cancel-by-OID (not cancel-all)
  - frontendOpenOrders for reliable order listing
  - SL as stop-market, trailing stop via cancel+replace
"""
import json
import time

import requests
from eth_account import Account

import config

try:
    import msgpack
except ImportError:
    msgpack = None


def float_to_wire(x: float) -> str:
    rounded = round(x, 8)
    if abs(rounded) < 1e-8:
        return "0"
    return f"{rounded:.8f}".rstrip("0").rstrip(".")


def address_to_bytes(address: str) -> bytes:
    return bytes.fromhex(address[2:] if address.startswith("0x") else address)


def log(msg):
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] {msg}", flush=True)


class HyperliquidClient:
    def __init__(self):
        self.api_url = config.API_URL
        self.session = requests.Session()
        self.is_mainnet = not config.TESTNET

        key = config.PRIVATE_KEY
        if key:
            if not key.startswith("0x"):
                key = "0x" + key
            self.account = Account.from_key(key)
            self.wallet_address = self.account.address
            cfg_addr = config.WALLET_ADDRESS
            if cfg_addr and cfg_addr.lower() != self.wallet_address.lower():
                log(f"WARNING: Config wallet ({cfg_addr}) != key wallet ({self.wallet_address})")
            log(f"HL client init: {self.wallet_address}")
        else:
            self.account = None
            self.wallet_address = config.WALLET_ADDRESS

        self._asset_to_index = None
        self._asset_sz_decimals = {}

    # ═════════════════════════════════════════════════════════════════════
    # MARKET DATA
    # ═════════════════════════════════════════════════════════════════════

    def get_mid_price(self, symbol: str) -> float:
        resp = self.session.post(f"{self.api_url}/info", json={"type": "allMids"}, timeout=10)
        resp.raise_for_status()
        return float(resp.json().get(symbol, 0))

    def get_meta(self) -> dict:
        if self._asset_to_index is not None:
            return self._asset_to_index
        resp = self.session.post(f"{self.api_url}/info", json={"type": "meta"}, timeout=10)
        resp.raise_for_status()
        meta = resp.json()
        self._asset_to_index = {}
        self._asset_sz_decimals = {}
        for i, asset in enumerate(meta.get("universe", [])):
            name = asset.get("name")
            self._asset_to_index[name] = i
            self._asset_sz_decimals[name] = asset.get("szDecimals", 5)
        log(f"Loaded {len(self._asset_to_index)} assets from meta")
        return self._asset_to_index

    def get_asset_index(self, symbol: str) -> int:
        self.get_meta()
        return self._asset_to_index.get(symbol, 0)

    def get_sz_decimals(self, symbol: str) -> int:
        self.get_meta()
        return self._asset_sz_decimals.get(symbol, 5)

    def round_size(self, symbol: str, size: float) -> float:
        return round(size, self.get_sz_decimals(symbol))

    def round_price(self, symbol: str, price: float) -> float:
        if symbol == "BTC":
            return round(price, 0)
        elif symbol == "ETH":
            return round(price, 1)
        return round(price, 2)

    # ═════════════════════════════════════════════════════════════════════
    # ACCOUNT
    # ═════════════════════════════════════════════════════════════════════

    def get_account_state(self, address: str = None) -> dict:
        addr = address or self.wallet_address
        resp = self.session.post(f"{self.api_url}/info",
                                 json={"type": "clearinghouseState", "user": addr}, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_spot_balance(self, address: str = None) -> float:
        addr = address or self.wallet_address
        try:
            resp = self.session.post(f"{self.api_url}/info",
                                     json={"type": "spotClearinghouseState", "user": addr}, timeout=10)
            resp.raise_for_status()
            for bal in resp.json().get("balances", []):
                if bal.get("coin") == "USDC":
                    return float(bal.get("total", 0))
        except Exception as e:
            log(f"Error getting spot balance: {e}")
        return 0.0

    def get_account_value(self, address: str = None) -> float:
        try:
            addr = address or self.wallet_address
            perp_state = self.get_account_state(addr)
            ms = perp_state.get("marginSummary", {})
            perp_val = float(ms.get("accountValue", 0))
            if perp_val == 0:
                perp_val = float(perp_state.get("crossMarginSummary", {}).get("accountValue", 0))
            if perp_val == 0:
                perp_val = float(perp_state.get("withdrawable", 0))

            spot_val = self.get_spot_balance(addr)
            total = max(perp_val, spot_val, perp_val + spot_val)
            log(f"Account value: Perp=${perp_val:.2f} | Spot=${spot_val:.2f} | "
                f"Using=${total:.2f} (unified account)")
            return total
        except Exception as e:
            log(f"Error getting account value: {e}")
            return 0.0

    def get_open_positions(self, address: str = None) -> list:
        state = self.get_account_state(address)
        positions = []
        for ap in state.get("assetPositions", []):
            pos = ap.get("position", {})
            sz = float(pos.get("szi", 0))
            if sz != 0:
                positions.append({
                    "symbol": pos.get("coin"),
                    "size": sz,
                    "entry_price": float(pos.get("entryPx", 0)),
                    "unrealized_pnl": float(pos.get("unrealizedPnl", 0)),
                    "leverage": float(pos.get("leverage", {}).get("value", 1))
                        if isinstance(pos.get("leverage"), dict)
                        else float(pos.get("leverage", 1)),
                })
        return positions

    # ═════════════════════════════════════════════════════════════════════
    # L1 ACTION SIGNING + SEND HELPER
    # ═════════════════════════════════════════════════════════════════════

    def _sign_l1_action(self, action: dict, nonce: int, vault_address: str = None) -> dict:
        if not self.account:
            raise ValueError("Private key not configured")
        if msgpack is None:
            raise ImportError("msgpack required: pip install msgpack")

        from eth_utils import keccak
        data = msgpack.packb(action)
        data += nonce.to_bytes(8, "big")
        data += (b"\x00" if vault_address is None
                 else b"\x01" + address_to_bytes(vault_address))

        connection_id = keccak(data)
        phantom_agent = {
            "source": "a" if self.is_mainnet else "b",
            "connectionId": connection_id,
        }
        typed_data = {
            "domain": {
                "chainId": 1337, "name": "Exchange",
                "verifyingContract": "0x0000000000000000000000000000000000000000",
                "version": "1",
            },
            "types": {
                "Agent": [
                    {"name": "source", "type": "string"},
                    {"name": "connectionId", "type": "bytes32"},
                ],
                "EIP712Domain": [
                    {"name": "name", "type": "string"},
                    {"name": "version", "type": "string"},
                    {"name": "chainId", "type": "uint256"},
                    {"name": "verifyingContract", "type": "address"},
                ],
            },
            "primaryType": "Agent",
            "message": phantom_agent,
        }
        try:
            from eth_account.messages import encode_typed_data
            encoded = encode_typed_data(full_message=typed_data)
        except ImportError:
            from eth_account.messages import encode_structured_data
            encoded = encode_structured_data(typed_data)

        from eth_utils import to_hex
        signed = self.account.sign_message(encoded)
        return {"r": to_hex(signed.r), "s": to_hex(signed.s), "v": signed.v}

    def _send_action(self, action: dict) -> dict:
        """Sign and send an action to the exchange endpoint."""
        nonce = int(time.time() * 1000)
        sig = self._sign_l1_action(action, nonce)
        payload = {"action": action, "nonce": nonce, "signature": sig}
        resp = self.session.post(f"{self.api_url}/exchange", json=payload,
                                 headers={"Content-Type": "application/json"}, timeout=15)
        return resp.json() if resp.text else {}

    # ═════════════════════════════════════════════════════════════════════
    # OID + FILL PRICE EXTRACTION
    # ═════════════════════════════════════════════════════════════════════

    @staticmethod
    def extract_oid(order_result: dict):
        """Extract order ID from an order placement response."""
        try:
            statuses = (order_result.get("response", {})
                        .get("data", {}).get("statuses", []))
            if statuses:
                s = statuses[0]
                if "resting" in s:
                    return s["resting"]["oid"]
                if "filled" in s:
                    return s["filled"]["oid"]
        except (AttributeError, TypeError, KeyError, IndexError):
            pass
        return None

    @staticmethod
    def extract_fill_price(order_result: dict):
        """Extract average fill price from order response."""
        try:
            statuses = (order_result.get("response", {})
                        .get("data", {}).get("statuses", []))
            if statuses:
                filled = statuses[0].get("filled", {})
                avg_px = filled.get("avgPx")
                if avg_px is not None:
                    return float(avg_px)
        except (AttributeError, TypeError, ValueError, IndexError):
            pass
        return None

    # ═════════════════════════════════════════════════════════════════════
    # ORDER MANAGEMENT
    # ═════════════════════════════════════════════════════════════════════

    def get_open_orders(self, address: str = None) -> list:
        """Get all open/resting orders including triggers (frontendOpenOrders)."""
        addr = address or self.wallet_address
        resp = self.session.post(f"{self.api_url}/info",
                                 json={"type": "frontendOpenOrders", "user": addr}, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def cancel_orders(self, symbol: str, oids: list) -> dict:
        """Cancel specific orders by OID. Ignores already-consumed orders."""
        if not config.LIVE_TRADING or not oids:
            return {"status": "skipped"}
        asset_idx = self.get_asset_index(symbol)
        cancels = [{"a": asset_idx, "o": int(oid)} for oid in oids if oid]
        if not cancels:
            return {"status": "no_oids"}
        action = {"type": "cancel", "cancels": cancels}
        try:
            result = self._send_action(action)
            log(f"Cancelled {len(cancels)} order(s): {result}")
            return result
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def cancel_all_orders(self, symbol: str) -> dict:
        """Cancel every open order for a symbol (cleanup utility)."""
        if not config.LIVE_TRADING:
            return {"status": "simulated"}
        orders = self.get_open_orders()
        oids = [o.get("oid") for o in orders if o.get("coin") == symbol]
        if not oids:
            return {"status": "nothing_to_cancel"}
        return self.cancel_orders(symbol, oids)

    # ═════════════════════════════════════════════════════════════════════
    # LEVERAGE
    # ═════════════════════════════════════════════════════════════════════

    def set_leverage(self, symbol: str, leverage: int, is_cross: bool = True) -> dict:
        if not config.LIVE_TRADING or not self.account:
            return {"status": "simulated"}
        action = {
            "type": "updateLeverage",
            "asset": self.get_asset_index(symbol),
            "isCross": is_cross,
            "leverage": leverage,
        }
        result = self._send_action(action)
        log(f"Set leverage {leverage}x: {result}")
        return result

    # ═════════════════════════════════════════════════════════════════════
    # ORDER EXECUTION
    # ═════════════════════════════════════════════════════════════════════

    def place_market_order(self, symbol: str, is_buy: bool, size: float,
                           reduce_only: bool = False) -> dict:
        if not config.LIVE_TRADING:
            return {"status": "simulated", "symbol": symbol,
                    "side": "buy" if is_buy else "sell", "size": size}

        current_price = self.get_mid_price(symbol)
        limit_price = current_price * (1.005 if is_buy else 0.995)
        limit_price = self.round_price(symbol, limit_price)
        size = self.round_size(symbol, size)

        order = {
            "a": self.get_asset_index(symbol),
            "b": is_buy,
            "p": float_to_wire(limit_price),
            "s": float_to_wire(size),
            "r": reduce_only,
            "t": {"limit": {"tif": "Ioc"}},
        }
        action = {"type": "order", "orders": [order], "grouping": "na"}

        log(f"Market order: {'BUY' if is_buy else 'SELL'} {size} {symbol} @ ~${limit_price:,.0f}")
        try:
            result = self._send_action(action)
            log(f"Order result: {result}")
            return result
        except Exception as e:
            log(f"Error placing order: {e}")
            import traceback; traceback.print_exc()
            return {"status": "error", "message": str(e)}

    def place_sl_order(self, symbol: str, size: float, is_long: bool,
                       sl_price: float) -> dict:
        """Place SL as stop-market trigger. Returns result with oid."""
        if not config.LIVE_TRADING or not self.account:
            return {"status": "simulated", "sl": sl_price, "sl_oid": None}

        size = self.round_size(symbol, size)
        sl_price = self.round_price(symbol, sl_price)
        close_is_buy = not is_long

        sl_order = {
            "a": self.get_asset_index(symbol),
            "b": close_is_buy,
            "p": float_to_wire(sl_price),
            "s": float_to_wire(size),
            "r": True,
            "t": {"trigger": {"isMarket": True,
                               "triggerPx": float_to_wire(sl_price),
                               "tpsl": "sl"}},
        }
        action = {"type": "order", "orders": [sl_order], "grouping": "na"}

        log(f"SL trigger at ${sl_price:,.0f}")
        try:
            result = self._send_action(action)
            sl_oid = self.extract_oid(result)
            log(f"SL result: oid={sl_oid} {result}")
            result["sl_oid"] = sl_oid
            return result
        except Exception as e:
            log(f"Error placing SL: {e}")
            return {"status": "error", "message": str(e), "sl_oid": None}

    def update_trailing_stop(self, symbol: str, size: float, is_long: bool,
                             new_sl_price: float, old_sl_oid=None) -> dict:
        """
        Update trailing stop: cancel old SL by OID, place new one.
        Returns new result with sl_oid.
        """
        if old_sl_oid:
            self.cancel_orders(symbol, [old_sl_oid])
        else:
            self.cancel_all_orders(symbol)
        time.sleep(0.2)
        return self.place_sl_order(symbol, size, is_long, new_sl_price)

    def close_position(self, symbol: str) -> dict:
        """Close position: cancel all orders first, then market close."""
        self.cancel_all_orders(symbol)
        time.sleep(0.2)
        positions = self.get_open_positions()
        for pos in positions:
            if pos["symbol"] == symbol:
                is_buy = pos["size"] < 0
                return self.place_market_order(symbol, is_buy, abs(pos["size"]),
                                               reduce_only=True)
        return {"status": "no_position"}

    def cleanup_orphaned_orders(self, symbol: str, tracked_oids: set) -> int:
        """Cancel orders not associated with any tracked position."""
        if not config.LIVE_TRADING:
            return 0
        try:
            orders = self.get_open_orders()
            orphans = [
                o["oid"] for o in orders
                if o.get("coin") == symbol and o.get("oid")
                and int(o["oid"]) not in tracked_oids
            ]
            if orphans:
                log(f"Cleaning up {len(orphans)} orphaned orders")
                self.cancel_orders(symbol, orphans)
            return len(orphans)
        except Exception as e:
            log(f"Orphan cleanup error: {e}")
            return 0
