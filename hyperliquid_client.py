"""
Hyperliquid API Client — v14 High-Conviction Bot
Handles market data, order execution, leverage management, and trailing stop updates.
Ported from hl-ml-ichimoku-long-2x with additions for SHORT support and trailing stop.
"""
import json
import time

import requests
from eth_account import Account

import config

try:
    import msgpack
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False
    print("Warning: msgpack not installed. Install with: pip install msgpack")


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
        self.private_key = config.PRIVATE_KEY
        self.session = requests.Session()
        self.is_mainnet = not config.TESTNET

        if self.private_key:
            key = self.private_key if self.private_key.startswith("0x") else "0x" + self.private_key
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
        self._asset_raw = {}

    # ═════════════════════════════════════════════════════════════════════
    # MARKET DATA
    # ═════════════════════════════════════════════════════════════════════

    def get_mid_price(self, symbol: str) -> float:
        payload = {"type": "allMids"}
        resp = self.session.post(f"{self.api_url}/info", json=payload)
        resp.raise_for_status()
        return float(resp.json().get(symbol, 0))

    def get_meta(self) -> dict:
        if self._asset_to_index is None:
            payload = {"type": "meta"}
            resp = self.session.post(f"{self.api_url}/info", json=payload)
            resp.raise_for_status()
            meta = resp.json()
            self._asset_to_index = {}
            for i, asset in enumerate(meta.get("universe", [])):
                name = asset.get("name")
                self._asset_to_index[name] = i
                self._asset_sz_decimals[name] = asset.get("szDecimals", 5)
                self._asset_raw[name] = asset
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

    def get_account_state(self, address: str = None) -> dict:
        addr = address or self.wallet_address
        payload = {"type": "clearinghouseState", "user": addr}
        resp = self.session.post(f"{self.api_url}/info", json=payload)
        resp.raise_for_status()
        return resp.json()

    def get_open_positions(self, address: str = None) -> list:
        state = self.get_account_state(address)
        positions = []
        for ap in state.get("assetPositions", []):
            pos = ap.get("position", {})
            szi = float(pos.get("szi", 0))
            if szi != 0:
                positions.append({
                    "symbol": pos.get("coin"),
                    "size": szi,
                    "entry_price": float(pos.get("entryPx", 0)),
                    "unrealized_pnl": float(pos.get("unrealizedPnl", 0)),
                    "leverage": float(pos.get("leverage", {}).get("value", 1)) if isinstance(pos.get("leverage"), dict) else 1,
                })
        return positions

    def get_account_value(self, address: str = None) -> float:
        try:
            addr = address or self.wallet_address
            perp_state = self.get_account_state(addr)
            margin = perp_state.get("marginSummary", {})
            cross = perp_state.get("crossMarginSummary", {})
            perp_val = float(margin.get("accountValue", 0)) or float(cross.get("accountValue", 0)) or float(perp_state.get("withdrawable", 0))
            log(f"Account value: ${perp_val:.2f}")
            return perp_val
        except Exception as e:
            log(f"Error getting account value: {e}")
            return 0.0

    def get_open_orders(self, address: str = None) -> list:
        addr = address or self.wallet_address
        payload = {"type": "openOrders", "user": addr}
        resp = self.session.post(f"{self.api_url}/info", json=payload)
        resp.raise_for_status()
        return resp.json()

    # ═════════════════════════════════════════════════════════════════════
    # L1 ACTION SIGNING
    # ═════════════════════════════════════════════════════════════════════

    def _sign_l1_action(self, action: dict, nonce: int, vault_address: str = None) -> dict:
        if not self.account:
            raise ValueError("Private key not configured")
        if not HAS_MSGPACK:
            raise ImportError("msgpack required for signing")

        from eth_utils import keccak

        data = msgpack.packb(action)
        data += nonce.to_bytes(8, "big")
        data += b"\x00" if vault_address is None else (b"\x01" + address_to_bytes(vault_address))

        connection_id = keccak(data)
        source = "a" if self.is_mainnet else "b"
        phantom_agent = {"source": source, "connectionId": connection_id}

        typed_data = {
            "domain": {
                "chainId": 1337,
                "name": "Exchange",
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

    # ═════════════════════════════════════════════════════════════════════
    # LEVERAGE
    # ═════════════════════════════════════════════════════════════════════

    def set_leverage(self, symbol: str, leverage: int, is_cross: bool = True) -> dict:
        if not config.LIVE_TRADING or not self.account:
            return {"status": "simulated"}
        asset_idx = self.get_asset_index(symbol)
        nonce = int(time.time() * 1000)
        action = {
            "type": "updateLeverage",
            "asset": asset_idx,
            "isCross": is_cross,
            "leverage": leverage,
        }
        sig = self._sign_l1_action(action, nonce)
        payload = {"action": action, "nonce": nonce, "signature": sig}
        resp = self.session.post(f"{self.api_url}/exchange", json=payload)
        result = resp.json() if resp.text else {}
        log(f"Set leverage {leverage}x: {result}")
        return result

    # ═════════════════════════════════════════════════════════════════════
    # ORDER EXECUTION
    # ═════════════════════════════════════════════════════════════════════

    def place_market_order(self, symbol: str, is_buy: bool, size: float, reduce_only: bool = False) -> dict:
        if not config.LIVE_TRADING:
            return {"status": "simulated", "symbol": symbol, "side": "buy" if is_buy else "sell", "size": size}
        if not self.account:
            raise ValueError("Private key not configured")

        current_price = self.get_mid_price(symbol)
        limit_price = current_price * (1.005 if is_buy else 0.995)
        limit_price = self.round_price(symbol, limit_price)

        asset_idx = self.get_asset_index(symbol)
        sz_dec = self.get_sz_decimals(symbol)
        size = round(size, sz_dec)

        nonce = int(time.time() * 1000)
        order = {
            "a": asset_idx,
            "b": is_buy,
            "p": float_to_wire(limit_price),
            "s": float_to_wire(size),
            "r": reduce_only,
            "t": {"limit": {"tif": "Ioc"}},
        }
        action = {"type": "order", "orders": [order], "grouping": "na"}

        try:
            sig = self._sign_l1_action(action, nonce)
            payload = {"action": action, "nonce": nonce, "signature": sig}
            log(f"Market order: {'BUY' if is_buy else 'SELL'} {size} {symbol} @ ~${limit_price:,.0f}")
            resp = self.session.post(f"{self.api_url}/exchange", json=payload, headers={"Content-Type": "application/json"})
            result = resp.json() if resp.text else {}
            log(f"Order result: {result}")
            return result
        except Exception as e:
            log(f"Error placing order: {e}")
            import traceback; traceback.print_exc()
            return {"status": "error", "message": str(e)}

    def place_sl_order(self, symbol: str, size: float, is_long: bool, sl_price: float) -> dict:
        """Place a stop-loss trigger order for an existing position."""
        if not config.LIVE_TRADING or not self.account:
            return {"status": "simulated", "sl": sl_price}

        asset_idx = self.get_asset_index(symbol)
        size = round(size, self.get_sz_decimals(symbol))
        sl_price = self.round_price(symbol, sl_price)

        close_is_buy = not is_long
        sl_order = {
            "a": asset_idx,
            "b": close_is_buy,
            "p": float_to_wire(sl_price),
            "s": float_to_wire(size),
            "r": True,
            "t": {"trigger": {"isMarket": True, "triggerPx": float_to_wire(sl_price), "tpsl": "sl"}},
        }

        nonce = int(time.time() * 1000)
        action = {"type": "order", "orders": [sl_order], "grouping": "na"}
        try:
            sig = self._sign_l1_action(action, nonce)
            payload = {"action": action, "nonce": nonce, "signature": sig}
            log(f"SL trigger at ${sl_price:,.0f}")
            resp = self.session.post(f"{self.api_url}/exchange", json=payload)
            result = resp.json() if resp.text else {}
            log(f"SL result: {result}")
            return result
        except Exception as e:
            log(f"Error placing SL: {e}")
            return {"status": "error", "message": str(e)}

    def cancel_all_orders(self, symbol: str) -> dict:
        """Cancel all open orders (trigger + limit) for a symbol."""
        if not config.LIVE_TRADING or not self.account:
            return {"status": "simulated"}

        try:
            orders = self.get_open_orders()
            asset_idx = self.get_asset_index(symbol)
            cancelled = 0
            for order in orders:
                oid = order.get("oid")
                if oid is None:
                    continue
                nonce = int(time.time() * 1000)
                action = {"type": "cancel", "cancels": [{"a": asset_idx, "o": oid}]}
                sig = self._sign_l1_action(action, nonce)
                payload = {"action": action, "nonce": nonce, "signature": sig}
                self.session.post(f"{self.api_url}/exchange", json=payload)
                cancelled += 1
                time.sleep(0.05)
            log(f"Cancelled {cancelled} orders for {symbol}")
            return {"status": "ok", "cancelled": cancelled}
        except Exception as e:
            log(f"Error cancelling orders: {e}")
            return {"status": "error", "message": str(e)}

    def update_trailing_stop(self, symbol: str, size: float, is_long: bool, new_sl_price: float) -> dict:
        """
        Update trailing stop: cancel existing SL trigger, place new one at higher level.
        HL has no native trailing stop so we manage it manually.
        """
        self.cancel_all_orders(symbol)
        time.sleep(0.2)
        return self.place_sl_order(symbol, size, is_long, new_sl_price)

    def close_position(self, symbol: str) -> dict:
        positions = self.get_open_positions()
        for pos in positions:
            if pos["symbol"] == symbol:
                is_buy = pos["size"] < 0  # short -> buy to close
                size = abs(pos["size"])
                self.cancel_all_orders(symbol)
                time.sleep(0.2)
                return self.place_market_order(symbol, is_buy, size, reduce_only=True)
        return {"status": "no_position", "symbol": symbol}
