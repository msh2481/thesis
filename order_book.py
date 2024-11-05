import heapq
from dataclasses import dataclass
from typing import Literal

from beartype import beartype as typed


@dataclass
class Order:
    side: Literal["buy", "sell"]
    price: float
    size: float
    timestamp: float
    agent_id: int


@dataclass
class Trade:
    price: float
    size: float
    timestamp: float
    maker_id: int
    taker_id: int


class OrderBook:
    @typed
    def __init__(self):
        """Initialize empty order book with two sides (bids and asks)"""
        # For bids we use negative price for correct ordering in heap
        self.bids: list[tuple[float, float, Order]] = []  # (-price, timestamp, Order)
        self.asks: list[tuple[float, float, Order]] = []  # (price, timestamp, Order)

    @typed
    def add_order(self, order: Order) -> list[Trade] | None:
        """
        Add new order to the book. If it matches with existing orders,
        execute trades and return them. Otherwise, add to book and return None.
        """
        if order.side == "buy":
            return self._handle_buy(order)
        else:
            return self._handle_sell(order)

    @typed
    def get_best_bid(self) -> float | None:
        """Get highest bid price"""
        if not self.bids:
            return None
        neg_price, _, _ = self.bids[0]
        return -neg_price

    @typed
    def get_best_ask(self) -> float | None:
        """Get lowest ask price"""
        if not self.asks:
            return None
        price, _, _ = self.asks[0]
        return price

    @typed
    def get_mid_price(self) -> float | None:
        """Get mid price between best bid and ask"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid is None or best_ask is None:
            return None
        return (best_bid + best_ask) / 2

    @typed
    def _handle_buy(self, order: Order) -> list[Trade] | None:
        trades = []
        remaining_size = order.size

        while remaining_size > 0 and self.asks:
            ask_price, _, ask_order = self.asks[0]

            if ask_price > order.price:  # No match possible
                break

            heapq.heappop(self.asks)  # Remove the ask we're about to match

            trade_size = min(remaining_size, ask_order.size)
            trades.append(
                Trade(
                    price=ask_price,
                    size=trade_size,
                    timestamp=order.timestamp,
                    maker_id=ask_order.agent_id,
                    taker_id=order.agent_id,
                )
            )

            remaining_size -= trade_size
            if trade_size < ask_order.size:  # Put back remaining size
                ask_order.size -= trade_size
                heapq.heappush(self.asks, (ask_price, ask_order.timestamp, ask_order))

        if remaining_size > 0:  # Add remaining order to book
            order.size = remaining_size
            heapq.heappush(self.bids, (-order.price, order.timestamp, order))
            return None

        return trades

    @typed
    def _handle_sell(self, order: Order) -> list[Trade] | None:
        trades = []
        remaining_size = order.size

        while remaining_size > 0 and self.bids:
            neg_bid_price, _, bid_order = self.bids[0]
            bid_price = -neg_bid_price

            if bid_price < order.price:  # No match possible
                break

            heapq.heappop(self.bids)  # Remove the bid we're about to match

            trade_size = min(remaining_size, bid_order.size)
            trades.append(
                Trade(
                    price=bid_price,
                    size=trade_size,
                    timestamp=order.timestamp,
                    maker_id=bid_order.agent_id,
                    taker_id=order.agent_id,
                )
            )

            remaining_size -= trade_size
            if trade_size < bid_order.size:  # Put back remaining size
                bid_order.size -= trade_size
                heapq.heappush(
                    self.bids, (neg_bid_price, bid_order.timestamp, bid_order)
                )

        if remaining_size > 0:  # Add remaining order to book
            order.size = remaining_size
            heapq.heappush(self.asks, (order.price, order.timestamp, order))
            return None

        return trades
