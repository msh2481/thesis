import heapq
from dataclasses import dataclass
from typing import Any, Literal

from beartype import beartype as typed
from beartype.typing import Self


@dataclass
class Order:
    side: Literal["buy", "sell"]
    price: float
    size: float
    timestamp: float
    agent_id: int
    agent: Any  # Reference to the agent placing the order


@dataclass
class Trade:
    price: float
    size: float
    timestamp: float
    maker_id: int
    taker_id: int
    taker_is_buy: bool


@typed
@dataclass
class Candle:
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    num_trades: int

    @classmethod
    def from_trades(cls, trades: list[Trade], start_time: float) -> Self | None:
        """
        Construct a candle from a list of trades that occurred in the period.

        Args:
            trades: List of trades in chronological order
            start_time: Starting timestamp of the period

        Returns:
            Candle object or None if no trades occurred
        """
        if not trades:
            return None

        prices = [trade.price for trade in trades]
        volumes = [trade.size for trade in trades]

        return cls(
            timestamp=start_time,
            open=prices[0],
            high=max(prices),
            low=min(prices),
            close=prices[-1],
            volume=sum(volumes),
            num_trades=len(trades),
        )


class OrderBook:
    @typed
    def __init__(self):
        """Initialize empty order book with two sides (bids and asks)"""
        # For bids (+1) we use positive price, for asks (-1) negative price
        self.heaps: dict[int, list[tuple[float, float, Order]]] = {
            +1: [],  # Bids: (-price, timestamp, Order)
            -1: [],  # Asks: (price, timestamp, Order)
        }

    @typed
    def add_order(self, order: Order) -> list[Trade] | None:
        """
        Add new order to the book. If it matches with existing orders,
        execute trades and return them. Otherwise, add to book and return None.
        """
        direction = +1 if order.side == "buy" else -1
        return self._handle_order(order, direction)

    @typed
    def get_best_bid(self) -> float | None:
        """Get highest bid price"""
        if not self.heaps[+1]:
            return None
        neg_price, _, _ = self.heaps[+1][0]
        return -neg_price

    @typed
    def get_best_ask(self) -> float | None:
        """Get lowest ask price"""
        if not self.heaps[-1]:
            return None
        price, _, _ = self.heaps[-1][0]
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
    def execute(
        self, taker_order: Order, maker_order: Order, taker_direction: int
    ) -> Trade | None:
        """Execute a trade between two orders"""
        taker_price = taker_order.price
        maker_price = maker_order.price
        if taker_direction * (taker_price - maker_price) < 0:
            return None

        price = maker_price

        # Execute trade
        buy_order = taker_order if taker_direction == +1 else maker_order
        sell_order = maker_order if taker_direction == +1 else taker_order

        trade_size = min(taker_order.size, maker_order.size)
        trade_size = min(trade_size, buy_order.agent.balance / price)
        trade_size = min(trade_size, sell_order.agent.position)

        if trade_size < 1e-6:
            return None

        buy_agent = buy_order.agent
        sell_agent = sell_order.agent
        buy_agent.balance -= trade_size * price
        sell_agent.balance += trade_size * price
        buy_agent.position += trade_size
        sell_agent.position -= trade_size

        trade = Trade(
            price=price,
            size=trade_size,
            timestamp=max(taker_order.timestamp, maker_order.timestamp),
            maker_id=maker_order.agent_id,
            taker_id=taker_order.agent_id,
            taker_is_buy=(taker_direction == +1),
        )
        return trade

    @typed
    def _handle_order(self, order: Order, direction: int) -> list[Trade] | None:
        trades = []
        opposite_heap = self.heaps[-direction]

        while order.size > 0 and opposite_heap:
            maker_price_signed, _, maker_order = opposite_heap[0]
            trade = self.execute(order, maker_order, direction)
            if trade is None:
                break

            heapq.heappop(opposite_heap)
            trades.append(trade)

            order.size -= trade.size
            maker_order.size -= trade.size

            if maker_order.size > 0:  # Put back remaining size
                heapq.heappush(
                    opposite_heap,
                    (maker_price_signed, maker_order.timestamp, maker_order),
                )

        if order.size > 0:  # Add remaining order to book
            heapq.heappush(
                self.heaps[direction],
                (order.price * (-direction), order.timestamp, order),
            )
            return trades if trades else None

        return trades
