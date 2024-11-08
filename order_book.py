import heapq
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from agent import Agent, Order, Trade

from beartype import beartype as typed
from loguru import logger


class OrderBook:
    @typed
    def __init__(self, expiration_time: float | int = 10):
        """Initialize empty order book with two sides (bids and asks)"""
        # For bids (+1) we use positive price, for asks (-1) negative price
        self.heaps: dict[int, list[tuple[float, float, Order]]] = {
            +1: [],  # Bids: (-price, timestamp, Order)
            -1: [],  # Asks: (price, timestamp, Order)
        }
        self.expiration_time = expiration_time

    @typed
    def remove_orders_before(self, timestamp: float) -> None:
        """Remove all orders with timestamp older than the given timestamp"""
        for side in self.heaps.values():
            side[:] = [
                (price, timestamp, order)
                for price, timestamp, order in side
                if order.timestamp >= timestamp
            ]

    @typed
    def add_order(self, order: Order) -> list[Trade]:
        """
        Add new order to the book. If it matches with existing orders,
        execute trades and return them. Otherwise, add to book and return None.
        """
        direction = +1 if order.side == "buy" else -1
        trades = self._handle_order(order, direction)
        self.remove_orders_before(order.timestamp - self.expiration_time)
        return trades

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
        buy_order.size = min(buy_order.size, buy_order.agent.balance / price)
        sell_order.size = min(sell_order.size, sell_order.agent.position)
        trade_size = min(taker_order.size, maker_order.size)
        if trade_size < 1e-6:
            return None

        buy_agent = buy_order.agent
        sell_agent = sell_order.agent
        buy_agent.balance -= trade_size * price
        sell_agent.balance += trade_size * price
        buy_agent.position += trade_size
        sell_agent.position -= trade_size
        buy_order.size -= trade_size
        sell_order.size -= trade_size

        trade = Trade(
            price=price,
            size=trade_size,
            timestamp=max(taker_order.timestamp, maker_order.timestamp),
            maker_id=maker_order.agent.id,
            taker_id=taker_order.agent.id,
            taker_is_buy=(taker_direction == +1),
        )
        return trade

    @typed
    def _handle_order(self, order: Order, direction: int) -> list[Trade] | None:
        eps = 1e-6
        trades = []
        opposite_heap = self.heaps[-direction]

        while order.size > eps and opposite_heap:
            _, _, maker_order = opposite_heap[0]
            trade = self.execute(order, maker_order, direction)
            if trade is not None:
                trades.append(trade)

            if maker_order.size < eps:
                heapq.heappop(opposite_heap)

            if direction * (order.price - maker_order.price) < 0:
                break

        if order.size > eps:
            heapq.heappush(
                self.heaps[direction],
                (order.price * (-direction), order.timestamp, order),
            )
        return trades

    @typed
    def __repr__(self) -> str:
        if not (self.heaps[+1] or self.heaps[-1]):
            return "Empty order book"

        lines = []

        # Process bids (buy orders)
        bids = [(-p, t, o) for p, t, o in sorted(self.heaps[+1])]
        if bids:
            lines.append("Bids:")
            for _, _, order in bids[::-1]:
                lines.append(
                    f"  {order.price:8.2f} | size: {order.size:8.2f} | agent: {order.agent.id} | timestamp: {order.timestamp:.2f}"
                )

        # Process asks (sell orders)
        asks = [(p, t, o) for p, t, o in sorted(self.heaps[-1])]
        if asks:
            if bids:
                lines.append("-" * 50)
            lines.append("Asks:")
            for _, _, order in asks:
                lines.append(
                    f"  {order.price:8.2f} | size: {order.size:8.2f} | agent: {order.agent.id} | timestamp: {order.timestamp:.2f}"
                )

        return "\n".join(lines)


if __name__ == "__main__":
    book = OrderBook()
    np.random.seed(42)

    # Generate random orders
    n_orders = 20
    mean_price = 100.0
    price_std = 2.0
    size_mean = 5.0
    size_std = 2.0
    agent = Agent(
        id=0,
        alpha=0.5,
        k=1.0,
        sigma=0.0,
        wealth=1e3,
        delay=0.0,
        aggressiveness=0.5,
        uncertainty=0.0,
    )
    agent.position = 1000.0

    # Generate buy and sell orders with Gaussian distributed prices and sizes
    for i in range(n_orders):
        # Random price and size
        price = np.random.normal(mean_price, price_std)
        size = abs(np.random.normal(size_mean, size_std))

        # Randomly choose buy/sell
        side = "buy" if np.random.random() < 0.5 else "sell"

        order = Order(
            side=side,
            price=price,
            size=size,
            timestamp=float(i),
            agent=agent,
        )
        print(order)
        trades = book.add_order(order)
        logger.debug(f"Trades: {trades}")
        print(book)
        input()
