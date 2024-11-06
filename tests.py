import unittest
from dataclasses import dataclass

from order_book import Order, OrderBook, Trade


@dataclass
class MockAgent:
    """Mock agent for testing"""

    balance: float = 1000.0  # Large enough balance for test trades
    position: float = 1000.0  # Large enough position for test trades


class TestOrderBook(unittest.TestCase):
    def setUp(self):
        self.book = OrderBook()

    def test_empty_book_best_prices(self):
        """Test that empty book returns None for best prices"""
        self.assertIsNone(self.book.get_best_bid())
        self.assertIsNone(self.book.get_best_ask())
        self.assertIsNone(self.book.get_mid_price())

    def test_single_buy_order(self):
        """Test adding a single buy order"""
        order = Order(
            side="buy",
            price=100.0,
            size=1.0,
            timestamp=1.0,
            agent_id=1,
            agent=MockAgent(),
        )
        result = self.book.add_order(order)

        self.assertIsNone(result)  # No trades executed
        self.assertEqual(self.book.get_best_bid(), 100.0)
        self.assertIsNone(self.book.get_best_ask())

    def test_single_sell_order(self):
        """Test adding a single sell order"""
        order = Order(
            side="sell",
            price=100.0,
            size=1.0,
            timestamp=1.0,
            agent_id=1,
            agent=MockAgent(),
        )
        result = self.book.add_order(order)

        self.assertIsNone(result)  # No trades executed
        self.assertIsNone(self.book.get_best_bid())
        self.assertEqual(self.book.get_best_ask(), 100.0)

    def test_matching_orders(self):
        """Test matching buy and sell orders"""
        sell_order = Order(
            side="sell",
            price=100.0,
            size=1.0,
            timestamp=1.0,
            agent_id=1,
            agent=MockAgent(),
        )
        self.book.add_order(sell_order)

        buy_order = Order(
            side="buy",
            price=100.0,
            size=1.0,
            timestamp=2.0,
            agent_id=2,
            agent=MockAgent(),
        )
        trades = self.book.add_order(buy_order)

        self.assertIsNotNone(trades)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].price, 100.0)
        self.assertEqual(trades[0].size, 1.0)
        self.assertEqual(trades[0].maker_id, 1)
        self.assertEqual(trades[0].taker_id, 2)

        # Book should be empty after full match
        self.assertIsNone(self.book.get_best_bid())
        self.assertIsNone(self.book.get_best_ask())

    def test_partial_fill(self):
        """Test partial order filling"""
        sell_order = Order(
            side="sell",
            price=100.0,
            size=2.0,
            timestamp=1.0,
            agent_id=1,
            agent=MockAgent(),
        )
        self.book.add_order(sell_order)

        buy_order = Order(
            side="buy",
            price=100.0,
            size=1.0,
            timestamp=2.0,
            agent_id=2,
            agent=MockAgent(),
        )
        trades = self.book.add_order(buy_order)

        self.assertIsNotNone(trades)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].size, 1.0)

        # Check remaining order
        self.assertIsNone(self.book.get_best_bid())
        self.assertEqual(self.book.get_best_ask(), 100.0)

    def test_price_time_priority(self):
        """Test that orders are matched according to price-time priority"""
        # Add two sell orders at different prices
        self.book.add_order(Order("sell", 101.0, 1.0, 1.0, 1, agent=MockAgent()))
        self.book.add_order(Order("sell", 100.0, 1.0, 2.0, 2, agent=MockAgent()))
        # Buy order should match with the better price
        buy_order = Order("buy", 101.0, 1.0, 3.0, 3, agent=MockAgent())
        trades = self.book.add_order(buy_order)

        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].price, 100.0)
        self.assertEqual(trades[0].maker_id, 2)

    def test_mid_price(self):
        """Test mid price calculation"""
        self.book.add_order(Order("buy", 99.0, 1.0, 1.0, 1, agent=MockAgent()))
        self.book.add_order(Order("sell", 101.0, 1.0, 2.0, 2, agent=MockAgent()))

        self.assertEqual(self.book.get_mid_price(), 100.0)


if __name__ == "__main__":
    unittest.main()
