from enum import IntEnum

class HandRank(IntEnum):
    """Hand rankings from highest to lowest."""
    HIGH_CARD = 0
    ONE_PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8
    ROYAL_FLUSH = 9

class Suit(IntEnum):
    """Enum class for the suits of a card."""
    HEARTS = 0
    DIAMONDS = 1
    CLUBS = 2
    SPADES = 3

class Card():
    """Represents a playing card in a deck of cards."""
    def __init__(self, rank: int, suit: Suit):
        """
        Initialize a card with a rank and a suit.

        Args:
            rank: Card rank (1-13, where 1=Ace, 11=Jack, 12=Queen, 13=King)
            suit: Card suit (from Suit enum)
        """
        self.rank = rank
        self.suit = suit

    def __str__(self):
        """Return string representation of card."""
        rank_str = {1: 'A', 11: 'J', 12: 'Q', 13: 'K'}.get(self.rank, str(self.rank))
        suit_str = {
            Suit.HEARTS: '♥',
            Suit.DIAMONDS: '♦',
            Suit.CLUBS: '♣',
            Suit.SPADES: '♠'
        }[self.suit]
        return f"{rank_str}{suit_str}"

    def __eq__(self, other) -> bool:
        """Compare equality with another card."""
        if not isinstance(other, Card):
            return False
        return self.rank == other.rank and self.suit == other.suit

    def __repr__(self):
        """Return string representation of card."""
        return f"{self.rank} of {self.suit.name}"
    
    def __hash__(self) -> int:
        """Hash function for card."""
        return self.rank * 4 + self.suit