import random
import numpy as np
from enum import IntEnum
from card import Card, Suit, HandRank
from typing import List, Dict, Tuple, Optional, Any

class Action(IntEnum):
    """Possible poker actions."""
    FOLD   =    0
    CHECK  =    1
    CALL   =    2
    RAISE  =    3
    ALL_IN =    4

class PokerEnv:
    """
    Poker enviroment.
    
    Implements a Texas Hold'em poker game.
    Follows OpenAI Gym API.
    """
    def __init__(self,
                 num_players: int = 2,
                 starting_chips: int = 1000,
                 small_blind: int = 10,
                 big_blind: int = 20
    ) -> None:
        """
        Initialize the poker enviroment.
        
        Args:
            num_players: Number of players in the game.
            starting_chips: Number of chips each player starts with.
            small_blind: Small blind amount.
            big_blind: Big blind amount.
        """
        if num_players < 2:
            raise ValueError("Number of players must be at least 2.")
        
        self.num_players = num_players
        self.starting_chips = starting_chips
        self.small_blind = small_blind
        self.big_blind = big_blind
        
        
        self.deck = []
        self.community_cards = []
        self.player_hands = [[] for _ in range(num_players)]
        self.player_chips = np.zeros(num_players, dtype=np.int32)
        self.player_bets = np.zeros(num_players, dtype=np.int32)
        self.folded_players = set()
        self.all_in_players = set()


        self.current_player = 0
        self.pot = 0
        self.dealer_position = 0
        self.current_min_raise = big_blind

        # Reset to initialize.
        self.reset()

    def _generate_deck(self) -> List[Tuple[int, int]]:
        """Generate a deck of cards."""
        return [
                Card(rank, suit)
                for rank in range(1, 14)
                for suit in range(4)
            ]
    
    def _build_card_lookup(self):
        """Build lookup table for card evaluation."""
        self._all_cards = self._generate_deck()
        
        self._rank_lookup = {}
        for rank in range(1, 14):
            self._rank_lookup[rank] = rank if rank > 1 else 14

    def reset(self) -> Dict[str, Any]:
        """
        Reset the enviroment to initial state.
        
        Returns:
            Initial observation of the enviroment.
        """
        # Reset deck.
        self.deck = self._all_cards.copy()
        random.shuffle(self.deck)

        # Reset community cards.
        self.community_cards = []
        self.player_hands = [[] for _ in range(self.num_players)]
        self.player_chips.fill(self.starting_chips)
        self.player_bets.fill(0)
        self.pot = 0
        self.folded_players = set()
        self.all_in_players = set()

        self.dealer_position = (self.dealer_position + 1) % self.num_players

        self._deal_hands()
        self._post_blinds()
        
        self.current_player = (self.dealer_position + 3) % self.num_players
        if self.num_players == 2:
            self.current_player = (self.dealer_position + 1) % self.num_players

        self._advance_to_next_valid_player()

        return self._get_observation()
    
    def _deal_hands(self) -> None:
        """Deal two cards to each player."""
        for _ in range(2):
            for i in range(self.num_players):
                self.player_hands[i].append(self.deck.pop())
    
    def _post_blinds(self) -> None:
        """Post small and big blinds."""
        small_blind_pos = (self.dealer_position + 1) % self.num_players
        big_blind_pos = (self.dealer_position + 2) % self.num_players

        if self.num_players == 2:
            small_blind_pos = self.dealer_position
            big_blind_pos = (self.dealer_position + 1) % self.num_players
        
        self.player_bets[small_blind_pos] = min(self.small_blind, self.player_chips[small_blind_pos])
        self.player_chips[small_blind_pos] -= self.player_bets[small_blind_pos]
        if self.player_chips[small_blind_pos] == 0:
            self.all_in_players.add(small_blind_pos)
        
        self.player_bets[big_blind_pos] = min(self.big_blind, self.player_chips[big_blind_pos])
        self.player_chips[big_blind_pos] -= self.player_bets[big_blind_pos]
        if self.player_chips[big_blind_pos] == 0:
            self.all_in_players.add(big_blind_pos)
        
        self.current_min_raise = self.big_blind

    def _deal_flop(self) -> None:
        """Deal the flop."""
        self.deck.pop()
        
        self.community_cards.extend(self.deck.pop() for _ in range(3))
    
    def _deal_turn_or_river(self):
        """Deal a single community card (turn or river)."""
        # Burn a card
        self.deck.pop()
        
        # Deal one card
        self.community_cards.append(self.deck.pop())
    
    def _advance_to_next_valid_player(self):
        """Advance to the next player who hasn't folded or gone all-in."""
        active_players =  self.num_players - len(self.folded_players) - len(self.all_in_players)

        if active_players <= 1:
            return
        
        initial_player = self.current_player
        while True:
            self.current_player = (self.current_player + 1) % self.num_players
            if (self.current_player not in self.folded_players and 
                self.current_player not in self.all_in_players):
                break
            
            if self.current_player == initial_player:
                break
    
    def _get_observation(self) -> Dict[str, Any]:
        """
        Get current observation of the environment.
        
        Returns:
            Dictionary containing the current game state
        """
        return {
            'player_id': self.current_player,
            'hand': self.player_hands[self.current_player].copy() if self.current_player < len(self.player_hands) else [],
            'community_cards': self.community_cards.copy(),
            'pot': self.pot,
            'player_chips': self.player_chips.copy(),
            'player_bets': self.player_bets.copy(),
            'folded_players': list(self.folded_players),
            'all_in_players': list(self.all_in_players),
            'current_min_raise': self.current_min_raise,
            'dealer_position': self.dealer_position,
            'num_players': self.num_players,
            'valid_actions': self._get_valid_actions()
        }

    def _get_valid_actions(self) -> List[Tuple[Action, int]]:
        """
        Get valid actions for the current player.

        - Player can always fold.
        - If no one has bet, player can check or raise.
        - If player has chips, they can call, check, raise or go all-in.
        
        Returns:
            List of tuples containing Action and the corresponding amount.
    
        Returns:
            List of valid actions for the current player
        """
        valid_actions = []
        player_id = self.current_player
        player_chips = self.player_chips[player_id]
        current_bet = self.player_bets[player_id]
        max_bet = max(self.player_bets)
        call_amount = max_bet - current_bet

        valid_actions.append((Action.FOLD, 0))

        if call_amount == 0:
            valid_actions.append((Action.CHECK, call_amount))

        if call_amount > 0 and call_amount <= player_chips:
            valid_actions.append((Action.CALL, call_amount))
        
        min_raise = call_amount + self.current_min_raise
        
        if min_raise <= player_chips:
            valid_actions.append((Action.RAISE, min_raise))
        
        if player_chips > 0:
            valid_actions.append((Action.ALL_IN, player_chips))
        
        return valid_actions
    
    def step(self, action: Tuple[Action, int]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Take a step in the enviroment.

        Args:
            action: Tuple containing (action_type, amount).
                action_type: Type of action (fold, check, call, raise, all-in)
                amount: Amount to bet or raise
        Returns:
            Tuple containing:
                - observation: Current observation of the enviroment
                - reward: Reward received
                - done: Whether the episode is done
                - info: Additional information (if any)
        """
        action_type, amount = action
        player_id = self.current_player

        # Validate action
        valid_actions = self._get_valid_actions()
        if (action_type, amount) not in valid_actions:
            action_type, amount = valid_actions[0] if valid_actions else (Action.FOLD, 0)
        
        if action_type == Action.FOLD:
            self.folded_players.add(player_id)
        elif action_type == Action.CHECK:
            pass
        elif action_type == Action.CALL:
            max_bet = max(self.player_bets)
            call_amount = max_bet - self.player_bets[player_id]
            self.player_chips[player_id] -= call_amount
            self.player_bets[player_id] += call_amount

            if self.player_chips[player_id] == 0:
                self.all_in_players.add(player_id)
        elif action_type == Action.RAISE:
            current_bet = self.player_bets[player_id]
            max_bet = max(self.player_bets)
            call_amount = max_bet - current_bet
            raise_amount = amount - call_amount

            self.player_chips[player_id] -= amount
            self.player_bets[player_id] += amount

            self.current_min_raise = max(self.current_min_raise, raise_amount)

            if self.player_chips[player_id] == 0:
                self.all_in_players.add(player_id)
        elif action_type == Action.ALL_IN:
            self.player_bets[player_id] += amount
            self.player_chips[player_id] = 0
            self.all_in_players.add(player_id)

            current_bet = self.player_bets[player_id]
            max_bet_before = max(self.player_bets[i] if i != player_id else 0 for i in range(self.num_players))
            if current_bet > max_bet_before:
                self.current_min_raise = max(self.current_min_raise, current_bet - max_bet_before)
        
        betting_round_over = self._is_betting_round_over()

        if betting_round_over:
            self._advance_game_state()
        else:
            self._advance_to_next_valid_player()
        
        hand_over = self._is_hand_over()
        rewards = 0.0
        if hand_over:
            rewards = self._calculate_rewards()
        observation = self._get_observation()
        info = {
            'hand_over': hand_over,
            'active_players': self.num_players - len(self.folded_players),
        }

        return observation, rewards, hand_over, info
    
    def _is_betting_round_over(self) -> bool:
        """
        Check if the current betting round is over.
        
        Returns:
            True if betting round is over, False otherwise
        """
        # Count active players (not folded, not all-in)
        active_players = [
            i for i in range(self.num_players)
            if i not in self.folded_players and i not in self.all_in_players
        ]
        
        # If 0 or 1 active players, betting is over
        if len(active_players) <= 1:
            return True
        
        # Check if all active players have bet the same amount
        active_bets = [self.player_bets[i] for i in active_players]
        return len(set(active_bets)) == 1
    
    def _advance_game_state(self):
        """Advance the game state to the next stage."""
        # Move bets to pot
        self.pot += sum(self.player_bets)
        self.player_bets.fill(0)
        
        # Check if hand is over due to only one player remaining
        active_players = [i for i in range(self.num_players) if i not in self.folded_players]
        if len(active_players) <= 1:
            return
        
        # Deal community cards based on current stage
        if len(self.community_cards) == 0:
            self._deal_flop()
        elif len(self.community_cards) == 3 or len(self.community_cards) == 4:
            self._deal_turn_or_river()
        
        # Reset current player to first active player after dealer
        self.current_player = self.dealer_position
        self._advance_to_next_valid_player()
    
    def _is_hand_over(self) -> bool:
        """
        Check if the current hand is over.
        
        Returns:
            True if hand is over, False otherwise
        """
        # Hand is over if only one player hasn't folded
        active_players = [i for i in range(self.num_players) if i not in self.folded_players]
        if len(active_players) == 1:
            return True
        
        # Hand is over if all community cards are dealt and betting is complete
        if len(self.community_cards) == 5 and self._is_betting_round_over():
            return True
        
        # Hand is over if all players are either folded or all-in
        return len(self.folded_players) + len(self.all_in_players) == self.num_players
    
    def _evaluate_hand(self, player_id: int) -> Tuple[HandRank, List[int]]:
        """
        Evaluate the poker hand strength for a player.
        
        Args:
            player_id: Player ID to evaluate
        
        Returns:
            Tuple of (hand_rank, kickers) where kickers are used for tie-breaking
        """
        if player_id in self.folded_players:
            return HandRank.HIGH_CARD, []
        
        # Combine player's hole cards with community cards
        cards = self.player_hands[player_id] + self.community_cards
        
        # Convert to ranks and suits
        ranks = [card.rank for card in cards]
        suits = [card.suit for card in cards]
        
        rank_counts = {rank: ranks.count(rank) for rank in set(ranks)}
        suit_counts = {suit: suits.count(suit) for suit in set(suits)}    
        flush_suit = next((suit for suit, count in suit_counts.items() if count >=5), None)
        
        # Check for straight
        unique_ranks = sorted(set(ranks))
        straight_high_card = None
        
        # Special case: A-5 straight
        if 1 in rank_counts and all(r in rank_counts for r in range(2, 6)):
            straight_high_card = 5
        
        if len(unique_ranks) >= 5:
            for i in range(len(unique_ranks) - 4):
                if (unique_ranks[i + 4] - unique_ranks[i]) == 4:
                    straight_high_card = unique_ranks[i+4]
        
        # Check for regular straights
        for i in range(len(unique_ranks) - 4):
            if (unique_ranks[i] - unique_ranks[i + 4]) == 4:
                straight_high_card = unique_ranks[i]
                break
        
        # Check for straight flush and royal flush
        straight_flush_high_card = None
        if flush_suit is not None and straight_high_card is not None:
            flush_cards = [card.rank for card  in cards if card.suit == flush_suit]
            flush_ranks = sorted(set(flush_cards))

            # Special case: A-5 straight flush
            if 1 in flush_ranks and all(r in flush_ranks for r in range(2, 6)):
                straight_flush_high_card = 5
            
            # Check for regular straight flushes
            if len(flush_ranks) >= 5:
                for i in range(len(flush_ranks) - 4):
                    if (flush_ranks[i + 4] - flush_ranks[i]) == 4:
                        straight_flush_high_card = flush_ranks[i+4]
        
        # Find hand rank and kickers
        if straight_flush_high_card:
            if straight_flush_high_card == 14 or (straight_flush_high_card == 13 and 1 in rank_counts):
                return HandRank.ROYAL_FLUSH, []
            return HandRank.STRAIGHT_FLUSH, [straight_flush_high_card]
        
        # Four of a kind
        four_of_a_kind = next((rank for rank, count in rank_counts.items() if count == 4), None)
        if four_of_a_kind:
            kickers = sorted([r for r in ranks if r != four_of_a_kind], reverse=True)
            return HandRank.FOUR_OF_A_KIND, [four_of_a_kind, kickers[0]]
        
        # Full House
        three_of_a_kind = next((rank for rank, count in rank_counts.items() if count == 3), None)
        pairs = [rank for rank, count in rank_counts.items() if count == 2]
        
        if three_of_a_kind and pairs:
            return HandRank.FULL_HOUSE, [three_of_a_kind, max(pairs)]
        
        # Flush
        if flush_suit:
            flush_cards = sorted([card.rank for card in cards if card.suit == flush_suit], reverse=True)
            return HandRank.FLUSH, flush_cards[:5]
        
        # Straight
        if straight_high_card:
            return HandRank.STRAIGHT, [straight_high_card]
        
        # Three of a Kind
        if three_of_a_kind:
            kickers = sorted([r for r in ranks if r != three_of_a_kind], reverse=True)
            return HandRank.THREE_OF_A_KIND, [three_of_a_kind, kickers[0], kickers[1]]
        
        # Two Pair
        if len(pairs) >= 2:
            pairs.sort(reverse=True)
            kickers = sorted([r for r in ranks if r not in pairs[:2]], reverse=True)
            return HandRank.TWO_PAIR, [pairs[0], pairs[1], kickers[0]]
        
        # One Pair
        if pairs:
            pair_rank = pairs[0]
            kickers = sorted([r for r in ranks if r != pair_rank], reverse=True)
            return HandRank.ONE_PAIR, [pair_rank, kickers[0], kickers[1], kickers[2]]
        
        # High Card
        ranks.sort(reverse=True)
        return HandRank.HIGH_CARD, ranks[:5]
    
    def _compare_hands(self, player1: int, player2: int) -> int:
        """
        Compare poker hands between two players.
        
        Args:
            player1: First player ID
            player2: Second player ID
        
        Returns:
            1 if player1 wins, -1 if player2 wins, 0 if tie
        """
        rank1, kickers1 = self._evaluate_hand(player1)
        rank2, kickers2 = self._evaluate_hand(player2)
        
        # Compare hand ranks
        if rank1.value > rank2.value:
            return 1
        if rank1.value < rank2.value:
            return -1
        
        # Compare kickers for tie-breaking
        for k1, k2 in zip(kickers1, kickers2):
            if k1 > k2:
                return 1
            if k1 < k2:
                return -1
        
        # Complete tie
        return 0
    
    def _calculate_rewards(self) -> float:
        """
        Calculate rewards for the current player.
        
        Returns:
            Reward value (change in chips)
        """
        #TODO: Implement reward calculation based on hand evaluation
        return 0.0
    
    def render(self, mode: bool = True) -> None:
        """
        Render the current state of the environment.
        
        Args:
            mode: Rendering toggle (default: True)
        """
        if not mode:
            return
        
        print("\n" + "=" * 80)
        print(f"Poker Game - {self.num_players} players")
        print(f"Dealer: Player {self.dealer_position}")
        print(f"Pot: {self.pot}")
        print(f"Community cards: {' '.join(str(card) for card in self.community_cards)}")
        print("-" * 80)
        
        for i in range(self.num_players):
            status = ""
            if i in self.folded_players:
                status = "(folded)"
            elif i in self.all_in_players:
                status = "(all-in)"
            elif i == self.current_player:
                status = "(current)"
            
            hand = ' '.join(str(card) for card in self.player_hands[i])
            if i != self.current_player and i not in self.folded_players:
                hand = "** **"  # Hide other players' cards
                
            print(f"Player {i}: Chips {self.player_chips[i]}, Bet {self.player_bets[i]}, Hand {hand} {status}")
        
        print("=" * 80)