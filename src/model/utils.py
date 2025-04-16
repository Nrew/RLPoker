from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate

try:
    from . import config
except ImportError:
    import config # Fallback for running directly


# --- Constants ---
STREET_MAP: Dict[str, int] = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
RANK_MAP: Dict[str, int] = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                           'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
SUIT_MAP: Dict[str, int] = {'C': 1, 'D': 2, 'H': 3, 'S': 4}
INV_SUIT_MAP: Dict[int, str] = {v: k for k, v in SUIT_MAP.items()}
INV_RANK_MAP: Dict[int, str] = {v: k for k, v in RANK_MAP.items()}


# --- Card Encoding ---
def encode_card(card_str: Optional[str]) -> List[float]:
    """
    Encodes a card string (e.g., 'S2', 'HA') into a normalized numerical vector [rank, suit].

    Args:
        card_str: The card string or None.

    Returns:
        A list containing the normalized rank and suit [norm_rank, norm_suit], or [0.0, 0.0] for invalid input.
    """
    if card_str is None or not isinstance(card_str, str) or len(card_str) < 2: return [0.0, 0.0]

    suit: str = card_str[0].upper()
    rank: str = card_str[1:].upper()
    if rank == '10': rank = 'T'  # Standardize rank representation

    norm_rank: float = RANK_MAP.get(rank, 0) / 14.0
    norm_suit: float = SUIT_MAP.get(suit, 0) / 4.0
    return [norm_rank, norm_suit]


def card_strs_to_pokerengine_int(card_strs: Optional[List[str]]) -> List[int]:
    """
    Converts a list of card strings ['S2', 'HA'] to PyPokerEngine's internal integer representation.

    Args:
        card_strs: A list of card strings or None.

    Returns:
        A list of PyPokerEngine card integers, or an empty list on failure or invalid input.
    """
    if not isinstance(card_strs, (list, tuple)) or not all(isinstance(s, str) for s in card_strs): return []
    try:
        # gen_cards correctly handles the conversion: Suit (S,H,D,C) + Rank (2-9, T, J, Q, K, A)
        return gen_cards(card_strs)
    except Exception as e:
        # print(f"Warning: Failed to convert card strings {card_strs} to int: {e}")
        return []


# --- Feature Extraction Helpers ---
def _get_board_texture(community_cards_str: List[str]) -> Dict[str, float]:
    """
    Calculates simple board texture features (flop/turn paired/suited).

    Args:
        community_cards_str: List of community card strings (e.g., ['D5', 'S8', 'HT']).

    Returns:
        A dictionary containing texture features ('flop_paired', 'flop_mono', 'turn_mono').
    """
    features: Dict[str, float] = {'flop_paired': 0.0, 'flop_mono': 0.0, 'turn_mono': 0.0}
    num_cards: int = len(community_cards_str)

    if num_cards < 3:
        return features  # Not enough cards for flop texture

    # Process Flop
    flop_cards: List[str] = community_cards_str[:3]
    flop_ranks: List[int] = [RANK_MAP.get(c[1:].upper() if c[1:].upper() != '10' else 'T', 0) for c in flop_cards if len(c) >= 2]
    flop_suits: List[int] = [SUIT_MAP.get(c[0].upper(), 0) for c in flop_cards if len(c) >= 1]

    if len(flop_ranks) == 3 and len(flop_suits) == 3: # Check successful parsing
        # Flop Paired: Check if any rank appears more than once
        if len(set(flop_ranks)) < 3:
            features['flop_paired'] = 1.0
        # Flop Monotone: Check if all suits are the same
        if len(set(flop_suits)) == 1:
            features['flop_mono'] = 1.0

    # Process Turn (if available)
    if num_cards >= 4:
        turn_card: str = community_cards_str[3]
        if len(turn_card) >= 2:
            turn_suit: int = SUIT_MAP.get(turn_card[0].upper(), 0)
            if len(flop_suits) == 3: # Ensure flop suits were valid
                # Turn Monotone Check
                if features['flop_mono'] == 1.0 and turn_suit == flop_suits[0]:
                    features['turn_mono'] = 1.0
                elif len(set(flop_suits + [turn_suit])) == 1: # Check if first 4 cards make mono
                    features['turn_mono'] = 1.0
                # Could add more features: 4-to-straight, turn pairing the board, etc.

    return features


def _get_blinds_info(round_state: Dict[str, Any], seats: List[Dict[str, Any]], my_uuid: str) -> Tuple[float, float, float]:
    """
    Identifies SB/BB positions and the actual Big Blind amount.

    Args:
        round_state: The round state dictionary.
        seats: The list of seat dictionaries.
        my_uuid: The UUID of the agent.

    Returns:
        A tuple containing: (actual_big_blind_amount, is_agent_sb_flag, is_agent_bb_flag).
    """
    small_blind_amount: int = round_state.get("small_blind_amount", config.SMALL_BLIND)
    # Default BB, potentially updated from history
    big_blind_amount: int = config.BIG_BLIND
    sb_player_pos: int = -1
    bb_player_pos: int = -1

    # Find SB/BB players and amounts from preflop action history (most reliable)
    action_histories: Dict[str, List[Dict[str, Any]]] = round_state.get("action_histories", {})
    preflop_actions: List[Dict[str, Any]] = action_histories.get("preflop", [])
    if preflop_actions:
        for action in preflop_actions:
            action_type: str = action.get("action", "").upper()
            uuid: Optional[str] = action.get("uuid")
            amount: int = action.get("amount", 0)

            if uuid: # Ensure uuid exists
                 player_pos = next((i for i, p in enumerate(seats) if p.get('uuid') == uuid), -1)
                 if player_pos != -1:
                     if "SMALLBLIND" in action_type or (action_type == "POST" and amount == small_blind_amount):
                         sb_player_pos = player_pos
                         small_blind_amount = max(small_blind_amount, amount) # Use actual posted SB
                     elif "BIGBLIND" in action_type or (action_type == "POST" and amount > small_blind_amount):
                         bb_player_pos = player_pos
                         big_blind_amount = max(big_blind_amount, amount) # Use actual posted BB

    # Fallback: Use position relative to dealer if history is missing SB/BB actions
    if sb_player_pos == -1 or bb_player_pos == -1:
        dealer_btn_pos: int = round_state.get('dealer_btn', 0)
        num_players: int = len(seats)
        if num_players > 1:
            sb_player_pos = (dealer_btn_pos + 1) % num_players
            if num_players > 2:
                bb_player_pos = (dealer_btn_pos + 2) % num_players
            else: # Heads up: dealer is SB
                sb_player_pos = dealer_btn_pos
                bb_player_pos = (dealer_btn_pos + 1) % num_players

    # Determine if the agent is SB or BB
    my_pos: int = next((i for i, p in enumerate(seats) if p.get('uuid') == my_uuid), -1)
    is_sb: float = 1.0 if my_pos == sb_player_pos and my_pos != -1 else 0.0
    is_bb: float = 1.0 if my_pos == bb_player_pos and my_pos != -1 else 0.0

    return float(big_blind_amount), is_sb, is_bb


# --- Main State Extraction Function ---
def extract_state(hole_card: Optional[List[str]],
                  round_state: Dict[str, Any],
                  my_uuid: str,
                  initial_stack: int,
                  valid_actions: Optional[List[Dict[str, Any]]] = None
                  ) -> np.ndarray:
    """
    Converts game info into a complex, normalized numerical state vector (NumPy array).
    The state vector aims to capture crucial aspects of the game for decision making.
    Requires config.STATE_DIM to be 30.

    Args:
        hole_card: Player's hole card strings (e.g., ['S2', 'HA']).
        round_state: Current round state dictionary from PyPokerEngine.
        my_uuid: The UUID of the agent player.
        initial_stack: Initial stack size (used minimally).
        valid_actions: Optional list of valid actions (provides precise call/raise amounts).

    Returns:
        A NumPy array representing the state (shape: [config.STATE_DIM]). Returns zeros if state cannot be determined.
    """
    state_list: List[float] = [] # Build state as a list first
    try:
        # --- 0. Initial Setup and Data Extraction ---
        seats: List[Dict[str, Any]] = round_state.get('seats', [])
        player_count: int = len(seats)
        if player_count == 0: return np.zeros(config.STATE_DIM, dtype=np.float32)

        my_seat_info: Optional[Dict[str, Any]] = None
        my_seat_index: int = -1
        opponent_stacks: List[int] = []
        active_player_uuids: List[str] = []
        for idx, player_info in enumerate(seats):
            p_uuid = player_info.get('uuid')
            if p_uuid == my_uuid:
                my_seat_info = player_info
                my_seat_index = idx
            # Track active players and opponent stacks
            if player_info.get('state') in ['participating', 'allin'] and p_uuid:
                 active_player_uuids.append(p_uuid)
                 if p_uuid != my_uuid:
                      opponent_stacks.append(player_info.get('stack', 0))

        # If agent not found (e.g., observed state after busting), return zeros
        if my_seat_info is None or my_seat_index == -1:
            return np.zeros(config.STATE_DIM, dtype=np.float32)

        my_stack: int = my_seat_info.get('stack', 0)
        num_active_players: int = len(active_player_uuids)
        dealer_btn_seat_index: int = round_state.get('dealer_btn', 0)
        community_cards_str: List[str] = round_state.get('community_card', [])
        street_name: str = round_state.get('street', 'preflop')
        street_idx: int = STREET_MAP.get(street_name, 0)
        pot_info: Dict[str, Any] = round_state.get('pot', {})
        pot_size: int = pot_info.get('main', {}).get('amount', 0)
        for side_pot in pot_info.get('side', []): # Include side pots
             pot_size += side_pot.get('amount', 0)

        # Get actual Big Blind amount and agent's SB/BB status
        big_blind_amount, is_sb, is_bb = _get_blinds_info(round_state, seats, my_uuid)
        bb: float = max(big_blind_amount, config.EPSILON) # Use actual BB, ensure non-zero

        # === Feature Set (Total 30 Features) ===

        # 1. Hole Cards Encoded (4 features)
        # Ensure hole_card is valid before processing
        if not isinstance(hole_card, (list, tuple)) or len(hole_card) != 2:
             cards_encoded: List[List[float]] = [[0.0, 0.0], [0.0, 0.0]]
        else:
             cards_encoded = sorted([encode_card(c) for c in hole_card], key=lambda x: x[0], reverse=True)
        state_list.extend(cards_encoded[0])
        state_list.extend(cards_encoded[1])

        # 2. Community Cards Encoded (10 features) - Padded to 5 cards
        comm_cards_encoded: List[List[float]] = [encode_card(c) for c in community_cards_str]
        while len(comm_cards_encoded) < 5:
            comm_cards_encoded.append([0.0, 0.0])
        for card_features in comm_cards_encoded:
            state_list.extend(card_features)

        # 3. Estimated Win Rate (1 feature) - Post-flop Monte Carlo estimate
        win_rate: float = 0.5 # Default preflop / if error
        if street_idx > 0 and hole_card and len(hole_card) == 2:
            try:
                hole_cards_int: List[int] = card_strs_to_pokerengine_int(hole_card)
                community_cards_int: List[int] = card_strs_to_pokerengine_int(community_cards_str)
                # Ensure card conversions were successful
                if hole_cards_int and community_cards_int is not None:
                    num_opponents_for_est: int = max(1, num_active_players - 1) # Estimate vs active opponents
                    win_rate = estimate_hole_card_win_rate(
                        nb_simulation=100, # Lower simulation count for speed
                        nb_player=num_opponents_for_est + 1, # Total players in simulation
                        hole_card=hole_cards_int,
                        community_card=community_cards_int
                    )
            except Exception: # Catch potential errors during estimation
                win_rate = 0.5 # Fallback if estimation fails
        state_list.append(win_rate) # Already normalized 0-1

        # 4. Stack Sizes (Normalized by BB) (2 features)
        state_list.append(my_stack / bb)
        avg_opp_stack: float = np.mean(opponent_stacks) if opponent_stacks else 0
        state_list.append(avg_opp_stack / bb)

        # 5. Pot Size (Normalized by BB) (1 feature)
        state_list.append(pot_size / bb)

        # 6. Stack-to-Pot Ratio (SPR) (1 feature) - Normalized
        # Effective stack is the minimum stack involved in the pot typically
        effective_stack: float = min(my_stack, avg_opp_stack) if avg_opp_stack > 0 else my_stack
        spr: float = effective_stack / (pot_size + config.EPSILON)
        # Normalize SPR: clipping and scaling is simple; log scale could also work
        normalized_spr: float = np.clip(spr / 15.0, 0.0, 1.0) # Clip SPR > 15 BBs, scale 0-1
        state_list.append(normalized_spr)

        # 7. Position (Relative to Button, Normalized) (1 feature)
        norm_position: float = 0.0
        if player_count > 1:
            # Position relative to button (0=Button, 1=SB, ... clockwise)
            relative_position: int = (my_seat_index - dealer_btn_seat_index + player_count) % player_count
            # Normalize position 0 to 1 relative to number of opponents
            norm_position = relative_position / float(player_count -1 + config.EPSILON)
        state_list.append(norm_position)

        # 8. Is Small Blind / Is Big Blind Flags (2 features)
        state_list.append(is_sb)
        state_list.append(is_bb)

        # 9. Street Index (Normalized) (1 feature)
        state_list.append(street_idx / 3.0) # 0, 0.33, 0.66, 1.0

        # 10. Active Players Ratio (Normalized) (1 feature)
        active_ratio: float = num_active_players / float(player_count) if player_count > 0 else 0.0
        state_list.append(active_ratio)

        # --- Betting Information Features ---
        call_amount: int = 0
        # Get precise info from valid_actions if available (passed during declare_action)
        if valid_actions:
            call_action: Optional[Dict[str, Any]] = next((a for a in valid_actions if a['action'] == 'call'), None)
            if call_action:
                call_amount = call_action.get('amount', 0)
        else: # Estimate from history if valid_actions not provided (less accurate)
             action_histories: Dict[str, List[Dict[str, Any]]] = round_state.get('action_histories', {})
             current_street_history: List[Dict[str, Any]] = action_histories.get(street_name, [])
             my_paid_this_street: int = 0
             max_bet_this_street: int = 0
             if current_street_history:
                 # Simplified history parsing for estimation
                 current_bets: Dict[str, int] = {p['uuid']: 0 for p in seats if 'uuid' in p}
                 for action in current_street_history:
                     uuid: Optional[str] = action.get('uuid')
                     action_type: str = action.get('action','').lower()
                     amount: int = action.get('amount', 0)
                     add_amount: int = action.get('add_amount', 0) # Used in older versions? Check format. Amount usually includes total bet.

                     if uuid and uuid in current_bets:
                          if action_type in ['bet', 'raise']:
                              current_bets[uuid] = amount # Action amount is the total bet
                          elif action_type == 'call':
                              # Amount called is the total bet amount
                              current_bets[uuid] = amount
                          elif action_type in ['smallblind', 'bigblind', 'post']:
                              current_bets[uuid] = amount # Blind is the total bet

                 my_paid_this_street = current_bets.get(my_uuid, 0)
                 max_bet_this_street = max(current_bets.values()) if current_bets else 0

             estimated_call_amount: int = max(0, max_bet_this_street - my_paid_this_street)
             call_amount = min(estimated_call_amount, my_stack) # Cannot call more than stack

        # 11. Amount To Call (Normalized by Pot Size) (1 feature)
        norm_call_amount_pot: float = call_amount / (pot_size + config.EPSILON)
        # Clip to prevent extreme values if call >> pot
        state_list.append(np.clip(norm_call_amount_pot, 0.0, 2.0)) # E.g., Max call considered is 2x pot

        # 12. Pot Odds (If Call Required) (1 feature)
        pot_odds: float = 0.0
        if call_amount > 0:
            # Pot odds: amount to call / (current pot + amount to call)
            pot_odds = call_amount / (pot_size + call_amount + config.EPSILON)
        state_list.append(pot_odds) # Naturally normalized 0-1

        # 13. Number of Bets/Raises This Street (Normalized) (1 feature)
        num_bets_raises: int = 0
        current_street_history_count: List[Dict[str, Any]] = round_state.get('action_histories', {}).get(street_name, [])
        for action in current_street_history_count:
             action_type: str = action.get("action", "").lower()
             if action_type == "bet" or action_type == "raise":
                 num_bets_raises += 1
        # Normalize by capping at a reasonable number (e.g., 5)
        normalized_aggression: float = min(num_bets_raises / 5.0, 1.0)
        state_list.append(normalized_aggression)

        # 14. Board Texture Features (3 features)
        texture: Dict[str, float] = _get_board_texture(community_cards_str)
        state_list.append(texture['flop_paired'])
        state_list.append(texture['flop_mono'])
        state_list.append(texture['turn_mono'])


        # --- Final Validation and Conversion ---
        current_len: int = len(state_list)
        expected_len: int = config.STATE_DIM

        if current_len != expected_len:
            # This should not happen if logic above is correct and features counted right
            print(f"CRITICAL Error: State feature count mismatch! Expected {expected_len}, Got {current_len}. Padding/Truncating.")
            if current_len < expected_len:
                state_list.extend([0.0] * (expected_len - current_len))
            else:
                state_list = state_list[:expected_len]

        state_np: np.ndarray = np.array(state_list, dtype=np.float32)

        # Final check for NaNs/Infs that might have crept in
        if np.isnan(state_np).any() or np.isinf(state_np).any():
            print(f"Warning: NaN/Inf detected in final state vector! Replacing with zeros.")
            state_np = np.nan_to_num(state_np, nan=0.0, posinf=0.0, neginf=0.0) # Safe fallback

        # Ensure final shape is exactly correct after all manipulations
        if state_np.shape != (expected_len,):
             raise ValueError(f"Final state dimension mismatch after processing! Expected {(expected_len,)}, Got {state_np.shape}")

        return state_np

    except Exception as e:
        print(f"FATAL Error during state extraction process: {e}")
        import traceback
        traceback.print_exc()
        # Return a zero vector consistent with expected dimensions
        return np.zeros(config.STATE_DIM, dtype=np.float32)


# --- Action Mapping Function ---
def map_action_to_poker(action_idx: int,
                        valid_actions: List[Dict[str, Any]],
                        current_stack: int,
                        round_state: Dict[str, Any]
                        ) -> Tuple[str, int]:
    """
    Maps a discrete action index (policy output) to a valid PyPokerEngine action tuple.
    Corrected to prevent returning a dictionary for the amount.

    Args:
        action_idx: Policy output index (0:Fold, 1:Call, 2:Raise Min, 3:Raise Pot, 4:All-in).
        valid_actions: List of valid action dicts from PyPokerEngine.
        current_stack: Agent's current stack size.
        round_state: Current round state dictionary (needed for pot size calc).

    Returns:
        A tuple containing the validated action name (str) and integer amount (int).
    """
    if not valid_actions:
        print("Warning: map_action_to_poker received empty valid_actions. Defaulting to fold.")
        return 'fold', 0

    # Check availability of basic actions
    can_fold: bool = any(a['action'] == 'fold' for a in valid_actions)
    can_call: bool = any(a['action'] == 'call' for a in valid_actions)
    can_raise: bool = any(a['action'] == 'raise' for a in valid_actions)

    # Extract detailed info for each possible action type
    fold_action_info: Dict[str, Any] = next((a for a in valid_actions if a['action'] == 'fold'), {'action': 'fold', 'amount': 0})
    call_action_info: Optional[Dict[str, Any]] = next((a for a in valid_actions if a['action'] == 'call'), None)
    raise_action_info: Optional[Dict[str, Any]] = next((a for a in valid_actions if a['action'] == 'raise'), None)

    # Initialize proposed action name and amount (start with safest)
    proposed_action_name: str = fold_action_info['action']
    proposed_amount_int: int = fold_action_info['amount'] # Ensure this is always int

    call_amount_req: int = call_action_info['amount'] if call_action_info else 0

    # --- Action Interpretation based on policy output (action_idx) ---
    # Determine the DESIRED action and amount based on policy index
    if action_idx == 0: # Fold
        proposed_action_name = fold_action_info['action']
        proposed_amount_int = fold_action_info['amount']

    elif action_idx == 1: # Call
        if can_call and call_action_info:
            proposed_action_name = 'call'
            proposed_amount_int = call_action_info['amount']
        # If cannot call, proposed remains fold

    elif action_idx == 2: # Raise Min
        if can_raise and raise_action_info:
            proposed_action_name = 'raise'
            # --- CRITICAL: Extract INT here ---
            proposed_amount_int = raise_action_info['amount']['min']
        elif can_call and call_action_info: # Fallback to call
            proposed_action_name = 'call'
            proposed_amount_int = call_action_info['amount']
        # If cannot raise or call, proposed remains fold

    elif action_idx == 3: # Raise Pot
        if can_raise and raise_action_info:
            proposed_action_name = 'raise'
            pot_info: Dict[str, Any] = round_state.get('pot', {})
            pot_size: int = pot_info.get('main', {}).get('amount', 0)
            for side_pot in pot_info.get('side', []): pot_size += side_pot.get('amount', 0)

            target_total_bet: int = pot_size + (2 * call_amount_req)
            # --- CRITICAL: Extract INTs here ---
            min_allowed: int = raise_action_info['amount']['min']
            max_allowed: int = raise_action_info['amount']['max']
            # Calculate and clamp using INTs only
            calculated_amount: int = max(min_allowed, target_total_bet)
            proposed_amount_int = min(max_allowed, calculated_amount)
        elif can_call and call_action_info: # Fallback to call
            proposed_action_name = 'call'
            proposed_amount_int = call_action_info['amount']
        # If cannot raise or call, proposed remains fold

    elif action_idx == 4: # All-in (Max Raise or Call if needed)
        if can_raise and raise_action_info: # Prefer raising all-in if possible
            proposed_action_name = 'raise'
            # --- CRITICAL: Extract INT here ---
            proposed_amount_int = raise_action_info['amount']['max']
        elif can_call and call_action_info and call_amount_req >= current_stack: # All-in call scenario
            proposed_action_name = 'call'
            proposed_amount_int = call_action_info['amount'] # Amount should be stack
        elif can_call and call_action_info: # If cannot raise max, but can call (not all-in)
            proposed_action_name = 'call'
            proposed_amount_int = call_action_info['amount']
        # If cannot raise or call, proposed remains fold

    # --- Final Validation against valid_actions list ---
    # Now, check if the (proposed_action_name, proposed_amount_int) pair is actually valid
    chosen_action_is_valid: bool = False
    final_action_name: str = proposed_action_name
    final_amount_int: int = int(max(0, proposed_amount_int)) # Ensure non-negative int

    for valid_act in valid_actions:
        act_name: str = valid_act['action']
        if act_name == final_action_name:
            if act_name == 'raise':
                # Check if the INT amount is within the valid range dict
                min_val = valid_act.get('amount', {}).get('min', -1)
                max_val = valid_act.get('amount', {}).get('max', -1)
                if min_val <= final_amount_int <= max_val:
                    chosen_action_is_valid = True
                    break
            elif act_name == 'call':
                # Check if the INT amount matches exactly
                if valid_act.get('amount', -1) == final_amount_int:
                    chosen_action_is_valid = True
                    break
            elif act_name == 'fold': # Fold amount should be 0
                if final_amount_int == 0:
                    chosen_action_is_valid = True
                    break

    # If the interpretation resulted in an invalid action/amount combo, default safely
    if not chosen_action_is_valid:
        # print(f"Warning: Proposed action/amount ({final_action_name}/{final_amount_int}) invalid based on valid_actions. Defaulting.")
        if can_call and call_action_info: # Default to call if possible
            final_action_name = 'call'
            final_amount_int = call_action_info['amount']
        elif can_fold: # Otherwise, default to fold
            final_action_name = 'fold'
            final_amount_int = 0
        elif can_raise and raise_action_info: # If only raise possible, default to min raise
             # print("Warning: Defaulting to minimum raise as fold/call invalid.")
             final_action_name = 'raise'
             final_amount_int = raise_action_info['amount']['min'] # Extract INT
        else: # Should be impossible if valid_actions is correct & non-empty
             print("CRITICAL WARNING: No valid action could be determined! Defaulting to fold 0.")
             final_action_name = 'fold'
             final_amount_int = 0

    # Ensure the final returned amount is definitely an integer
    final_return_amount = int(final_amount_int)

    # Optional Debug print before returning
    # print(f"DEBUG: map_action_to_poker returning: action='{final_action_name}', amount={final_return_amount} (type: {type(final_return_amount)})")

    return final_action_name, final_return_amount