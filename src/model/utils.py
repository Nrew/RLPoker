import numpy as np
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate

try:
    from . import config
except ImportError:
    import config # Fallback for running directly


# --- Constants ---
STREET_MAP = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
RANK_MAP = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
            'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
SUIT_MAP = {'C': 1, 'D': 2, 'H': 3, 'S': 4}
INV_SUIT_MAP = {v: k for k, v in SUIT_MAP.items()}
INV_RANK_MAP = {v: k for k, v in RANK_MAP.items()}

# --- Card Encoding ---
def encode_card(card_str):
    """Encodes a card string (e.g., 'S2', 'HA') into a normalized numerical vector [rank, suit]."""
    if card_str is None or len(card_str) < 2:
        return [0.0, 0.0]
    suit = card_str[0].upper()
    rank = card_str[1:].upper()
    # Handle '10' card represented as 'T' by pypokerengine internals but sometimes '10' in strings
    if rank == '10': rank = 'T'

    norm_rank = RANK_MAP.get(rank, 0) / 14.0
    norm_suit = SUIT_MAP.get(suit, 0) / 4.0
    return [norm_rank, norm_suit]

def card_strs_to_pokerengine_int(card_strs):
    """Converts a list of card strings ['S2', 'HA'] to PyPokerEngine's internal integer representation."""
    if not isinstance(card_strs, (list, tuple)) or not all (isinstance(s, str) for s in card_strs):
        # print(f'Warning: Invalid input format for card_strs_to_pokerengine_int: {card_strs}')
        return []
    try:
        return gen_cards(card_strs) # gen_cards handles the conversion
    except Exception as e:
        # print(f"Warning: Failed to convert card strings {card_strs} to int: {e}")
        return [] # Return empty list on failure


# --- Feature Extraction Helpers ---
def _get_board_texture(community_cards):
    """Calculates simple board texture features (flop/turn paired/suited)."""
    features = {'flop_paired': 0.0, 'flop_mono': 0.0, 'turn_mono': 0.0}
    if not community_cards or len(community_cards) < 3:
        return features # Not enough cards for texture

    flop_cards = community_cards[:3]
    flop_ranks = [RANK_MAP.get(c[1:], 0) for c in flop_cards]
    flop_suits = [SUIT_MAP.get(c[0], 0) for c in flop_cards]

    if len(flop_ranks) != 3 or len (flop_suits) != 3:
        # print(f"Warning: Invalid flop cards for texture extraction: {flop_cards}")
        return features

    # Flop Paired: Check if any rank appears more than once
    if len(set(flop_ranks)) < 3:
        features['flop_paired'] = 1.0

    # Flop Monotone: Check if all suits are the same
    if len(set(flop_suits)) == 1:
        features['flop_mono'] = 1.0

    if len(community_cards) >= 4:
        turn_card = community_cards[3]
        if len(turn_card) < 2:
            # print(f"Warning: Invalid turn card for texture extraction: {turn_card}")
            return features
        turn_suit = SUIT_MAP.get(turn_card[0], 0)
        # Turn Monotone Check (if flop was already monotone, or 3 suits match now)
        if features['flop_mono'] == 1.0 and turn_suit == flop_suits[0]:
             features['turn_mono'] = 1.0
        elif len(set(flop_suits + [turn_suit])) == 1: # Check if first 4 cards make mono
            features['turn_mono'] = 1.0
        # Could add more features: 3-to-straight, 4-to-straight, etc.

    return features

def _get_blinds_info(round_state, seats, my_uuid):
    """Identifies SB/BB positions and amounts (more robustly)."""
    small_blind_amount = round_state.get("small_blind_amount", config.SMALL_BLIND)
    big_blind_amount = config.BIG_BLIND
    sb_player_pos = -1
    bb_player_pos = -1

    # Find SB/BB based on action history (more reliable than just position)
    action_histories = round_state.get("action_histories", {})
    preflop_actions = action_histories.get("preflop", [])
    if preflop_actions:
        for action in preflop_actions:
             # Check if action is SMALLBLIND or BIGBLIND (adjust key if needed)
             action_type = action.get("action", "").upper()
             uuid = action.get("uuid")
             amount = action.get("amount", 0)

             if "SMALLBLIND" in action_type or (action_type == "POST" and amount == small_blind_amount):
                 sb_player_pos = next((i for i, p in enumerate(seats) if p['uuid'] == uuid), -1)
                 small_blind_amount = max(small_blind_amount, amount) # Update SB if posted amount > config
             elif "BIGBLIND" in action_type or (action_type == "POST" and amount > small_blind_amount):
                 bb_player_pos = next((i for i, p in enumerate(seats) if p['uuid'] == uuid), -1)
                 big_blind_amount = max(big_blind_amount, amount)

    # Fallback: Use position relative to dealer if history is missing
    if sb_player_pos == -1 or bb_player_pos == -1:
         dealer_btn_pos = round_state.get('dealer_btn', 0)
         num_players = len(seats)
         if num_players > 1:
            sb_player_pos = (dealer_btn_pos + 1) % num_players
            if num_players > 2:
                 bb_player_pos = (dealer_btn_pos + 2) % num_players
            else: # Heads up: dealer is SB
                 sb_player_pos = dealer_btn_pos
                 bb_player_pos = (dealer_btn_pos + 1) % num_players

    # Check if agent is SB or BB
    my_pos = next((i for i, p in enumerate(seats) if p['uuid'] == my_uuid), -1)
    is_sb = 1.0 if my_pos == sb_player_pos else 0.0
    is_bb = 1.0 if my_pos == bb_player_pos else 0.0

    return big_blind_amount, is_sb, is_bb

# --- State Extraction ---

def extract_state(hole_card, round_state, my_uuid, initial_stack, valid_actions=None):
    """
    Converts game info into a complex, normalized numerical state vector (NumPy array).

    Args:
        hole_card (list): Player's hole card strings (e.g., ['S2', 'HA']).
        round_state (dict): Current round state from PyPokerEngine.
        my_uuid (str): The UUID of the agent player.
        initial_stack (int): Initial stack size (less critical with BB normalization).
        valid_actions (list, optional): List of valid actions (provides precise call/raise amounts).

    Returns:
        np.ndarray: A NumPy array representing the state (shape: [config.STATE_DIM]).
    """
    state = []
    try:
        seats = round_state.get('seats', [])
        player_count = len(seats)
        if player_count == 0: return np.zeros(config.STATE_DIM, dtype=np.float32) # Cannot extract state

        # --- Basic Info & Agent Identification ---
        my_seat_info = None
        my_seat_index = -1
        opponent_stacks = []
        active_player_uuids = []
        for idx, player_info in enumerate(seats):
            if player_info['uuid'] == my_uuid:
                my_seat_info = player_info
                my_seat_index = idx
            # Track active players and opponent stacks regardless of agent finding
            # Check 'state' field, which can be 'participating', 'allin', 'folded'
            if player_info.get('state') in ['participating', 'allin']:
                 active_player_uuids.append(player_info['uuid'])
                 if player_info['uuid'] != my_uuid:
                      opponent_stacks.append(player_info['stack'])

        if my_seat_info is None or my_seat_index == -1:
            print(f"Error: Agent UUID {my_uuid} not found in round_state seats. Returning zeros.")
            return np.zeros(config.STATE_DIM, dtype=np.float32)

        my_stack = my_seat_info.get(['stack'], 0 )
        num_active_players = len(active_player_uuids)
        dealer_btn_seat_index = round_state.get('dealer_btn', 0)
        community_cards_str = round_state.get('community_card', [])
        street_name = round_state.get('street', 'preflop')
        street_idx = STREET_MAP.get(street_name, 0)
        pot_size = round_state.get('pot', {}).get('main', {}).get('amount', 0)
        # Add side pots to total pot? Maybe less relevant for value func. Stick to main pot.

        # --- Blinds Info ---
        # Get actual BB amount for normalization (handles edge cases like HU SB being dealer)
        big_blind_amount, is_sb, is_bb = _get_blinds_info(round_state, seats, my_uuid)
        bb = max(big_blind_amount, config.EPSILON) # Ensure BB is at least epsilon

        # === Feature Set ===

        # 1. Hole Cards Encoded (4 features)
        if not isinstance(hole_card, (list, tuple)) or len(hole_card) != 2:
             cards_encoded = [[0.0, 0.0], [0.0, 0.0]]
        else:
             cards_encoded = sorted([encode_card(c) for c in hole_card], key=lambda x: x[0], reverse=True)
        state.extend(cards_encoded[0])
        state.extend(cards_encoded[1])

        # 2. Community Cards Encoded (10 features)
        comm_cards_encoded = [encode_card(c) for c in community_cards_str]
        while len(comm_cards_encoded) < 5:
            comm_cards_encoded.append([0.0, 0.0])
        for card_features in comm_cards_encoded:
            state.extend(card_features)

        # 3. Estimated Win Rate (1 feature) - Use PyPokerEngine's estimator
        win_rate = 0.5 # Default if preflop or error
        if street_idx > 0 and hole_card and len(hole_card) == 2: # Only estimate post-flop
            try:
                hole_cards_int = card_strs_to_pokerengine_int(hole_card)
                community_cards_int = card_strs_to_pokerengine_int(community_cards_str)
                # Estimate against avg number of likely opponents remaining
                num_opponents_for_est = max(1, num_active_players - 1) # Estimate vs at least 1 active opponent
                win_rate = estimate_hole_card_win_rate(
                    nb_simulation=100, # Low simulation count for speed
                    nb_player=num_opponents_for_est + 1, # Total players for simulation
                    hole_card=hole_cards_int,
                    community_card=community_cards_int
                )
            except Exception as e:
                # print(f"Warning: Win rate estimation failed: {e}")
                win_rate = 0.5 # Fallback
        state.append(win_rate)

        # 4. Stack Sizes (Normalized by BB) (2 features)
        state.append(my_stack / bb)
        avg_opp_stack = np.mean(opponent_stacks) if opponent_stacks else 0
        state.append(avg_opp_stack / bb)

        # 5. Pot Size (Normalized by BB) (1 feature)
        state.append(pot_size / bb)

        # 6. Stack-to-Pot Ratio (SPR) (1 feature) - Effective stack vs pot
        # Use smaller of my stack or average opponent stack as effective stack
        effective_stack = min(my_stack, avg_opp_stack) if avg_opp_stack > 0 else my_stack
        spr = effective_stack / (pot_size + config.EPSILON)
        state.append(np.clip(spr / 10.0, 0, 1)) # Normalize SPR (e.g., clip > 10 and scale 0-1)

        # 7. Position (Relative to Button) (1 feature)
        if player_count > 1:
            relative_position = (my_seat_index - dealer_btn_seat_index + player_count) % player_count
            norm_position = relative_position / float(player_count -1 + config.EPSILON) # Normalize 0 to 1
        else: norm_position = 0.0
        state.append(norm_position)

        # 8. Is Small Blind / Is Big Blind (2 features)
        state.append(is_sb)
        state.append(is_bb)

        # 9. Street (Normalized) (1 feature)
        state.append(street_idx / 3.0)

        # 10. Active Players Ratio (1 feature)
        state.append(num_active_players / float(player_count) if player_count > 0 else 0.0)

        # --- Betting Information ---
        call_amount = 0
        min_raise = 0
        max_raise = 0

        if valid_actions: # Use precise info if available
            call_action = next((a for a in valid_actions if a['action'] == 'call'), None)
            raise_action = next((a for a in valid_actions if a['action'] == 'raise'), None)
            if call_action:
                call_amount = call_action['amount']
                can_call = True
            if raise_action:
                min_raise = raise_action['amount']['min']
                max_raise = raise_action['amount']['max']
                can_raise = True
        else: # Estimate from history (less reliable)
             action_histories = round_state.get('action_histories', {}).get(street_name, [])
             my_paid_this_street = 0
             max_bet_this_street = 0
             if action_histories:
                 round_bets = {}
                 for action in action_histories:
                     uuid = action.get('uuid')
                     paid = action.get('amount', 0) + action.get('add_amount', 0) # Amount *added* by raise
                     # Accumulate total bet per player this street
                     round_bets[uuid] = round_bets.get(uuid, 0) + paid
                 my_paid_this_street = round_bets.get(my_uuid, 0)
                 max_bet_this_street = max(round_bets.values()) if round_bets else 0

             estimated_call_amount = max(0, max_bet_this_street - my_paid_this_street)
             call_amount = min(estimated_call_amount, my_stack) # Cannot call more than stack
             # Cannot reliably estimate min/max raise without valid_actions

        # 11. Amount To Call (Normalized by Pot) (1 feature)
        norm_call_amount_pot = call_amount / (pot_size + config.EPSILON)
        state.append(np.clip(norm_call_amount_pot, 0, 1.5)) # Clip relative call amount

        # 12. Pot Odds (1 feature)
        pot_odds = 0.0
        if call_amount > 0:
            pot_odds = call_amount / (pot_size + call_amount + config.EPSILON)
        state.append(pot_odds) # Naturally normalized between 0 and 1

        # 13. Number of Bets/Raises This Street (1 feature)
        num_bets_raises = 0
        current_street_history = round_state.get('action_histories', {}).get(street_name, [])
        for action in current_street_history:
             # Count 'bet' and 'raise' actions (adjust if engine uses different keys)
             action_type = action.get("action", "").lower()
             if action_type == "bet" or action_type == "raise":
                 num_bets_raises += 1
        # Normalize (e.g., divide by a reasonable max like 5)
        state.append(min(num_bets_raises / 5.0, 1.0))

        # 14. Board Texture (3 features)
        texture = _get_board_texture(community_cards_str)
        state.append(texture['flop_paired'])
        state.append(texture['flop_mono'])
        state.append(texture['turn_mono']) # Could add more texture features here


        # --- Final Padding / Truncation ---
        current_len = len(state)
        expected_len = config.STATE_DIM

        if current_len < expected_len:
            padding = [0.0] * (expected_len - current_len)
            state.extend(padding)
        elif current_len > expected_len:
            print(f"Warning: State length ({current_len}) exceeded STATE_DIM ({expected_len}). Truncating.")
            state = state[:expected_len]

        state_np = np.array(state, dtype=np.float32)

        # Final check for NaNs/Infs introduced by calculations
        if np.isnan(state_np).any() or np.isinf(state_np).any():
            print(f"Warning: NaN/Inf detected in final state vector! State: {state_np}")
            state_np = np.nan_to_num(state_np, nan=0.0, posinf=1.0, neginf=-1.0) # Replace safely

        # Ensure final shape is correct
        if state_np.shape[0] != expected_len:
             raise ValueError(f"Final state dimension mismatch! Expected {expected_len}, Got {state_np.shape[0]}")

        return state_np

    except Exception as e:
        print(f"FATAL Error during state extraction: {e}")
        import traceback
        traceback.print_exc()
        # Return a zero vector consistent with expected dimensions
        return np.zeros(config.STATE_DIM, dtype=np.float32)


# --- Action Mapping ---
def map_action_to_poker(action_idx, valid_actions, current_stack, round_state):
    """
    Maps a discrete action index (0-4) to a valid PyPokerEngine action tuple (action_name, amount).
    Includes added robustness checks.

    Args:
        action_idx (int): Policy output (0:Fold, 1:Call, 2:Raise Min, 3:Raise Pot, 4:All-in).
        valid_actions (list): List of valid action dicts from PyPokerEngine.
        current_stack (int): Agent's current stack size.
        round_state (dict): Current round state (needed for pot size calc).

    Returns:
        tuple: (action_name (str), amount (int))
    """
    can_fold = any(a['action'] == 'fold' for a in valid_actions)
    can_call = any(a['action'] == 'call' for a in valid_actions)
    can_raise = any(a['action'] == 'raise' for a in valid_actions)

    # Extract precise action info
    fold_action_info = next((a for a in valid_actions if a['action'] == 'fold'), {'action': 'fold', 'amount': 0})
    call_action_info = next((a for a in valid_actions if a['action'] == 'call'), None)
    raise_action_info = next((a for a in valid_actions if a['action'] == 'raise'), None)

    action_name = 'fold' # Safest default
    amount = 0
    call_amount_req = call_action_info['amount'] if call_action_info else 0

    # --- Action Interpretation ---
    if action_idx == 0: # Fold
        action_name = 'fold'
        amount = 0
    elif action_idx == 1: # Call
        if can_call:
            action_name = 'call'
            amount = call_action_info['amount']
        else: # Cannot call (e.g., must raise or is all-in already), fold if possible
            action_name = fold_action_info['action'] # 'fold'
            amount = fold_action_info['amount']     # 0
    elif action_idx == 2: # Raise Min
        if can_raise:
            action_name = 'raise'
            amount = raise_action_info['amount']['min']
        elif can_call: # Fallback to call
            action_name = 'call'
            amount = call_action_info['amount']
        else: # Fallback to fold
            action_name = fold_action_info['action']
            amount = fold_action_info['amount']
    elif action_idx == 3: # Raise Pot
        if can_raise:
            action_name = 'raise'
            pot_size = round_state.get('pot', {}).get('main', {}).get('amount', 0)
            # Pot size bet calculation: total bet = current pot + 2 * amount needed to call
            target_total_bet = pot_size + 2 * call_amount_req
            # Clamp raise to valid range [min_raise, max_raise]
            amount = max(raise_action_info['amount']['min'], target_total_bet)
            amount = min(raise_action_info['amount']['max'], amount)
        elif can_call: # Fallback to call
            action_name = 'call'
            amount = call_action_info['amount']
        else: # Fallback to fold
            action_name = fold_action_info['action']
            amount = fold_action_info['amount']
    elif action_idx == 4: # All-in (Max Raise or Call if needed)
        if can_raise: # Prefer raising all-in if possible
            action_name = 'raise'
            amount = raise_action_info['amount']['max']
        elif can_call and call_amount_req >= current_stack: # All-in call scenario
            action_name = 'call'
            amount = call_action_info['amount'] # This should equal current_stack
        elif can_call: # If cannot raise max, but can call (and it's not all-in)
            action_name = 'call'
            amount = call_action_info['amount']
        else: # Cannot raise or call, must fold
            action_name = fold_action_info['action']
            amount = fold_action_info['amount']

    # --- Final Validation and Sanitization ---
    chosen_action_is_valid = False
    validated_action = action_name
    validated_amount = int(max(0, amount)) # Ensure non-negative integer

    for valid_act in valid_actions:
        if valid_act['action'] == validated_action:
            if validated_action == 'raise':
                # Check if the calculated raise amount is within the valid range
                if valid_act['amount']['min'] <= validated_amount <= valid_act['amount']['max']:
                    chosen_action_is_valid = True
                    break
            elif validated_action == 'call':
                # Check if the call amount matches exactly
                if valid_act['amount'] == validated_amount:
                    chosen_action_is_valid = True
                    break
            elif validated_action == 'fold':
                chosen_action_is_valid = True
                break

    # If the logic resulted in an invalid action/amount combo, default safely:
    if not chosen_action_is_valid:
        # print(f"Warning: PPO action index {action_idx} mapped to invalid action/amount: {validated_action}/{validated_amount}. Defaulting.")
        if can_call:
            # Default to call if possible and seems reasonable
            validated_action = 'call'
            validated_amount = call_action_info['amount']
        elif can_fold:
            # Otherwise, fold
            validated_action = 'fold'
            validated_amount = 0
        else:
            # Should not happen if valid_actions is correct, but handle defensively
            # This implies only 'raise' is valid, likely an all-in raise scenario missed?
            # Attempting the minimum possible raise might be better than erroring.
             if can_raise:
                  validated_action = 'raise'
                  validated_amount = raise_action_info['amount']['min']
             else: # Truly stuck - this signals an engine/logic inconsistency
                  print("CRITICAL WARNING: No valid action could be determined! Returning fold 0.")
                  validated_action = 'fold'
                  validated_amount = 0


    return validated_action, int(validated_amount) # Return validated action and amount