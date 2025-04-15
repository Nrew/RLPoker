def process_poker_state_for_nn(hole_card, round_state, uuid):
    """
    Process the poker round state into a format suitable for neural network input

    Args:
        hole_card (list): List of hole cards, e.g. ['CA', 'DK']
        round_state (dict): The round state dictionary from PyPokerEngine
        uuid (str): Your player's UUID

    Returns:
        dict: Organized state information for neural network processing
    """
    # Initialize the structured state
    state = {}

    # 1. Process hole cards - convert to rank and suit encoding
    state['hole_cards'] = []
    for card in hole_card:
        suit = card[0]  # C, D, H, S
        rank = card[1]  # A, 2-9, T, J, Q, K
        state['hole_cards'].append({'rank': rank, 'suit': suit})

    # 2. Process community cards
    state['community_cards'] = []
    for card in round_state['community_card']:
        suit = card[0]
        rank = card[1]
        state['community_cards'].append({'rank': rank, 'suit': suit})

    # 3. Game stage/street
    state['street'] = round_state['street']

    # 4. Position information
    dealer_pos = round_state['dealer_btn']
    player_count = len(round_state['seats'])
    player_idx = None

    # Find our position
    for i, seat in enumerate(round_state['seats']):
        if seat['uuid'] == uuid:
            player_idx = i
            break

    if player_idx is not None:
        # Calculate relative position (0 = dealer, 1 = small blind)
        state['position'] = (player_idx - dealer_pos) % player_count
        state['is_dealer'] = player_idx == dealer_pos
        state['is_small_blind'] = player_idx == round_state['small_blind_pos']
        state['is_big_blind'] = player_idx == round_state['big_blind_pos']
    else:
        state['position'] = -1  # Error case

    # 5. Pot odds and stack information
    total_pot = round_state['pot']['main']['amount']
    for side_pot in round_state['pot']['side']:
        total_pot += side_pot['amount']

    state['pot_size'] = total_pot

    # Get player stacks and normalize by total chips in play
    state['player_stacks'] = []
    total_chips = 0
    my_stack = 0

    for seat in round_state['seats']:
        stack = seat['stack']
        total_chips += stack
        if seat['uuid'] == uuid:
            my_stack = stack
            state['my_stack'] = stack

        state['player_stacks'].append({
            'uuid': seat['uuid'],
            'stack': stack,
            'state': seat['state']  # participating, folded, etc.
        })

    # Normalize stack sizes and pot
    total_chips += total_pot  # Include pot in total chips
    state['my_stack_ratio'] = my_stack / total_chips if total_chips > 0 else 0
    state['pot_ratio'] = total_pot / total_chips if total_chips > 0 else 0

    # 6. Action history
    state['action_history'] = {}
    for street, actions in round_state['action_histories'].items():
        state['action_history'][street] = []
        for action in actions:
            action_type = action['action']
            amount = action.get('amount', 0)

            # Normalize bet amount by pot size
            amount_ratio = amount / total_pot if total_pot > 0 else 0

            state['action_history'][street].append({
                'player': action['uuid'],
                'action': action_type,
                'amount': amount,
                'amount_ratio': amount_ratio
            })

    # 7. Count active players
    active_players = [seat for seat in round_state['seats'] if seat['state'] == 'participating']
    state['active_player_count'] = len(active_players)

    # 8. Last aggressive action
    state['last_aggressor'] = None
    state['last_raise_ratio'] = 0

    # Examine the current street for raises
    current_street = round_state['street']
    if current_street in round_state['action_histories']:
        for action in reversed(round_state['action_histories'][current_street]):
            if action['action'] in ['RAISE', 'BIGBLIND']:
                state['last_aggressor'] = action['uuid']
                state['last_raise_ratio'] = action['amount'] / total_pot if total_pot > 0 else 0
                break

    # 9. Calculate immediate pot odds (if facing a bet)
    state['pot_odds'] = 0
    state['facing_amount'] = 0
    if current_street in round_state['action_histories']:
        call_amount = 0
        for action in round_state['action_histories'][current_street]:
            if action['uuid'] != uuid and action['action'] in ['BIGBLIND', 'CALL', 'RAISE']:
                call_amount = max(call_amount, action['amount'])

        if call_amount > 0:
            state['facing_amount'] = call_amount
            state['pot_odds'] = call_amount / (total_pot + call_amount)

    return state