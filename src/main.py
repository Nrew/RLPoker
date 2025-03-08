from enviroment import Action, PokerEnv


def interactive_test():
    """Interactive test function for manual testing."""
    env = PokerEnv(num_players=4)
    obs = env.reset()
    
    print("Starting interactive test. Enter action as: ACTION AMOUNT")
    print("Valid actions: FOLD, CHECK, CALL, RAISE, ALL_IN")
    print("Enter 'q' to quit")
    
    done = False
    while not done:
        env.render()
        
        # Display valid actions
        valid_actions = env._get_valid_actions()
        print("Valid actions:")
        for action, amount in valid_actions:
            print(f"  {action.name} {amount}")
        
        # Get user input
        user_input = input("Enter action: ").strip()
        if user_input.lower() == 'q':
            break
        
        try:
            action_str, amount_str = user_input.split()
            action = Action[action_str.upper()]
            amount = int(amount_str)
            
            obs, reward, done, info = env.step((action, amount))
            
            if info.get('hand_over', False):
                print("Hand over! Final state:")
                env.render()
                print(f"Reward: {reward}")
                
                # Continue with a new hand
                if input("Play another hand? (y/n): ").lower() == 'y':
                    obs = env.reset()
                    done = False
                else:
                    break
                
        except (ValueError, KeyError) as e:
            print(f"Invalid input: {e}")
    
    print("Interactive test completed.")