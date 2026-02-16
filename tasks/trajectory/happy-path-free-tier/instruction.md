# Happy Path: Free Tier Account Creation

This scenario tests a straightforward account creation flow where the user selects the free plan.

```json
{
  "eval_mode": "trajectory",
  "turns": [
    {
      "user_message": "Hi there!",
      "turn_notes": "greeting"
    },
    {
      "user_message": "I'd like to create an account",
      "turn_notes": "collect_name"
    },
    {
      "user_message": "My name is Priya Sharma",
      "turn_notes": "collect_email"
    },
    {
      "user_message": "[email protected]",
      "turn_notes": "verify_email"
    },
    {
      "user_message": "The code is 123456",
      "turn_notes": "select_plan"
    },
    {
      "user_message": "I'll go with the free plan for now",
      "turn_notes": "collect_preferences"
    },
    {
      "user_message": "I prefer email notifications, and dark mode if that's an option",
      "turn_notes": "confirm"
    },
    {
      "user_message": "Yes, looks good to me!",
      "turn_notes": "complete"
    }
  ]
}
```
