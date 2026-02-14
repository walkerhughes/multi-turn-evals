# Happy Path: Pro Tier Account Creation

This scenario tests an account creation flow where the user asks about plan differences before selecting the pro plan.

```json
{
  "eval_mode": "trajectory",
  "turns": [
    {
      "user_message": "Hello! I'm interested in signing up",
      "turn_notes": "greeting"
    },
    {
      "user_message": "Sure, I'm Marcus Chen",
      "turn_notes": "collect_name"
    },
    {
      "user_message": "You can reach me at [email protected]",
      "turn_notes": "collect_email"
    },
    {
      "user_message": "Got it, the verification code is 123456",
      "turn_notes": "verify_email"
    },
    {
      "user_message": "What's the difference between the free and pro plans?",
      "turn_notes": "select_plan"
    },
    {
      "user_message": "That sounds great, I'll take the pro plan",
      "turn_notes": "collect_preferences"
    },
    {
      "user_message": "I'd like push notifications enabled and would prefer light mode",
      "turn_notes": "confirm"
    },
    {
      "user_message": "Perfect, let's do it!",
      "turn_notes": "complete"
    }
  ]
}
```
