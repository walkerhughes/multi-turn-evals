# Happy Path: Enterprise Tier Account Creation

This scenario tests an account creation flow where the user asks about enterprise features before selecting the enterprise plan.

```json
{
  "eval_mode": "trajectory",
  "turns": [
    {
      "user_message": "Hey, I need to set up an account for my team",
      "turn_notes": "greeting"
    },
    {
      "user_message": "I'm Sofia Rodriguez",
      "turn_notes": "collect_name"
    },
    {
      "user_message": "My email is [email protected]",
      "turn_notes": "collect_email"
    },
    {
      "user_message": "Just received it - the code is 123456",
      "turn_notes": "verify_email"
    },
    {
      "user_message": "What kind of features does the enterprise plan include?",
      "turn_notes": "select_plan"
    },
    {
      "user_message": "Excellent, we'll need the enterprise plan then",
      "turn_notes": "collect_preferences"
    },
    {
      "user_message": "We'd like email notifications for the whole team and dark mode as default",
      "turn_notes": "confirm"
    },
    {
      "user_message": "All set, please proceed!",
      "turn_notes": "complete"
    }
  ]
}
```
