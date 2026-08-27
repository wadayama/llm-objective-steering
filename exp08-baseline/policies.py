"""Policy sets for the translation comparison.

CANONICAL: the seven benchmark policies, verbatim. keyword_baseline.py was
written against these.

PARAPHRASE: the same seven intents in wording an operator might plausibly
use instead. Authored after keyword_baseline.py was frozen, without
consulting its rules, and the baseline was not revised afterwards. Each
paraphrase inherits the check of the canonical scenario it restates, so a
correct translation is judged by exactly the same criterion.
"""

PARAPHRASE = [
    # T1: maximize total throughput
    ("T1", "Get as much total data through the link as you can."),
    ("T1", "We want peak aggregate rate right now."),
    ("T1", "Run the link for maximum capacity."),
    # T2: similar data rates across channels
    ("T2", "No channel should lag far behind the others in rate."),
    ("T2", "Even out the per-channel bit rates."),
    ("T2", "Keep the rates comparable so nobody gets starved."),
    # T3: equalize transmit power
    ("T3", "Spread the transmit power evenly over the channels."),
    ("T3", "Every channel should draw the same amount of power."),
    ("T3", "Use a flat power profile."),
    # T4: shut down channels 0-3
    ("T4", "Take the first four channels offline."),
    ("T4", "Channels 0 through 3 are out of service."),
    ("T4", "Stop transmitting on the four weakest channels."),
    # T5: prioritize channels 6 and 7
    ("T5", "Channels 6 and 7 matter most right now."),
    ("T5", "Favour the top two channels."),
    ("T5", "Give preference to channel 6 and channel 7."),
    # T6: minimize power subject to sum rate >= 10 bits
    ("T6", "Use as little power as you can while still delivering at least "
           "10 bits in total."),
    ("T6", "We need 10 bits aggregate; spend the minimum power to get there."),
    ("T6", "Be as power-frugal as possible, but never drop below 10 bits "
           "overall."),
    # T7: minimize power subject to per-channel MI >= 1.0 bit
    ("T7", "Every channel must carry at least 1 bit; do it with minimal "
           "power."),
    ("T7", "No channel below 1 bit, and keep the power draw as low as "
           "possible."),
    ("T7", "Guarantee one bit per channel while minimising total power."),
]
