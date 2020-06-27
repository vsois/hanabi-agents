from .ruleset import Ruleset

flawed_rules = [
    Ruleset.play_safe_card,
    Ruleset.play_probably_safe_factory(0.25),
    Ruleset.tell_randomly,
    Ruleset.osawa_discard,
    Ruleset.discard_oldest_first,
    Ruleset.discard_randomly
]

iggi_rules = [
    Ruleset.play_if_certain,
    Ruleset.play_safe_card,
    Ruleset.tell_playable_card_outer,
    Ruleset.osawa_discard,
    Ruleset.discard_oldest_first,
    Ruleset.legal_random
]

outer_rules = [
    Ruleset.play_safe_card,
    Ruleset.osawa_discard,
    Ruleset.tell_playable_card_outer,
    Ruleset.tell_unknown,
    Ruleset.discard_randomly
]

piers_rules = [
    Ruleset.hail_mary,
    Ruleset.play_safe_card,
    Ruleset.play_probably_safe_factory(0.6, True),
    Ruleset.tell_anyone_useful_card,
    Ruleset.tell_dispensable_factory(3),
    Ruleset.osawa_discard,
    Ruleset.discard_oldest_first,
    Ruleset.tell_randomly,
    Ruleset.discard_randomly
]
