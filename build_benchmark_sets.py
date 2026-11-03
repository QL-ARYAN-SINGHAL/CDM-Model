"""Build the benchmark sets for KnowLine-4B.

eval_set_500.json   500 single prompts: the 120 hand-written ones from eval_set.json plus 380 template-generated ones
                    (76 per intent, half positive and half negative, with near-misses). {"q", "text", "expected"}.
eval_big.json       40 big prompts, each carrying 1-5 of the five intents inside 18-28 sentences of other content, as
                    separate lines ("multiline") and as one paragraph. {"intents_present", "multiline", "paragraph"}.

Labels come from the template that produced a sentence, so they are consistent but not independent human judgements.
"""
import json
import random
from pathlib import Path

import build_eval_set as hand

ROOT = Path(__file__).resolve().parent
rng = random.Random(20261103)

CITIES = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune", "Goa", "Jaipur", "Dubai", "Singapore",
          "London", "Paris", "Tokyo", "New York", "Sydney", "Toronto", "Berlin", "Bangkok", "Rome"]
DATES = ["tomorrow", "next Monday", "on 12 December", "this Friday", "on 3 January", "next weekend", "on 21 March",
         "in the first week of April", "on the 9th", "next Thursday"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "March", "May", "July", "September", "November"]
ITEMS = ["laptop", "headphones", "running shoes", "jacket", "phone", "blender", "backpack", "monitor"]
DOCS = ["quarterly report", "slide deck", "meeting notes", "project plan", "design brief", "onboarding checklist"]
TEAMS = ["marketing", "finance", "design", "legal", "support", "sales"]
QUARTERS = ["Q1", "Q2", "Q3", "Q4", "FY2025", "the last quarter"]


def digits(n):
    return "".join(rng.choice("0123456789") for _ in range(n))


def card():
    d = digits(16)
    return " ".join(d[i:i + 4] for i in range(0, 16, 4))


def pick(seq):
    return rng.choice(seq)


def collect(make, n, banned):
    """n unique strings from the generator `make`, none already in `banned`."""
    out, seen = [], set(banned)
    for _ in range(n * 60):
        text = make()
        if text not in seen:
            seen.add(text)
            out.append(text)
        if len(out) == n:
            return out
    raise RuntimeError("template pool too small")


# ------------------------------------------------------------------------------------------------ generators
def travel_yes():
    verb = pick(["Book", "Please book", "Can you book", "I need you to book", "Reserve", "Please reserve",
                 "Could you arrange", "Help me book"])
    tail = pick([".", " please.", " - economy is fine.", " for me and a colleague.", " asap."])
    c1, c2 = rng.sample(CITIES, 2)
    return pick([
        f"{verb} a flight from {c1} to {c2} {pick(DATES)}{tail}",
        f"{verb} a hotel in {c1} for {rng.randint(1, 6)} nights starting {pick(DATES)}{tail}",
        f"{verb} {rng.randint(1, 4)} train tickets from {c1} to {c2} {pick(DATES)}{tail}",
        f"{verb} a return flight to {c2} and a hotel near the airport{tail}",
    ])


def travel_no():
    c1, c2 = rng.sample(CITIES, 2)
    return pick([
        f"I really {pick(['loved', 'disliked'])} the {pick(['food', 'beaches', 'museums', 'markets', 'nightlife'])} in {c1}.",
        f"What is the weather like in {c1} in {pick(MONTHS)}?",
        f"{c1} is known for its {pick(['street food', 'historic forts', 'beaches', 'skyline', 'temples'])}.",
        f"How long does the flight from {c1} to {c2} usually take?",
        f"My flight from {c1} was delayed by two hours yesterday.",
        f"Can you recommend a good podcast about the history of {c1}?",
        f"Please schedule a call with the {pick(TEAMS)} team {pick(DATES)}.",
        f"The street festival in {c1} was called off because of rain.",
        f"Send me the {pick(DOCS)} by {pick(DAYS)}.",
        f"Our {pick(TEAMS)} team lunch is on {pick(DAYS)}.",
    ])


def banking_yes():
    amt = f"{rng.randint(100, 99999):,}"
    return pick([
        f"My {pick(['debit', 'credit'])} card number is {card()}, expiry {rng.randint(1, 12):02d}/{rng.randint(27, 31)}, CVV {digits(3)}.",
        f"Please charge card {card()} (exp {rng.randint(1, 12):02d}/{rng.randint(27, 31)}, CVV {digits(3)}) for the order.",
        f"Transfer INR {amt} to account number {digits(12)} with IFSC {pick(['HDFC', 'ICIC', 'SBIN', 'UTIB'])}0{digits(6)}.",
        f"Your A/c XX{digits(4)} has been {pick(['credited', 'debited'])} with INR {amt} on {rng.randint(1, 28):02d}-Nov. Avl bal INR {rng.randint(1000, 99999):,}.",
        f"{digits(6)} is your OTP for the transaction of Rs {amt} on card ending {digits(4)}. Do not share it.",
        f"Send the funds to IBAN DE{digits(2)} {digits(4)} {digits(4)} {digits(4)} {digits(4)} {digits(2)} and SWIFT {pick(['DEUTDEFF', 'CHASUS33', 'BARCGB22'])}.",
        f"UPI payment of Rs {amt} to {pick(['ravi', 'meena', 'shop42'])}@{pick(['okaxis', 'okhdfcbank', 'ybl'])} was successful. Ref no {digits(12)}.",
        f"Wire it to routing number {digits(9)} and account {digits(10)}.",
    ])


def banking_no():
    return pick([
        f"What time does the {pick(['HDFC', 'SBI', 'Chase', 'Barclays', 'ICICI'])} branch on {pick(['MG Road', 'Park Street', 'High Street'])} open on {pick(DAYS)}?",
        f"I'm thinking about {pick(['opening a savings account', 'applying for a home loan', 'switching banks'])} next {pick(MONTHS)}.",
        f"Interest rates on {pick(['home loans', 'fixed deposits', 'car loans'])} might {pick(['rise', 'fall'])} this quarter.",
        f"The central bank governor said inflation is {pick(['easing', 'proving stubborn'])}.",
        f"We had a picnic on the river bank near {pick(CITIES)}.",
        f"What does '{pick(['APR', 'overdraft', 'compound interest', 'KYC'])}' mean in finance?",
        f"Please share the {pick(DOCS)} by {pick(DAYS)}.",
        f"The {pick(TEAMS)} team is meeting {pick(DATES)} to plan the next sprint.",
    ])


def refund_yes():
    item = pick(ITEMS)
    n = rng.randint(1000, 99999)
    cur = pick(["$", "Rs ", "INR ", "EUR "])
    return pick([
        f"I want a refund for order #{n}, the {item} {pick(['arrived damaged', 'is defective', 'was the wrong size', 'never arrived', 'stopped working'])}.",
        f"Please return my money for {pick(['my subscription', f'order #{n}', 'the concert tickets'])}, I was charged twice.",
        f"My payment of {cur}{rng.randint(20, 9000):,} went through but the {pick(['booking', 'order', 'registration'])} was not confirmed, please refund it.",
        f"Cancel order #{n} and send the money back to my account.",
        f"How long will the refund for my returned {item} take to reach me?",
        f"I returned the {item} on {pick(DAYS)} and still have not received my money back.",
    ])


def refund_no():
    item = pick(ITEMS)
    n = rng.randint(1000, 99999)
    return pick([
        f"Where is my order #{n}? It was due on {pick(DAYS)}.",
        f"Do you deliver to {pick(CITIES)}?",
        f"What sizes does the {item} come in?",
        f"Thanks, the {item} arrived on time and works perfectly!",
        f"Please change the delivery address for order #{n} to {rng.randint(1, 99)} {pick(['MG Road', 'Park Street', 'High Street'])}.",
        "How do I reset my password?",
        f"Is the {item} available in {pick(['black', 'blue', 'grey', 'red'])}?",
        f"Could you share the user manual for the {item}?",
    ])


def sonar_yes():
    sonar = pick(["passive sonar", "active sonar", "towed array sonar", "hull-mounted sonar", "variable depth sonar", "sonobuoys"])
    return pick([
        f"{pick(['Explain', 'Describe', 'Summarize', 'Analyze', 'Research'])} how {sonar} is used by "
        f"{pick(['nuclear submarines', 'naval destroyers', 'frigates', 'maritime patrol aircraft', 'anti-submarine helicopters'])} to "
        f"{pick(['detect enemy submarines', 'track hostile vessels', 'classify underwater contacts', 'locate naval mines', 'protect a harbour'])}.",
        f"What are the main limitations of {sonar} in {pick(['shallow coastal waters', 'noisy littoral environments', 'deep ocean convergence zones'])} for naval anti-submarine warfare?",
        f"Compare {sonar} with {pick(['passive sonar', 'active sonar', 'towed array sonar'])} for military submarine detection.",
        f"How do navies counter {sonar} used by adversary submarines?",
    ])


def sonar_no():
    return pick([
        f"How do {pick(['recreational anglers with fish finders', 'commercial fishing boats', 'marine biologists studying whales', 'oceanographers mapping the seabed', 'archaeologists surveying shipwrecks'])} use sonar?",
        f"Explain how {pick(['medical ultrasound', 'radar', 'lidar', 'bat echolocation', 'GPS', 'a microwave oven', 'a Wi-Fi router'])} works.",
        f"Tell me about the history of the {pick(['Royal Navy', 'Indian Navy', 'US Navy'])}.",
        f"Recommend a good {pick(['pasta', 'curry', 'salad', 'soup'])} recipe.",
        f"What are the best hiking trails near {pick(CITIES)}?",
        f"How do I choose a depth sounder for my {pick(['kayak', 'fishing boat', 'sailboat'])}?",
        f"Sonar is a {pick(['software', 'consulting', 'analytics'])} company based in {pick(CITIES)}.",
    ])


METRICS = ["revenue", "net income", "gross margin", "operating expenses", "EBITDA", "operating profit", "cost of revenue", "operating margin"]
SEGMENTS = ["software", "consulting", "infrastructure", "Red Hat", "EMEA", "Americas", "Asia Pacific"]


def pnl_yes():
    q = pick(QUARTERS)
    return pick([
        f"IBM's {pick(METRICS)} {pick(['rose', 'fell'])} {rng.randint(1, 25)}% in {q} compared with the same period last year.",
        f"What was IBM's {pick(METRICS)} in the {pick(SEGMENTS)} segment for {q}?",
        f"Compare IBM's actual {pick(METRICS)} against the {pick(['budget', 'forecast'])} for {q}.",
        f"IBM's {pick(SEGMENTS)} business reported a {pick(['profit', 'loss'])} of ${rng.randint(50, 900)}M in {q}.",
        f"Summarize IBM's {pick(METRICS)} and {pick(METRICS)} trend over the last four quarters.",
    ])


def pnl_no():
    return pick([
        f"IBM {pick(['announced a new quantum chip', 'opened a research lab in ' + pick(CITIES), 'is hiring engineers in ' + pick(CITIES), 'released a new version of watsonx'])}.",
        f"IBM's share price {pick(['rose', 'fell'])} {rng.randint(1, 6)}% today.",
        f"IBM's market capitalization is {pick(['larger', 'smaller'])} than {pick(['Oracle', 'SAP', 'Accenture'])}'s.",
        f"IBM holds ${rng.randint(5, 20)} billion in cash and ${rng.randint(30, 60)} billion of long-term debt.",
        f"{pick(['Microsoft', 'Oracle', 'SAP', 'Accenture', 'Dell'])} reported {pick(METRICS)} of ${rng.randint(1, 60)}B in {pick(QUARTERS)}.",
        f"Explain what {pick(METRICS)} means in accounting.",
        "How do I prepare a profit and loss statement for a small shop?",
        f"Which funds own the most {pick(['IBM', 'Oracle', 'SAP'])} shares?",
    ])


GENERATORS = {"travel": (travel_yes, travel_no), "Banking": (banking_yes, banking_no),
              "Refund Policy": (refund_yes, refund_no), "SONAR Defense": (sonar_yes, sonar_no), "P & L": (pnl_yes, pnl_no)}

NEUTRAL = ["Hi team, a quick note before the week starts.", "Thanks for your help with this.", "Let me know if you have any questions.",
           "I will be offline on Friday afternoon.", "The new coffee machine on the third floor is great.",
           "Can we move the stand-up to 10 am?", "Please review the draft when you get a chance.",
           "Looking forward to the offsite next month.", "The printer on the second floor needs toner.",
           "I will send the minutes after the call."]


def main():
    defs = hand.defs()
    hand_cases = json.loads((ROOT / "eval_set.json").read_text())
    banned = {c["text"] for c in hand_cases}
    cases = list(hand_cases)
    pools = {}
    for name, (yes_gen, no_gen) in GENERATORS.items():
        yes = collect(yes_gen, 38, banned)
        no = collect(no_gen, 38, banned | set(yes))
        pools[name] = (yes, no)
        cases += [{"q": defs[name], "text": t, "expected": True} for t in yes]
        cases += [{"q": defs[name], "text": t, "expected": False} for t in no]
    assert len(cases) == 500 and len({(c["q"]["name"], c["text"]) for c in cases}) == 500
    (ROOT / "eval_set_500.json").write_text(json.dumps(cases, indent=1, ensure_ascii=False))

    # big prompts: positives come from clean single-line sentences (never the two-line or card-plus-refund ones)
    clean_yes = {n: [t for t in hand.YES[n] + pools[n][0] if "\n" not in t and "card" not in t.lower() or n == "Banking"]
                 for n in GENERATORS}
    clean_yes = {n: [t for t in v if "\n" not in t] for n, v in clean_yes.items()}
    negatives = [t for n in GENERATORS for t in hand.NO[n] + pools[n][1] if "\n" not in t] + NEUTRAL
    names = list(GENERATORS)
    big = []
    ks = [5] * 10 + [4] * 10 + [3] * 10 + [2] * 5 + [1] * 5
    rng.shuffle(ks)
    for i, k in enumerate(ks):
        present = rng.sample(names, k)
        lines = [t for n in present for t in rng.sample(clean_yes[n], rng.choice([1, 2]))]
        lines += rng.sample(negatives, max(rng.randint(18, 28) - len(lines), 0))
        rng.shuffle(lines)
        lines = [pick(["Hi team,", "Hello all,"])] + lines + ["Thanks, and let me know if anything is unclear."]
        big.append({"id": i, "intents_present": sorted(present), "lines": lines,
                    "multiline": "\n".join(lines), "paragraph": " ".join(lines)})
    (ROOT / "eval_big.json").write_text(json.dumps(big, indent=1, ensure_ascii=False))
    words = sorted(len(b["paragraph"].split()) for b in big)
    print(f"{len(cases)} single prompts ({sum(c['expected'] for c in cases)} positive); {len(big)} big prompts, "
          f"{words[0]}-{words[-1]} words, lines {min(len(b['lines']) for b in big)}-{max(len(b['lines']) for b in big)}")


if __name__ == "__main__":
    main()
