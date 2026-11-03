"""Build eval_set.json: 120 labelled prompts (24 per intent) using the intent definitions already in kev_tracking.csv.

Each intent has 12 prompts the intent applies to and 12 it does not, including near-misses written from the intent's
own Don't text (general talk, other domains, same words used in another sense). Two prompts per intent are two lines, so
the bridge's per-line scoring is exercised. A label is true when the intent applies to the prompt (or any of its lines).
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def defs():
    found = {}
    with (ROOT / "archive" / "kev_0.8b_tracking_2026-11-03.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            for q in json.loads(row["questions"]):
                if q["name"] in ("travel", "Refund Policy", "SONAR Defense", "P & L"):
                    found[q["name"]] = q
    found["Banking"] = {"name": "Banking", "description": "Does this message contain any banking related information?",
                        "do": "Actual cards , cvv , banking messages , banking identifiers",
                        "dont": "General banking terms , banking words , non related words"}
    return found


YES = {
    "travel": [
        "Book me a flight from Mumbai to Dubai on 14 December.",
        "Please reserve a double room at the Taj in Jaipur for two nights starting Friday.",
        "I need to book a hotel near Heathrow for tonight.",
        "Can you get me a one-way ticket to Singapore next Tuesday?",
        "Book two economy seats on the 6am IndiGo flight to Bangalore.",
        "Find and book a cheap hotel in Paris for the first week of March.",
        "Reserve a flight to Tokyo for me and my wife, leaving on the 3rd.",
        "Hey, can you sort out a hotel in Goa for the team offsite, 10 rooms, 5th to 8th?",
        "I want to book a round trip to New York in January, business class.",
        "Get me a hotel room in Delhi tomorrow night, budget under 5000 rupees.",
        "Please book a flight to London for next Monday morning and a hotel near King's Cross.",
        "Hi team\nPlease book my flight to Hyderabad for Thursday evening.",
    ],
    "Banking": [
        "My debit card number is 5500 0000 0000 0004, expiry 08/27, CVV 456.",
        "Please transfer Rs 25,000 to account number 50100234567891, IFSC HDFC0000123.",
        "Your A/c XX7712 has been credited with INR 12,450.00 on 02-Nov. Avl bal 38,210.55.",
        "IBAN: GB29 NWBK 6016 1331 9268 19, please send the invoice payment there.",
        "OTP 482913 for your transaction of Rs 4,999 at Flipkart on card ending 4421.",
        "Wire the money to routing number 021000021 and account 123456789.",
        "SWIFT code is CHASUS33 and the account number is 000123456789.",
        "Your credit card ending 9087 has an outstanding amount of Rs 18,300 due on 12 Nov.",
        "UPI payment of Rs 600 to ramesh@okhdfcbank successful, ref no 330912334455.",
        "My CVV is 321 and the card is 4000 1234 5678 9010, use it for the booking.",
        "Alert: Rs 9,999 debited from A/c XX1190 via ATM withdrawal at 22:14.",
        "Hi accounts team\nRefund to my card 4242 4242 4242 4242 expiring 11/28.",
    ],
    "Refund Policy": [
        "I want a refund for my order #4471, the item was defective.",
        "Please return my money, I was charged twice for the same subscription.",
        "My payment failed but the amount was deducted from my account.",
        "Can I get my money back for the cancelled event ticket?",
        "I returned the shoes last week, when will the refund reach my card?",
        "The wrong product was delivered, I want a full refund.",
        "Please reverse the duplicate charge of $29.99 on my invoice.",
        "My order was cancelled by the seller, I need the payment refunded.",
        "I paid for the premium plan but never got access, refund me please.",
        "Hey Team, my order went back to the warehouse, I want my money back.",
        "Payment of Rs 1,200 was debited but the booking was not confirmed, please refund.",
        "Hello support\nI would like to claim a refund for the damaged laptop I bought on Monday.",
    ],
    "SONAR Defense": [
        "How do passive sonar arrays on nuclear submarines detect enemy vessels?",
        "Explain the principles of anti-submarine warfare sonar and its detection ranges.",
        "Describe how a towed array sonar is deployed by a naval frigate.",
        "What countermeasures do submarines use against active sonar pings from destroyers?",
        "Summarize the signal processing used in military sonar for target classification.",
        "Compare hull-mounted and variable depth sonar for naval surveillance.",
        "How does sonar help navies detect mines in littoral waters?",
        "Design a sonar-based underwater surveillance system for harbour defense.",
        "What are the limitations of active sonar in detecting quiet diesel-electric submarines?",
        "Explain how sonobuoys dropped by maritime patrol aircraft track submarines.",
        "Analyze the acoustic detection capabilities of the SOSUS underwater surveillance network.",
        "I'm writing a defense report.\nCan you explain how naval sonar tracks submarines using beamforming?",
    ],
    "P & L": [
        "IBM's net income rose 12% in Q2 as software revenue outpaced expectations.",
        "What was IBM's gross profit margin in the infrastructure segment last quarter?",
        "Show IBM's operating expenses versus budget for the consulting business in Q3.",
        "IBM reported a loss in its hybrid cloud unit because of higher restructuring costs.",
        "Summarize IBM's revenue and EBITDA trend over the last four quarters.",
        "IBM's EMEA region missed its profit forecast by 5% this year.",
        "How did IBM's operating income change year over year?",
        "Explain the variance between IBM's actual and planned net profit for FY2025.",
        "IBM's cost of revenue grew faster than sales, squeezing margins.",
        "Prepare a P&L summary for IBM covering revenue, costs and operating profit.",
        "IBM's Red Hat division generated $1.8B in revenue with a 21% operating margin.",
        "Quick question for finance.\nDid IBM's software segment beat its profit target this quarter?",
    ],
}

NO = {
    "travel": [
        "I loved the beaches in Goa last year.",
        "What is the best time to visit Kyoto?",
        "Flights have become so expensive lately.",
        "Our hotel was fine but the breakfast was bad.",
        "Explain how airlines decide ticket prices.",
        "Can you recommend some good books about the Silk Road?",
        "What is the baggage allowance on international flights?",
        "The travel industry is recovering after the pandemic.",
        "I'm working from home today, call me if you need anything.",
        "Write a poem about a long train journey.",
        "My flight landed an hour late yesterday.",
        "Hotels in this city are overpriced, in my opinion.",
    ],
    "Banking": [
        "Good morning bank manager, how is your day going?",
        "Banks usually open at 10 am on weekdays.",
        "I'm planning to apply for a home loan next year, any general tips?",
        "The river bank was flooded after the storm.",
        "What is the difference between a savings and a current account?",
        "I work at a bank and the hours are long.",
        "Interest rates are expected to fall next quarter.",
        "Can you explain what a credit score is?",
        "Let's grab lunch near the bank on Main Street.",
        "My friend says fixed deposits are safer than stocks.",
        "The central bank announced a new policy today.",
        "Please send me the meeting notes by evening.",
    ],
    "Refund Policy": [
        "Thanks for the quick delivery, the package looks great!",
        "Where is my order? It has not arrived yet.",
        "What are your store opening hours on Sundays?",
        "Can you tell me more about the features of this phone?",
        "I really like the new design of your website.",
        "Please update my delivery address to 12 MG Road.",
        "How do I change my account password?",
        "The weather is lovely today, let's go for a walk.",
        "Could you recommend a good laptop for programming?",
        "Please send me the product manual in PDF.",
        "I'd like to know if this jacket comes in blue.",
        "Great service, will order again!",
    ],
    "SONAR Defense": [
        "How do fish finders use sonar to locate fish for recreational anglers?",
        "Explain how ultrasound imaging works for pregnant women.",
        "Describe how whales use echolocation to find prey.",
        "What are the best hiking trails near Denver?",
        "How does radar help air traffic controllers track planes?",
        "Compare depth sounders used on commercial fishing boats.",
        "Sonar Inc is a software company that sells HR analytics tools.",
        "Explain how oceanographers use acoustic sensors to map the seabed for research.",
        "What is the history of the Royal Navy in the 18th century?",
        "How do submarines dive and surface using ballast tanks?",
        "Explain how bats navigate in the dark.",
        "Recommend a good recipe for chicken curry.",
    ],
    "P & L": [
        "IBM announced a new quantum computing roadmap with 1,000 qubits.",
        "IBM's share price rose 4% after the announcement.",
        "IBM has $9 billion of cash and $50 billion of long-term debt on its balance sheet.",
        "Microsoft reported record revenue and net income this quarter.",
        "Explain what EBITDA means and how analysts use it.",
        "IBM's new CEO talked about company culture and hiring plans.",
        "What is IBM's market capitalization compared with Oracle?",
        "How do I write a profit and loss statement for a small bakery?",
        "IBM is hosting a developer conference in Bangalore next month.",
        "Which institutions own the most IBM shares?",
        "IBM's Watson product helps hospitals analyze medical images.",
        "Let's schedule a meeting to review the vendor contract.",
    ],
}

if __name__ == "__main__":
    d = defs()
    cases = [{"q": d[name], "text": text, "expected": label}
             for name in d for label, pool in ((True, YES), (False, NO)) for text in pool[name]]
    assert len(cases) == 120 and len({(c["q"]["name"], c["text"]) for c in cases}) == 120
    (ROOT / "eval_set.json").write_text(json.dumps(cases, indent=1, ensure_ascii=False))
    print(len(cases), "cases written to eval_set.json")
