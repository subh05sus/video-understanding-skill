# The loop, worked

## A good loop

**Question:** "In this 12-minute product demo, how much does the Pro plan cost, and does he show the checkout?"

```
think   Price: probably said and shown. Checkout: a screen, so frames. Depth normal.
act     overview demo.mp4 -q "Pro price + checkout shown?" --depth normal
observe 12:04, 63 cuts, speech 81%. The timeline has "…pricing page…" at 4:10–4:52, and the segment
        8:30–9:40 is fast, with "…let's buy it…". Storyboard: a pricing table at 4:31, a card form at 9:02.
think   Search first (free), then look at the two candidates. Skip 0:00–4:00 (intro + features, no price
        words) and 5:00–8:30 (a still talking head).
act     search demo.mp4 "pro" "per month" "checkout" "€"
observe said 4:22 "the pro plan is twenty-nine a month"; shown 4:30 "Pro €29/mo"; said 8:58 "checkout".
act     frames demo.mp4 --from 4:20 --to 4:40 --count 6 --why "find the pricing table"
observe The table is from 4:26 to 4:38. The text is too small to quote.
act     zoom demo.mp4 --at 4:31 --region center --why "read the Pro price"
observe "Pro €29 / month, billed yearly (€348)". The "billed yearly" was never said.
act     frames demo.mp4 --from 8:55 --to 9:20 --count 9 --why "is there a checkout?"
observe 9:01: a Stripe card form; 9:12: "Payment successful". It's fast: 9:03–9:10 is a blur at this rate.
act     frames demo.mp4 --from 9:03 --to 9:10 --fps 6 --why "what happens between form and success"
observe He types a test card number (4242…) and clicks Pay at 9:06.5. It's a test-mode checkout.
act     note demo.mp4 "Pro €29/mo billed yearly €348 (4:31); test-mode Stripe checkout 9:01–9:12" --at 4:31
answer  Pro costs €29 a month, billed yearly (€348). The yearly part is only on screen, never said…
```

That's 30 frames used out of 150. It's sure of the price because the price was read at full resolution. It
noticed "test mode" only because it re-watched the fast part.

## The passive version (don't)

```
act     frames demo.mp4 --from 0 --to end --count 40
observe 40 small frames, one every 18 s. A pricing table is visible at 4:30 (unreadable). No checkout
        frame: the sample went from 8:48 to 9:06 and the form was only up from 9:01 to 9:12.
answer  "The price isn't clearly shown; there seems to be no checkout."   ← wrong twice
```

## Habits

- **Say the plan, and what you're skipping.** "Skipping X because the timeline shows Y" is a good line.
  "I looked at everything" is a bad one.
- **Change senses when one is stuck.** A blurry price can be OCR'd. A mumbled word can be read off the
  on-screen caption. A missed click shows up as a spike in the motion map.
- **Higher fps only where it pays**: a second or two around a change, never across a whole minute.
- **Compare before concluding "not there."** Search the transcript and the on-screen text, check the
  segments the timeline gives as candidates, then say which you checked.
- **Several videos:** run one overview over all of them, then the same probe on each. Answer in a table.
