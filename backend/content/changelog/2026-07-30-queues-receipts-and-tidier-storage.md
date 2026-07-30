# Queues, receipts, and clips that don't linger

Housekeeping, mostly — the kind you only notice when it's missing.

**Your clip now tells you where it is in line.** Analysis is heavy work, and
until now several clips arriving at once all pretended to be processing while
they queued behind each other. The app now runs a fixed number at a time and
says so plainly: *"2 clips ahead of yours."* If we're genuinely at capacity, the
upload is refused up front, with your file kept ready for one-tap retry —
instead of being accepted and left to time out.

**Uploads are deleted on a schedule, not by luck.** Your video and its overlay
were always meant to be temporary, but the only thing that actually removed them
was a server restart. They now expire on a timer, whether or not we deploy.

**Expert Review leaves a receipt.** A one-time review doesn't change your plan,
so there was nowhere in the app that showed you'd bought one. Purchases now
appear on the pricing page with their status, from *payment received* through to
*delivered*.

**No more accidental second subscription.** Picking a different plan while
already subscribed now opens billing so you can switch, rather than starting a
second subscription alongside the first.

**Fixed:** free reports said the overlay video "couldn't be rendered" when it
was simply a paid feature that had never been attempted. The progress read-out
no longer stalls other people's while it's being written.
