"""Static metadata for the xBD disaster events.

Two things live here that the raw dataset does not give us cleanly:

1. A normalised `event_type` vocabulary. xBD's per-tile metadata carries a
   `disaster_type` field, but its vocabulary is coarse ("wind" covers both
   hurricanes and tornadoes). We want the distinction, because it is exactly the
   kind of context the model is supposed to condition on.

2. An event date, so `days_since_event` can be derived from each tile's capture
   date rather than invented.

The dates below are APPROXIMATE — landfall, main shock, or the widely cited
start of the event. They are good enough to bucket imagery into "days after"
but should be verified before being quoted anywhere. Where an event unfolded
over weeks (wildfires, floods), the date is the start.
"""

from __future__ import annotations

from datetime import date

# Normalised event types. Note that xBD includes tornadoes, which the project
# brief's list omits.
EVENT_TYPES = [
    "hurricane",
    "tornado",
    "wildfire",
    "flood",
    "tsunami",
    "earthquake",
    "volcanic-eruption",
]

EVENTS: dict[str, dict] = {
    "guatemala-volcano": {"event_type": "volcanic-eruption", "date": date(2018, 6, 3)},
    "hurricane-florence": {"event_type": "hurricane", "date": date(2018, 9, 14)},
    "hurricane-harvey": {"event_type": "hurricane", "date": date(2017, 8, 25)},
    "hurricane-matthew": {"event_type": "hurricane", "date": date(2016, 10, 4)},
    "hurricane-michael": {"event_type": "hurricane", "date": date(2018, 10, 10)},
    "joplin-tornado": {"event_type": "tornado", "date": date(2011, 5, 22)},
    "lower-puna-volcano": {"event_type": "volcanic-eruption", "date": date(2018, 5, 3)},
    "mexico-earthquake": {"event_type": "earthquake", "date": date(2017, 9, 19)},
    "midwest-flooding": {"event_type": "flood", "date": date(2019, 3, 15)},
    "moore-tornado": {"event_type": "tornado", "date": date(2013, 5, 20)},
    "nepal-flooding": {"event_type": "flood", "date": date(2017, 8, 11)},
    "palu-tsunami": {"event_type": "tsunami", "date": date(2018, 9, 28)},
    "pinery-bushfire": {"event_type": "wildfire", "date": date(2015, 11, 25)},
    "portugal-wildfire": {"event_type": "wildfire", "date": date(2017, 6, 17)},
    "santa-rosa-wildfire": {"event_type": "wildfire", "date": date(2017, 10, 8)},
    "socal-fire": {"event_type": "wildfire", "date": date(2017, 12, 4)},
    "sunda-tsunami": {"event_type": "tsunami", "date": date(2018, 12, 22)},
    "tuscaloosa-tornado": {"event_type": "tornado", "date": date(2011, 4, 27)},
    "woolsey-fire": {"event_type": "wildfire", "date": date(2018, 11, 8)},
}


def event_type_for(disaster: str) -> str:
    """Normalised event type for an xBD disaster name, or 'unknown'."""
    return EVENTS.get(disaster, {}).get("event_type", "unknown")


def days_since_event(disaster: str, capture: date | None) -> int | None:
    """Days between the event and the image capture.

    Returns None when the event is unknown, the capture date is missing, or the
    arithmetic produces something implausible (negative, or more than two years
    out) — which usually means the capture date belongs to the pre-event image
    or the event date above is wrong.
    """
    meta = EVENTS.get(disaster)
    if meta is None or capture is None:
        return None
    delta = (capture - meta["date"]).days
    if delta < 0 or delta > 730:
        return None
    return delta
