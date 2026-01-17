import dataclasses
from datetime import date, timedelta, datetime


from .sources.model import FeedItem
from .log import logger


class Timeshift:
    def __init__(
        self, *, for_feeditems: set[str], timeshift_between: tuple[date, date]
    ):
        self.for_feeditems = for_feeditems
        self.earliest_start_year = date(1940, 1, 1)
        (
            self.started_watching_media,
            self.account_created,
        ) = timeshift_between  # 2000, 2016

    def matches(self, feeditem: FeedItem) -> bool:
        return (
            feeditem.ftype in self.for_feeditems
            and feeditem.when is not None
            and feeditem.when.date() < self.account_created
        )

    def _determine_timeshift(self, feeditem: FeedItem) -> date | None:
        # shift items before the account creation date to somewhere between the media start and end date
        # leave items after the account creation date alone

        # something released in 1920 -> 2005
        # something released in 1925 -> 2006
        # something released in 1980 -> 2009
        # something released in 2000 -> 2014
        # something released in 2016 -> 2016
        # something released in 2017 -> None

        if feeditem.when.date() < self.account_created:
            if feeditem.when.date() < self.earliest_start_year:
                return self.started_watching_media

            # feeditem date is larger than min date, this is the number of days between the two
            numerator: timedelta = feeditem.when.date() - self.earliest_start_year
            denominator: timedelta = self.account_created - self.earliest_start_year

            # frac is the fraction of the way between the min and max date
            frac = numerator.days / denominator.days
            assert 1 >= frac >= 0

            # scale the fraction to the number of days between started_watching_media and account_created
            # and add that to started_watching_media
            add_to_diff: timedelta = frac * (
                self.account_created - self.started_watching_media
            )
            shifted_date: date = self.started_watching_media + add_to_diff
            return shifted_date

        else:
            return None

    def timeshift(self, feeditem: FeedItem) -> FeedItem | None:
        if new_date := self._determine_timeshift(feeditem):
            logger.debug(
                f"timeshift {feeditem.ftype} {feeditem.title} from {feeditem.when.date()} to {new_date}"
            )
            data = dataclasses.asdict(feeditem)
            data["when"] = datetime.combine(
                new_date, feeditem.when.time(), tzinfo=feeditem.when.tzinfo
            )
            return FeedItem(**data)
        return None
