from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from calendar import monthrange


@dataclass
class TemporalInfo:
    text: str = ""
    start: int | None = None
    end: int | None = None
    granularity: str = "point"
    relation: str = "at"
    confidence: float = 1.0

    @property
    def timestamp(self) -> int | None:
        return self.start


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _base(ts: int) -> datetime:
    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc)


def _year_range(year: int) -> tuple[int, int]:
    start = _ms(datetime(year, 1, 1, tzinfo=timezone.utc))
    end = _ms(datetime(year, 12, 31, 23, 59, 59, 999000, tzinfo=timezone.utc))
    return start, end


def _month_range(year: int, month: int) -> tuple[int, int]:
    start = _ms(datetime(year, month, 1, tzinfo=timezone.utc))
    end = _ms(datetime(year, month, monthrange(year, month)[1], 23, 59, 59, 999000, tzinfo=timezone.utc))
    return start, end


def normalize_temporal(text: str, reference_ts: int) -> TemporalInfo:
    """Normalize common Chinese and English temporal expressions.

    The parser is deterministic and intentionally conservative. It supplies
    retrieval/order constraints; it does not try to answer temporal questions
    by itself.
    """
    if not text:
        return TemporalInfo()

    base = _base(reference_ts)
    year = base.year

    # Absolute dates: 2026-03-12 / 2026年3月12日 / March 12, 2026.
    m = re.search(r"(20\d{2})[-/.年](\d{1,2})[-/.月](\d{1,2})日?", text)
    if m:
        y, mo, d = map(int, m.groups())
        dt = datetime(y, mo, d, tzinfo=timezone.utc)
        start = _ms(dt)
        return TemporalInfo(m.group(0), start, start + 86_399_999, "day", "at")

    m = re.search(
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?[,]?\s+(20\d{2})\b",
        text, flags=re.I,
    )
    if m:
        month = datetime.strptime(m.group(1)[:3], "%b").month
        d, y = int(m.group(2)), int(m.group(3))
        dt = datetime(y, month, d, tzinfo=timezone.utc)
        start = _ms(dt)
        return TemporalInfo(m.group(0), start, start + 86_399_999, "day", "at")

    # Month: 2026年3月 / March 2026 / July 2022.
    m = re.search(r"(20\d{2})年(\d{1,2})月", text)
    if m:
        y, mo = map(int, m.groups())
        start, end = _month_range(y, mo)
        return TemporalInfo(m.group(0), start, end, "month", "at")

    m = re.search(
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(20\d{2})\b",
        text, flags=re.I,
    )
    if m:
        month = datetime.strptime(m.group(1)[:3], "%b").month
        y = int(m.group(2))
        start, end = _month_range(y, month)
        return TemporalInfo(m.group(0), start, end, "month", "at")

    # Bare year: 2022 / in 2022.
    m = re.search(r"\b((?:19|20)\d{2})\b", text)
    if m:
        y = int(m.group(1))
        start, end = _year_range(y)
        return TemporalInfo(m.group(0), start, end, "year", "at", 0.92)

    number_words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    # Relative English intervals: 4 years ago / two months later.
    m = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(day|week|month|year)s?\s+ago\b", text, flags=re.I)
    if m:
        n = number_words.get(m.group(1).lower(), int(m.group(1)) if m.group(1).isdigit() else 1)
        unit = m.group(2).lower()
        days = n * {"day": 1, "week": 7, "month": 30, "year": 365}[unit]
        end = reference_ts - (days - 1) * 86_400_000
        start = reference_ts - days * 86_400_000
        return TemporalInfo(m.group(0), start, end, unit, "before", 0.88)

    m = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(day|week|month|year)s?\s+(?:later|after)\b", text, flags=re.I)
    if m:
        n = number_words.get(m.group(1).lower(), int(m.group(1)) if m.group(1).isdigit() else 1)
        unit = m.group(2).lower()
        days = n * {"day": 1, "week": 7, "month": 30, "year": 365}[unit]
        start = reference_ts + days * 86_400_000
        end = start + 86_399_999
        return TemporalInfo(m.group(0), start, end, unit, "after", 0.78)

    # Relative Chinese intervals.
    chinese_numbers = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
                       "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    m = re.search(r"([一二两三四五六七八九十\d]+)天前", text)
    if m:
        raw_n = m.group(1)
        n = chinese_numbers.get(raw_n, int(raw_n) if raw_n.isdigit() else 1)
        end = reference_ts - (n - 1) * 86_400_000
        start = reference_ts - n * 86_400_000
        return TemporalInfo(m.group(0), start, end, "day", "before", 0.88)

    m = re.search(r"([一二两三四五六七八九十\d]+)周前", text)
    if m:
        raw_n = m.group(1)
        n = chinese_numbers.get(raw_n, int(raw_n) if raw_n.isdigit() else 1)
        end = reference_ts - (n * 7 - 1) * 86_400_000
        start = reference_ts - n * 7 * 86_400_000
        return TemporalInfo(m.group(0), start, end, "week", "before", 0.86)

    for phrase, delta in (("上周", -1), ("本周", 0), ("下周", 1)):
        if phrase in text:
            week_start = base - timedelta(days=base.weekday())
            start_dt = datetime(week_start.year, week_start.month, week_start.day, tzinfo=timezone.utc) + timedelta(weeks=delta)
            end_dt = start_dt + timedelta(days=6, hours=23, minutes=59, seconds=59, milliseconds=999)
            relation = "before" if delta < 0 else ("after" if delta > 0 else "at")
            return TemporalInfo(phrase, _ms(start_dt), _ms(end_dt), "week", relation, 0.94)

    m = re.search(r"(\d+)个月前", text)
    if m:
        n = int(m.group(1))
        total = base.year * 12 + base.month - 1 - n
        y, mo = divmod(total, 12)
        mo += 1
        start, end = _month_range(y, mo)
        return TemporalInfo(m.group(0), start, end, "month", "before", 0.86)

    # Relative Chinese intervals.
    m = re.search(r"(\d+)年(?:前|以前)", text)
    if m:
        n = int(m.group(1))
        end = reference_ts - n * 365 * 86_400_000
        start = end - 31 * 86_400_000
        return TemporalInfo(m.group(0), start, end, "year", "before", 0.82)

    for phrase, delta in (("前年", -2), ("去年", -1), ("明年", 1)):
        if phrase in text:
            y = year + delta
            start, end = _year_range(y)
            relation = "before" if delta < 0 else "after"
            return TemporalInfo(phrase, start, end, "year", relation, 0.95)

    if "今年" in text:
        start, end = _year_range(year)
        return TemporalInfo("今年", start, end, "year", "at", 0.98)

    if "上个月" in text or "上月" in text:
        y, mo = year, base.month - 1
        if mo == 0:
            y, mo = y - 1, 12
        start, end = _month_range(y, mo)
        return TemporalInfo("上个月", start, end, "month", "before", 0.95)

    if "这个月" in text or "本月" in text:
        start, end = _month_range(year, base.month)
        return TemporalInfo("本月", start, end, "month", "at", 0.98)

    if "昨天" in text:
        day = base - timedelta(days=1)
        start = _ms(datetime(day.year, day.month, day.day, tzinfo=timezone.utc))
        return TemporalInfo("昨天", start, start + 86_399_999, "day", "before", 0.98)

    if "今天" in text:
        start = _ms(datetime(year, base.month, base.day, tzinfo=timezone.utc))
        return TemporalInfo("今天", start, start + 86_399_999, "day", "at", 0.98)

    if "明天" in text:
        day = base + timedelta(days=1)
        start = _ms(datetime(day.year, day.month, day.day, tzinfo=timezone.utc))
        return TemporalInfo("明天", start, start + 86_399_999, "day", "after", 0.98)

    if "以前" in text or "之前" in text:
        return TemporalInfo("以前" if "以前" in text else "之前", None, reference_ts, "open", "before", 0.85)

    if "之后" in text or "后来" in text:
        return TemporalInfo("之后" if "之后" in text else "后来", reference_ts, None, "open", "after", 0.85)

    if "最近" in text or re.search(r"\brecently\b", text, flags=re.I):
        return TemporalInfo("最近", reference_ts - 30 * 86_400_000, reference_ts, "range", "near", 0.8)

    return TemporalInfo()


def temporal_filter(info: TemporalInfo, timestamp: int) -> bool:
    if info.start is not None and timestamp < info.start:
        return False
    if info.end is not None and timestamp > info.end:
        return False
    return True
