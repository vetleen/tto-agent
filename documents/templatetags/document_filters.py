from django import template
from django.utils import timezone

register = template.Library()


@register.filter
def relative_upload_date(value):
    """Format a datetime as 'today at HH.mm', 'yesterday', 'x days ago', 'x months ago', or 'x years ago'."""
    if value is None:
        return ""
    now = timezone.now()
    # Ensure we're comparing in the same timezone
    if timezone.is_naive(value):
        value = timezone.make_aware(value)
    value = timezone.localtime(value)
    now = timezone.localtime(now)

    today = now.date()
    upload_date = value.date()
    delta = today - upload_date

    if delta.days == 0:
        return f"today at {value.strftime('%H.%M')}"
    if delta.days == 1:
        return "yesterday"
    if delta.days <= 30:
        return f"{delta.days} days ago"
    # Approximate months (30 days) and years (365 days)
    months = delta.days // 30
    if months <= 11:
        return "1 month ago" if months == 1 else f"{months} months ago"
    years = delta.days // 365
    return "1 year ago" if years == 1 else f"{years} years ago"
