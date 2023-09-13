with date_and_time as (
    select cast(strftime('%Y', datetime(timestamp, 'unixepoch')) as integer) as year,
       cast(strftime('%m', datetime(timestamp, 'unixepoch')) as integer) as month,
       cast(strftime('%d', datetime(timestamp, 'unixepoch')) as integer) as day,
       cast(strftime('%H', datetime(timestamp, 'unixepoch')) as integer) as hour,
       strftime('%Y-%m-%d', datetime(timestamp, 'unixepoch')) as date,
       timestamp
    from PTRMS
)
select year, month,
       iif(
           strftime('%Y-%m-01', date) = min(date),
           iif(date(strftime('%Y-%m-01', date), '+1 month', '-1 day') = max(date), 'Complete', 'Start'),
           iif(date(strftime('%Y-%m-01', date), '+1 month', '-1 day') = max(date), 'End', '')
           )
           as complete,
       count(distinct date) as n_date, min(date), max(date),
       min(timestamp), max(timestamp)
from date_and_time
group by year, month