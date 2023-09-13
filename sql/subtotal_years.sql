with date_and_time as (
    select cast(strftime('%Y', datetime(timestamp, 'unixepoch')) as integer) as year,
       cast(strftime('%m', datetime(timestamp, 'unixepoch')) as integer) as month,
       cast(strftime('%d', datetime(timestamp, 'unixepoch')) as integer) as day,
       cast(strftime('%H', datetime(timestamp, 'unixepoch')) as integer) as hour,
       strftime('%Y-%m-%d', datetime(timestamp, 'unixepoch')) as date,
       timestamp
    from PTRMS
)
select year, count(distinct date) as n_date
from date_and_time
group by year