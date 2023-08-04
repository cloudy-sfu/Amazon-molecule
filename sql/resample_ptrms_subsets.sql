  -- Arguments: start_date, start_date, end_date
with PTRMS_15min_mean as (
select (timestamp - ?) / 900 as qoh_index,  -- qoh: quarter of hour
       height,
       avg(M1) as M1, avg(M2) as M2, avg(M3) as M3, avg(M4) as M4, avg(M5) as M5,
       avg(M6) as M6, avg(M7) as M7, avg(M8) as M8, avg(M9) as M9, avg(M10) as M10,
       avg(M11) as M11, avg(M12) as M12, avg(M13) as M13
from PTRMS
where timestamp >= ? and timestamp <= ?
group by qoh_index, height
)
select *
from (select * from PTRMS_15min_mean where height = 80) t1 left outer join
     (select * from PTRMS_15min_mean where height = 150) t2
on t1.qoh_index = t2.qoh_index
left outer join (select * from PTRMS_15min_mean where height = 320) t3
on t1.qoh_index = t3.qoh_index
