select (timestamp - :start_timestamp) / :sampling_seconds as time_index,
       height,
       avg(M1) as M1, avg(M2) as M2, avg(M3) as M3, avg(M4) as M4, avg(M5) as M5,
       avg(M6) as M6, avg(M7) as M7, avg(M8) as M8, avg(M9) as M9, avg(M10) as M10,
       avg(M11) as M11, avg(M12) as M12, avg(M13) as M13
from PTRMS
where timestamp >= :start_timestamp and timestamp <= :end_timestamp
group by time_index, height
