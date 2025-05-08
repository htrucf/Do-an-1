/* Để trích xuất được dữ liệu, phải được grant access trên Google BigQuery đối với các dataset sau:
1. physionet-data.mimiciv_ed (source: https://physionet.org/content/mimic-iv-ed/2.2/)
2. physionet-data.mimiciv_icu, physionet-data.mimiciv_hosp, physionet-data.mimiciv_derived (source: https://physionet.org/content/mimiciv/2.2/)
5. physionet-data.mimiciv_note (source: https://www.physionet.org/content/mimic-iv-note/2.2/)
Tất cả đều thuộc project physionet-data. */

-- Chạy truy vấn trên Google Cloud Console của Google BigQuery
WITH ed_data AS (
  SELECT 
    t.*, 
    e.hadm_id, 
    e.intime AS ed_intime, 
    e.outtime AS ed_outtime
  FROM `physionet-data.mimiciv_ed.triage` t
  JOIN `physionet-data.mimiciv_ed.edstays` e 
    ON t.stay_id = e.stay_id
WHERE
  t.heartrate BETWEEN 40 AND 200 AND
  t.resprate BETWEEN 10 AND 50 AND
  t.temperature BETWEEN 94.1 AND 109.4 AND
  t.o2sat BETWEEN 60 AND 100 AND
  t.sbp BETWEEN 60 AND 250 AND
  t.dbp BETWEEN 30 AND 150 AND
  SAFE_CAST(t.pain AS FLOAT64) IS NOT NULL
),

ed_icu_link AS (
  SELECT 
    ed.*, 
    i.stay_id AS icu_stay_id, 
    i.intime AS icu_intime,
  FROM ed_data ed
  JOIN `physionet-data.mimiciv_icu.icustays` i 
    ON ed.subject_id = i.subject_id
    AND ed.hadm_id = i.hadm_id
    AND i.intime BETWEEN ed.ed_intime AND TIMESTAMP_ADD(ed.ed_outtime, INTERVAL 24 HOUR)
)

SELECT DISTINCT
  eic.*, 
  CASE 
    WHEN p.gender = 'M' THEN 1
    WHEN p.gender = 'F' THEN 0
    ELSE NULL 
  END AS gender,
  EXTRACT(YEAR FROM eic.ed_intime) - p.anchor_year + p.anchor_age AS age,
  IF(s.stay_id IS NOT NULL, 1, 0) AS sepsis,
  d.hadm_id, d.text
FROM ed_icu_link eic
LEFT JOIN `physionet-data.mimiciv_hosp.patients` p
  ON eic.subject_id = p.subject_id
LEFT JOIN `physionet-data.mimiciv_derived.sepsis3` s 
  ON eic.icu_stay_id = s.stay_id
LEFT JOIN `physionet-data.mimiciv_derived.suspicion_of_infection` soi
  ON eic.icu_stay_id = soi.stay_id;

  
SELECT eic.subject_id, d.hadm_id, d.text FROM ed_icu_link eic
LEFT JOIN `physionet-data.mimiciv_note.discharge` d
  ON eic.icu_stay_id = d.stay_id;
