# Pilot: classical solar indices vs. geomagnetic Ap

Source: GFZ Potsdam combined Kp/Ap/SN/F10.7 file, 1932-01-01 to present, CC BY 4.0.
https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_Ap_SN_F107_since_1932.txt

## Sunspot number / F10.7 -> Ap (Pearson r, lag in days)

| lag | SN, full record (n=34544) | SN, SDO era (n=5934) | F10.7, full record (n=28366) | F10.7, SDO era (n=5922) |
|---|---|---|---|---|
| 0 | 0.158 | 0.155 | 0.168 | 0.168 |
| 1 | 0.162 | 0.153 | 0.175 | 0.174 |
| 2 | 0.165 | 0.156 | 0.184 | 0.180 |
| 3 | 0.165 | 0.156 | 0.183 | 0.188 |
| 5 | 0.162 | 0.153 | 0.170 | 0.167 |
| 7 | 0.155 | 0.142 | 0.162 | 0.150 |
| 14 | 0.132 | 0.111 | 0.141 | 0.123 |

All p-values effectively zero (huge n). Effect size is small: r ~ 0.15-0.19 means classical indices explain only ~2-3.5% of daily Ap variance. Consistent with known physics: F10.7/SN track active-region emission, not coronal-hole geometry, which drives high-speed-stream geomagnetic activity.

## Persistence baseline (Ap autocorrelation, SDO era)

| lag (days) | r |
|---|---|
| 1 | 0.461 |
| 2 | 0.173 |
| 3 | 0.100 |
| 4 | 0.081 |
| 7 | 0.065 |

Persistence dominates at 1-day lag, falls below the classical-index level by day 3-4. Defines the useful window for this project: 3-14 days, where persistence has decayed and classical indices are known-weak, is where a richer embedding has room to add value.

## Conclusion

No small-sample problem (n in the thousands to tens of thousands, vs. n=16 for the earlier NAO/tropospheric-teleconnection pilot, which was abandoned for this reason). Real, statistically robust, but small classical-index signal — leaves clear headroom. Proceed with baseline probes next.
