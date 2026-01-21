# Entry and Exit Date Convention

## Straddle Format

A straddle string has the format: `|ntry_month|xpry_month|ntrc|ntrv|xprc|xprv|mult|`

Example: `|2024-01|2024-03|F|10|F|3|12.5|`
- `ntry_month` = 2024-01 (entry month)
- `xpry_month` = 2024-03 (expiry month)
- `ntrc` = F (entry anchor weekday code)
- `ntrv` = 10 (entry calendar day offset from anchor)
- `xprc` = F (expiry anchor weekday code)
- `xprv` = 3 (expiry Nth weekday occurrence)
- `mult` = 12.5 (multiplier)

## Weekday Codes

- `F` = Friday
- `R` = Thursday
- `W` = Wednesday
- `BD` = Business day (different rules, see below)

## Good Day Definition

A "good day" is a row where:
- `vol` is not "none"
- All `hedge` columns are not "none"

## Expiry Date (xpry action)

For F/R/W codes:

1. **Compute anchor date**: The Nth occurrence of the specified weekday in the expiry month
   - Example: `F|3` = 3rd Friday of the expiry month

2. **Find action date**: First good day at or after the anchor date, within the expiry month

3. **Mark action**: The found date gets the "xpry" action

## Entry Date (ntry action)

For F/R/W codes:

1. **Compute anchor date**: The Nth occurrence of the specified weekday in the **entry** month
   - Uses `xprc` (weekday code) and `xprv` (Nth occurrence) applied to entry month
   - Example: If `xprc=F` and `xprv=3`, anchor = 3rd Friday of entry month

2. **Add calendar day offset**: Add `ntrv` calendar days to the anchor date
   - Example: If anchor = 2024-01-19 and `ntrv=10`, target = 2024-01-29

3. **Find action date**: First good day at or after the target date
   - If target is past end of entry month, use last good day of entry month
   - If no good day exists at or after target (within entry month), use last good day of entry month

4. **Mark action**: The found date gets the "ntry" action

5. **Edge case**: If no good day exists in the entry month at all, there is no "ntry" action

## BD Rules (Business Day)

For BD code, the anchor is the Nth business day of the month (where N = `xprv`).

### Entry (ntry)
1. **Compute anchor date**: The `xprv`th business day of entry month
   - Example: `BD|10` = 10th business day of entry month
2. **Add calendar day offset**: Add `ntrv` calendar days to the anchor date
3. **Find action date**: First good day at or after target date
   - If target is past end of entry month, use last good day of entry month
   - If no good day exists at or after target (within entry month), use last good day of entry month
4. **Mark action**: The found date gets the "ntry" action

### Expiry (xpry)
1. **Compute anchor date**: The `xprv`th business day of expiry month
2. **Find action date**: First good day at or after the anchor date, within the expiry month
3. **Mark action**: The found date gets the "xpry" action

Note: Business days are calendar days that are not weekends (Mon-Fri). This is different from "good days" which require market data to exist.

## Examples

### Example 1: Normal Entry

Straddle: `|2024-01|2024-03|F|10|F|3|12.5|`

Entry calculation:
1. Anchor = 3rd Friday of January 2024 = 2024-01-19
2. Target = 2024-01-19 + 10 days = 2024-01-29
3. Find first good day at or after 2024-01-29, within January 2024
4. If 2024-01-29 is good → ntry = 2024-01-29
5. If 2024-01-29 is not good, check 2024-01-30, 2024-01-31...
6. If none found, use last good day of January 2024

### Example 2: Entry Rolls to Month End

Straddle: `|2024-01|2024-03|F|20|F|3|12.5|`

Entry calculation:
1. Anchor = 3rd Friday of January 2024 = 2024-01-19
2. Target = 2024-01-19 + 20 days = 2024-02-08 (past January)
3. Since target > end of entry month, use last good day of January 2024

### Example 3: Weekend Target

Straddle: `|2024-01|2024-03|F|1|F|3|12.5|`

Entry calculation:
1. Anchor = 3rd Friday of January 2024 = 2024-01-19
2. Target = 2024-01-19 + 1 day = 2024-01-20 (Saturday)
3. First good day at or after 2024-01-20 within January
4. Likely 2024-01-22 (Monday) if it's a good day

## Summary Table

| Code | Anchor Calculation | Entry Date | Expiry Date |
|------|-------------------|------------|-------------|
| F | Nth Friday | anchor + ntrv days, first good day | anchor, first good day |
| R | Nth Thursday | anchor + ntrv days, first good day | anchor, first good day |
| W | Nth Wednesday | anchor + ntrv days, first good day | anchor, first good day |
| BD | Nth business day (Mon-Fri) | anchor + ntrv days, first good day | anchor, first good day |

## Notes

- Entry anchor uses expiry weekday code (`xprc`) and occurrence (`xprv`), applied to entry month
- Entry offset (`ntrv`) is in **calendar days**, not good days
- If computation goes past entry month, fallback to last good day of entry month
- Expiry has no calendar day offset - it's just the first good day at or after the anchor
