# AMT YAML Format Reference

The AMT (Asset Management Table) YAML file defines the complete configuration for a portfolio management system, including backtest parameters, rule tables, expiry schedules, and asset definitions.

---

## Top-Level Structure

```yaml
comments:          # List of comments/change notes
backtest:          # Backtest configuration parameters
asset_config:      # Asset configuration formulas
signal_table:      # Signal calculation definitions
group_table:       # Asset grouping rules
subgroup_table:    # Asset subgrouping rules
liquidity_table:   # Liquidity classification rules
limit_overrides:   # Per-asset limit overrides
limit_table:       # Default limit rules
schedule_table:    # Per-asset schedule assignments
expiry_schedules:  # Named expiry schedule definitions
amt:               # Asset definitions (the main section)
```

---

## Comments Section

A list of strings documenting changes and notes:

```yaml
comments:
   - "zero out USDCNH, EURCNH, USDTRY, USDTWD"
   - "fix USSWAP5 Curncy continuity"
   - "leverage to 20"
```

---

## Backtest Section

Configuration for backtesting and portfolio management:

```yaml
backtest:
  # Database schema
  schema_prefix: result
  schema_suffix: stagger1

  # Strategy weight toggles (with YAML anchors for reuse)
  strategy_toggles:
    stonks: &STONKS 0.05
    swaps_jpy: &SWAPS_JPY 0
    fx_usd_new: &FX_USD_NEW 0.025

  # Backtest parameters
  expiry_overrides: expiry_overrides.csv
  backtest_start: 2001-01-01

  # Portfolio limits
  aum: 800.0
  leverage: 20.0
  reference_asset: "SPX Index"
  reference_size: 150

  # Signal parameters
  signal_window: 365
  vol_factor: 1.0

  # Strategy calculations
  verbose: yes
  residual_accumulate:
    winsorize_input: yes
    winsorize_result: yes
    multi_threaded: yes
    calc:
      - pearson
      - tau
  signal_accumulate:
    rank: group_max
    scale_const: min
    multi_threaded: yes
    calc:
      - rsi
      - calmar
  # ... more calculation configs
```

**Key Fields:**
- `aum`: Assets under management (in millions)
- `leverage`: Portfolio leverage multiplier
- `reference_asset`: Benchmark asset for sizing
- `strategy_toggles`: Named weights that can be referenced with YAML anchors (`*STONKS`)

---

## Rule Tables

Rule tables use a common format with `[field, rgx, value]` rows:

```yaml
group_table:
    Columns: [field, rgx, value]
    Types:   [character, unique_character, character]
    Rows:
    - [Underlying, '^(LQD).*$', 'rates']
    - [Underlying, '^(HYG|EMB|IBOXUMAE|IBOXHYSE|ITRXEBE|ITRXEXE|ITRXESE).*$', 'rates']
    - [Class,      '^(Rate|Swap)$', 'rates']
    - [Class,      '^Equity$', 'equity']
    - [Class,      '^Currency$', 'fx']
    - [Class,      '^Commodity$', 'commodity']
    - [Class,      '^SingleStock$', 'stonks']
    - [Underlying, '^.*$', 'error']  # Default catch-all
```

**Table Types:**
- `group_table`: Assigns assets to groups (rates, equity, fx, commodity, stonks, credit)
- `subgroup_table`: Assigns assets to subgroups (EAST, WEST, Growth, Value, AA, UE, etc.)
- `subgroup_table_auto`: Auto-generated subgroup assignments (c0, c1, c2, etc.)
- `liquidity_table`: Assigns liquidity classifications
- `limit_overrides`: Per-asset limit overrides
- `limit_table`: Default limit values

**Related Config Tables:**
- `group_config`: Group configuration (hedge, positioning settings)
- `group_risk_multiplier_table`: Risk multipliers by group
- `group_limit_multiplier_table`: Limit multipliers by group

**Matching:**
- `field`: The asset field to match (`Underlying` or `Class`)
- `rgx`: Regular expression pattern (e.g., `^Equity$`, `^AAPL US Equity.*$`)
- `value`: The value to assign if matched
- Rules are evaluated in order; use `^.*$` as a catch-all default at the end

---

## Schedule Table

Maps specific assets to named schedules:

```yaml
schedule_table:
     Columns: [asset, name]
     Types:   [unique_character, character]
     Rows:
       - ['LA Comdty','schedule7']
       - ['SPX Index','schedule2']
       - ['VIX Index','schedule4']
```

---

## Expiry Schedules Section

Named lists of expiry schedule entries:

```yaml
expiry_schedules:
  schedule1:
      - N0_BDa_25
      - N0_BDb_25
      - N0_BDc_25
      - N0_BDd_25

  schedule2:
      - N0_F1_25
      - N0_F2_25
      - N0_F3_25
      - N0_F4_25

  schedule7:
      - N0_OVERRIDE_33.3
      - N5_OVERRIDE_33.3
      - F10_OVERRIDE_12.5
      - F15_OVERRIDE_12.5
```

### Schedule Entry Format

Each entry follows the pattern: `{entry}_{expiry}_{weight}`

**Entry Codes (when to enter):**
- `N0`, `N2`, `N5`, `N7`: Near month offset (business days)
- `F0`, `F10`, `F12`, `F15`: Far month offset (business days)

**Expiry Codes (when to exit):**
- `F1`, `F2`, `F3`, `F4`: Monthly futures expiry (1st, 2nd, 3rd, 4th month)
- `BDa`, `BDb`, `BDc`, `BDd`: Business day offsets
- `R2`, `R3`: Roll schedules (2nd, 3rd month)
- `W1`, `W2`, `W3`, `W4`: Weekly expiries
- `LBa`, `LBb`, `LBc`, `LBd`: Last business day schedules
- `OVERRIDE`: Uses asset-specific override dates

**Weight:**
- Percentage weight for this schedule entry (e.g., `25`, `33.3`, `12.5`)
- Weights within a schedule should sum to 100

**Examples:**
- `N0_F1_25`: Enter at near month, exit at 1st month futures expiry, 25% weight
- `F10_F3_12.5`: Enter 10 days into far month, exit at 3rd month expiry, 12.5% weight
- `N0_OVERRIDE_100`: Enter at near month, exit at override date, 100% weight

---

## AMT Section (Asset Definitions)

The main section containing all asset definitions:

```yaml
amt:
  AAPL US Equity:
    Underlying: "AAPL US Equity"
    Description: "APPLE INC"
    Class: SingleStock
    Market: {Field: PX_LAST, Tickers: AAPL US Equity}
    Vol: {Source: BBG, Ticker: AAPL US Equity, Near: "30DAY_IMPVOL_100.0%MNY_DF", Far: "60DAY_IMPVOL_100.0%MNY_DF"}
    Options: schedule4
    Hedge: {Source: nonfut, Ticker: AAPL US Equity, Field: PX_LAST}
    Slippage: 0.6
    SlippageFactor: 1
    WeightCap: 0.05
    RiskAdj: 1
    Valuation: {Model: ES, S: px, X: strike, t: expiry_date - date, v: vol}
```

### Asset Fields

| Field | Required | Description |
|-------|----------|-------------|
| `Underlying` | Yes | Unique identifier for the asset |
| `Description` | Yes | Human-readable description |
| `Class` | Yes | Asset class (see below) |
| `Market` | Yes | Market data source configuration |
| `Vol` | Yes | Volatility data source configuration |
| `Options` | Yes | Name of expiry schedule to use |
| `Hedge` | Yes | Hedging instrument configuration |
| `Slippage` | Yes | Slippage cost in basis points |
| `SlippageFactor` | No | Multiplier for slippage (default: 1) |
| `WeightCap` | Yes | Maximum weight (0.0 to 1.0), can use YAML anchor |
| `RiskAdj` | No | Risk adjustment factor (default: 1) |
| `Valuation` | Yes | Valuation model configuration |

### Class Types

| Class | Description | Examples |
|-------|-------------|----------|
| `Commodity` | Commodity futures | CL Comdty, GC Comdty, C Comdty |
| `Equity` | Equity indices and ETFs | SPX Index, EEM US Equity |
| `Currency` | FX pairs | EURUSD Curncy, USDJPY Curncy |
| `Rate` | Fixed income | HYG US Equity, RX Comdty |
| `SingleStock` | Individual stocks | AAPL US Equity, GOOG US Equity |
| `Swap` | Interest rate swaps | USSWAP10 Curncy, EUR5Y5Y |

---

## Market Configuration

Specifies where to get market price data:

```yaml
Market: {Field: PX_LAST, Tickers: AAPL US Equity}
```

- `Field`: The Bloomberg field to fetch (usually `PX_LAST`)
- `Tickers`: The ticker symbol(s) for market data

---

## Vol (Volatility) Configuration

Specifies where to get implied volatility data. There are three source types:

### BBG Source (Bloomberg)

```yaml
Vol: {Source: BBG, Ticker: AAPL US Equity, Near: "30DAY_IMPVOL_100.0%MNY_DF", Far: "60DAY_IMPVOL_100.0%MNY_DF"}
```

For commodity futures with 1st/2nd month vol:
```yaml
Vol: {Source: BBG, Ticker: CL1 Comdty, Near: "1ST_MTH_IMPVOL_100.0%MNY_DF", Far: "2ND_MTH_IMPVOL_100.0%MNY_DF"}
```

For FX with composite tickers:
```yaml
Vol: {Source: BBG, Ticker: USDCADV1M CMPN Curncy, Near: PX_LAST, Far: PX_LAST}
```

- `Ticker`: Bloomberg ticker for vol data (supports transition format, see below)
- `Near`: Field for near-term vol
- `Far`: Field for far-term vol

**Transition Ticker Format:**

All Bloomberg tickers support a transition format for handling ticker changes over time:

```yaml
Ticker: "USSWAP5 CMPN Curncy:2023-06:USOSFR5 Curncy"
```

Format: `ticker1:YYYY-MM:ticker2`
- `ticker1` is valid **before** the transition date
- `ticker2` is valid **after** the transition date

When expanded, this creates two ticker entries with date constraints that are filtered based on the straddle's expiry date.

### CV Source (CitiVelocity)

For volatility data from CitiVelocity (Citi's data platform), typically used for swaption vols:

```yaml
Vol: {Source: CV, Near: RATES.VOL.USD.ATM.NORMAL.ANNUAL.1M.10Y, Far: NONE}
```

- `Near`: CitiVelocity ticker (stored in the field column when fetched, with empty ticker)
- `Far`: Usually `NONE` for CV source

### BBG_LMEVOL Source

For LME metals with tickerized vol data:

```yaml
Vol: {Source: BBG_LMEVOL}
```

No additional fields required; vol tickers are constructed programmatically.

---

## Hedge Configuration

Specifies the hedging instrument. There are four source types:

### fut (Futures)

For assets hedged with exchange-traded futures:

```yaml
Hedge: {Source: fut, generic: "CL1 Comdty", fut_code: CL, fut_month_map: GHJKMNQUVXZF, min_year_offset: 0, market_code: Comdty}
```

| Field | Description |
|-------|-------------|
| `generic` | Generic ticker for the front contract |
| `fut_code` | Root code for specific contracts |
| `fut_month_map` | 12-character map of month codes (Jan-Dec) |
| `min_year_offset` | Minimum year offset for contract selection |
| `market_code` | Bloomberg market code (Comdty, Index, etc.) |

**Month Map Characters:**
Each character maps to a month (F=Jan, G=Feb, H=Mar, J=Apr, K=May, M=Jun, N=Jul, Q=Aug, U=Sep, V=Oct, X=Nov, Z=Dec).
Repeated characters indicate non-standard delivery months.

### nonfut (Non-Futures)

For assets hedged with the underlying instrument:

```yaml
Hedge: {Source: nonfut, Ticker: AAPL US Equity, Field: PX_LAST}
```

Or with the generic field:
```yaml
Hedge: {Source: nonfut, generic: "SPY US Equity", Ticker: SPY US Equity, Field: PX_LAST}
```

| Field | Description |
|-------|-------------|
| `Ticker` | Bloomberg ticker for the hedge instrument |
| `Field` | Price field (usually `PX_LAST`) |
| `generic` | Optional generic name |

### cds (Credit Default Swaps)

For credit index options hedged with CDS:

```yaml
Hedge: {Source: cds, generic: "ITRXEBE Curncy", hedge: ITRXEBE CBBT Curncy, hedge1: EUSA5 Curncy}
```

With transition ticker (see transition format above):
```yaml
Hedge: {Source: cds, generic: "IBOXUMAE CBBT Curncy", hedge: IBOXUMAE CBBT Curncy, hedge1: "USSWAP5 CMPN Curncy:2023-06:USOSFR5 Curncy"}
```

| Field | Description |
|-------|-------------|
| `generic` | Generic name for the hedge |
| `hedge` | Primary CDS ticker (supports transition format) |
| `hedge1` | Secondary hedge (rate hedge, supports transition format) |

### calc (Calculated)

For assets with computed hedge positions (e.g., swaps):

```yaml
Hedge: {Source: calc, generic: USSWAP10 CMPN Curncy, ccy: USD, tenor: 10}
```

For forward-starting swaps:
```yaml
Hedge: {Source: calc, generic: USSWAP5 CMPN Curncy, ccy: USD, tenor: "2_2"}
```

| Field | Description |
|-------|-------------|
| `generic` | Generic name for reference |
| `ccy` | Currency code (USD, EUR, JPY, GBP, sUSD for SOFR) |
| `tenor` | Swap tenor (2, 5, 10, 20, 30) or forward format ("1_1", "2_2", "5_5") |

**Generated Tickers:**
The `calc` source generates 4 hedge tickers automatically:
- `{ccy}_fsw0m_{tenor}`: Forward swap 0m
- `{ccy}_fsw6m_{tenor}`: Forward swap 6m
- `{ccy}_pva0m_{tenor}`: PV01 annuity 0m
- `{ccy}_pva6m_{tenor}`: PV01 annuity 6m

---

## Valuation Configuration

Specifies the option pricing model:

```yaml
Valuation: {Model: ES, S: px, X: strike, t: expiry_date - date, v: vol}
```

For swaps with normal vol:
```yaml
Valuation: {Model: NS, S: fsw, X: strike, t: expiry_date - date, v: vol/10000, pva: pva}
```

For CDS options:
```yaml
Valuation: {Model: CDS_ES, tenor: 5, S: px, X: strike, t: expiry_date - date, v: vol}
```

| Model | Description | Key Fields |
|-------|-------------|------------|
| `ES` | European straddle (Black-Scholes) | S, X, t, v |
| `NS` | Normal straddle (Bachelier) | S, X, t, v, pva |
| `CDS_ES` | CDS European straddle | tenor, S, X, t, v |

---

## YAML Anchors

The file uses YAML anchors for reusable values:

**Define an anchor:**
```yaml
strategy_toggles:
  stonks: &STONKS 0.05
```

**Reference the anchor:**
```yaml
WeightCap: *STONKS
```

This allows changing all stonk weights in one place.

---

## Complete Asset Examples

### Commodity Future

```yaml
CL Comdty:
  Underlying: "CL Comdty"
  Description: "WTI CRUDE FUTURE"
  Class: Commodity
  Market: {Field: PX_LAST, Tickers: CL1 Comdty}
  Vol: {Source: BBG, Ticker: CL1 Comdty, Near: "1ST_MTH_IMPVOL_100.0%MNY_DF", Far: "2ND_MTH_IMPVOL_100.0%MNY_DF"}
  Options: schedule7
  Hedge: {Source: fut, generic: CL1 Comdty, fut_code: CL, fut_month_map: GHJKMNQUVXZF, min_year_offset: 0, market_code: Comdty}
  Slippage: 0.9
  SlippageFactor: 1
  WeightCap: 0.05
  RiskAdj: 1
  Valuation: {Model: ES, S: px, X: strike, t: expiry_date - date, v: vol}
```

### Currency Pair

```yaml
EURUSD Curncy:
  Underlying: "EURUSD Curncy"
  Description: "EUR-USD X-RATE"
  Class: Currency
  Market: {Field: PX_LAST, Tickers: EURUSD Curncy}
  Vol: {Source: BBG, Ticker: EURUSDV1M CMPN Curncy, Near: PX_LAST, Far: PX_LAST}
  Options: schedule1
  Hedge: {Source: nonfut, Ticker: EURUSD Curncy, Field: PX_LAST}
  Slippage: 0.1
  SlippageFactor: 1
  WeightCap: 0.05
  RiskAdj: 1
  Valuation: {Model: ES, S: px, X: strike, t: expiry_date - date, v: vol}
```

### Interest Rate Swap

```yaml
USSWAP10 Curncy:
  Underlying: "USSWAP10 Curncy"
  Description: "USD 10Y Swap"
  Class: Swap
  Market: {Field: PX_LAST, Tickers: USSWAP10 CMPN Curncy}
  Vol: {Source: CV, Near: RATES.VOL.USD.ATM.NORMAL.ANNUAL.1M.10Y, Far: NONE}
  Options: schedule1
  Hedge: {Source: calc, generic: USSWAP10 CMPN Curncy, ccy: USD, tenor: 10}
  Slippage: 0.125
  SlippageFactor: 1
  WeightCap: 0.05
  RiskAdj: 1
  Valuation: {Model: NS, S: fsw, X: strike, t: expiry_date - date, v: vol/10000, pva: pva}
```

### Single Stock

```yaml
AAPL US Equity:
  Underlying: "AAPL US Equity"
  Description: "APPLE INC"
  Class: SingleStock
  Market: {Field: PX_LAST, Tickers: AAPL US Equity}
  Vol: {Source: BBG, Ticker: AAPL US Equity, Near: "30DAY_IMPVOL_100.0%MNY_DF", Far: "60DAY_IMPVOL_100.0%MNY_DF"}
  Options: schedule4
  Hedge: {Source: nonfut, Ticker: AAPL US Equity, Field: PX_LAST}
  Slippage: 0.6
  SlippageFactor: 1
  WeightCap: *STONKS
  RiskAdj: 1
  Valuation: {Model: ES, S: px, X: strike, t: expiry_date - date, v: vol}
```

### Credit Index

```yaml
ITRXEBE Curncy:
  Underlying: "ITRXEBE Curncy"
  Description: "MARKIT ITRAXX EUROPE INDEX"
  Class: Rate
  Market: {Field: PX_LAST, Tickers: ITRXEBE CBBT Curncy}
  Vol: {Source: CV, Near: CREDIT.VOL.ITRAXX.EUROPE.5Y.ATM, Far: NONE}
  Options: schedule11
  Hedge: {Source: cds, generic: "ITRXEBE Curncy", hedge: ITRXEBE CBBT Curncy, hedge1: EUSA5 Curncy}
  Valuation: {Model: CDS_ES, tenor: 5, S: px, X: strike, t: expiry_date - date, v: vol}
  Slippage: 0.125
  SlippageFactor: 1
  WeightCap: 0.025
  RiskAdj: 1
```
