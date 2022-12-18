# Benchmarks

## Table of Contents

- [Benchmarks](#benchmarks)
  - [Table of Contents](#table-of-contents)
  - [Benchmark Results](#benchmark-results)
    - [small\_data](#small_data)
    - [large\_data](#large_data)

## Benchmark Results

### small_data 

|                             | ` residuals_from_front ` | ` all_residuals `               | ` all_residuals_par `          |
| :-------------------------- | :----------------------- | :------------------------------ | :----------------------------- |
| **` data_len=10, deg<=6`**  | `1.43 us` (✅ **1.00x**)  | `7.60 us` (❌ *5.31x slower*)    | `22.53 us` (❌ *15.73x slower*) |
| **` data_len=20, deg<=6`**  | `2.99 us` (✅ **1.00x**)  | `30.32 us` (❌ *10.13x slower*)  | `18.28 us` (❌ *6.11x slower*)  |
| **` data_len=30, deg<=6`**  | `4.52 us` (✅ **1.00x**)  | `71.37 us` (❌ *15.77x slower*)  | `21.13 us` (❌ *4.67x slower*)  |
| **` data_len=40, deg<=6`**  | `6.08 us` (✅ **1.00x**)  | `125.85 us` (❌ *20.69x slower*) | `28.76 us` (❌ *4.73x slower*)  |
| **` data_len=50, deg<=6`**  | `12.97 us` (✅ **1.00x**) | `205.75 us` (❌ *15.86x slower*) | `36.74 us` (❌ *2.83x slower*)  |
| **` data_len=60, deg<=6`**  | `9.34 us` (✅ **1.00x**)  | `271.95 us` (❌ *29.12x slower*) | `46.51 us` (❌ *4.98x slower*)  |
| **` data_len=70, deg<=6`**  | `10.50 us` (✅ **1.00x**) | `369.41 us` (❌ *35.20x slower*) | `55.04 us` (❌ *5.24x slower*)  |
| **` data_len=80, deg<=6`**  | `11.92 us` (✅ **1.00x**) | `478.52 us` (❌ *40.15x slower*) | `70.59 us` (❌ *5.92x slower*)  |
| **` data_len=90, deg<=6`**  | `13.36 us` (✅ **1.00x**) | `608.75 us` (❌ *45.55x slower*) | `78.37 us` (❌ *5.86x slower*)  |
| **` data_len=100, deg<=6`** | `14.88 us` (✅ **1.00x**) | `756.68 us` (❌ *50.87x slower*) | `91.33 us` (❌ *6.14x slower*)  |

### large_data 

|                                | ` residuals_from_front `  | ` all_residuals `                | ` all_residuals_par `            |
| :----------------------------- | :------------------------ | :------------------------------- | :------------------------------- |
| **` data_len=100, deg<=10`**   | `23.59 us` (✅ **1.00x**)  | `1.16 ms` (❌ *49.00x slower*)    | `129.26 us` (❌ *5.48x slower*)   |
| **` data_len=500, deg<=10`**   | `115.93 us` (✅ **1.00x**) | `28.98 ms` (❌ *250.00x slower*)  | `2.03 ms` (❌ *17.51x slower*)    |
| **` data_len=1000, deg<=10`**  | `231.76 us` (✅ **1.00x**) | `117.74 ms` (❌ *508.01x slower*) | `8.13 ms` (❌ *35.10x slower*)    |
| **` data_len=5000, deg<=10`**  | `1.16 ms` (✅ **1.00x**)   | `3.06 s` (❌ *2648.72x slower*)   | `227.15 ms` (❌ *196.49x slower*) |
| **` data_len=1000, deg<=100`** | `4.46 ms` (✅ **1.00x**)   | `2.17 s` (❌ *487.03x slower*)    | `176.49 ms` (❌ *39.59x slower*)  |

---
Made with [criterion-table](https://github.com/nu11ptr/criterion-table)

