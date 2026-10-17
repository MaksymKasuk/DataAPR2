import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

warnings.filterwarnings("ignore")

# ============================================================
# Практична робота № 2
# Тема: Прогнозування часових рядів з використанням моделей ARIMA
# Дослідження: Аналіз якості машинного перекладу тексту
# ============================================================

OUTPUT_DIR = Path("arima_translation_quality_results")
OUTPUT_DIR.mkdir(exist_ok=True)

ALPHA = 0.05
np.random.seed(42)

# ------------------------------------------------------------
# 1. Формування часового ряду інтегральної якості перекладу
# ------------------------------------------------------------
# Інтегральна якість формується за чотирма критеріями:
# 35% semantic + 25% grammar + 20% lexical + 20% n-gram.
# Шкала кожного критерію: 0-100 балів.

dates = pd.date_range("2023-01-01", periods=48, freq="MS")
t = np.arange(len(dates))

common_season = 1.6 * np.sin(2 * np.pi * t / 12)

semantic = np.clip(
    69 + 0.34 * t + common_season + np.random.normal(0, 1.1, len(t)),
    0, 100,
)
grammar = np.clip(
    74 + 0.25 * t
    + 0.9 * np.sin(2 * np.pi * t / 12 + 0.5)
    + np.random.normal(0, 1.0, len(t)),
    0, 100,
)
lexical = np.clip(
    67 + 0.31 * t
    + 1.2 * np.sin(2 * np.pi * t / 12 + 0.2)
    + np.random.normal(0, 1.15, len(t)),
    0, 100,
)
ngram_score = np.clip(
    63 + 0.36 * t
    + 1.8 * np.sin(2 * np.pi * t / 12 - 0.2)
    + np.random.normal(0, 1.25, len(t)),
    0, 100,
)

overall_quality = (
    0.35 * semantic
    + 0.25 * grammar
    + 0.20 * lexical
    + 0.20 * ngram_score
)
overall_quality += np.random.normal(0, 0.35, len(t))
overall_quality = np.clip(overall_quality, 0, 100)

series = pd.Series(
    np.round(overall_quality, 2),
    index=dates,
    name="overall_quality",
)

base_df = pd.DataFrame({
    "date": dates,
    "semantic_score": np.round(semantic, 2),
    "grammar_score": np.round(grammar, 2),
    "lexical_score": np.round(lexical, 2),
    "ngram_score": np.round(ngram_score, 2),
    "overall_quality": series.values,
})
base_df.to_csv(OUTPUT_DIR / "translation_quality_timeseries.csv", index=False, encoding="utf-8-sig")

print("=" * 78)
print("ПРАКТИЧНА РОБОТА № 2: ARIMA")
print("=" * 78)
print("\nПерші 12 значень часового ряду:")
print(series.head(12).to_string())

# Графік вихідного ряду
plt.figure(figsize=(12, 6))
plt.plot(series.index, series.values, marker="o", linewidth=1.5)
plt.title("Часовий ряд інтегральної якості машинного перекладу")
plt.xlabel("Період тестування")
plt.ylabel("Quality Score, балів (0-100)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_initial_time_series.png", dpi=180, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# 2. Перевірка стаціонарності вихідного ряду
# ------------------------------------------------------------

def adf_result(values):
    result = adfuller(values, autolag="AIC")
    return {
        "statistic": float(result[0]),
        "p_value": float(result[1]),
        "used_lag": int(result[2]),
        "nobs": int(result[3]),
        "critical_1%": float(result[4]["1%"]),
        "critical_5%": float(result[4]["5%"]),
        "critical_10%": float(result[4]["10%"]),
    }


def kpss_result(values):
    result = kpss(values, regression="c", nlags="auto")
    return {
        "statistic": float(result[0]),
        "p_value": float(result[1]),
        "used_lag": int(result[2]),
        "critical_10%": float(result[3]["10%"]),
        "critical_5%": float(result[3]["5%"]),
        "critical_2.5%": float(result[3]["2.5%"]),
        "critical_1%": float(result[3]["1%"]),
    }


adf_original = adf_result(series)
kpss_original = kpss_result(series)

print("\nПЕРЕВІРКА СТАЦІОНАРНОСТІ ВИХІДНОГО РЯДУ")
print("ADF H0: ряд має одиничний корінь (нестаціонарний)")
print(f"ADF statistic = {adf_original['statistic']:.4f}")
print(f"ADF p-value   = {adf_original['p_value']:.6f}")
print("KPSS H0: ряд є стаціонарним")
print(f"KPSS statistic = {kpss_original['statistic']:.4f}")
print(f"KPSS p-value   = {kpss_original['p_value']:.4f}")

rolling_mean = series.rolling(window=12).mean()
rolling_std = series.rolling(window=12).std()

plt.figure(figsize=(12, 6))
plt.plot(series.index, series, label="Вихідний ряд", linewidth=1.4)
plt.plot(rolling_mean.index, rolling_mean, label="Ковзне середнє (12)", linewidth=2)
plt.plot(rolling_std.index, rolling_std, label="Ковзне стандартне відхилення (12)", linewidth=1.5)
plt.title("Візуальна перевірка стаціонарності вихідного ряду")
plt.xlabel("Період тестування")
plt.ylabel("Значення")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_stationarity_original.png", dpi=180, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# 3. Приведення ряду до стаціонарного вигляду
# ------------------------------------------------------------
# Перша різниця використовується, якщо вихідний ряд нестаціонарний.
first_difference = series.diff().dropna()

adf_diff = adf_result(first_difference)
kpss_diff = kpss_result(first_difference)

print("\nПЕРША РІЗНИЦЯ")
print(f"ADF statistic = {adf_diff['statistic']:.4f}")
print(f"ADF p-value   = {adf_diff['p_value']:.8f}")
print(f"KPSS statistic = {kpss_diff['statistic']:.4f}")
print(f"KPSS p-value   = {kpss_diff['p_value']:.4f}")

# d визначається за результатом тестів.
if adf_original["p_value"] < ALPHA and kpss_original["p_value"] > ALPHA:
    d = 0
elif adf_diff["p_value"] < ALPHA and kpss_diff["p_value"] > ALPHA:
    d = 1
else:
    second_difference = first_difference.diff().dropna()
    adf_diff2 = adf_result(second_difference)
    kpss_diff2 = kpss_result(second_difference)
    d = 2 if adf_diff2["p_value"] < ALPHA and kpss_diff2["p_value"] > ALPHA else 1

print(f"Порядок інтегрування ARIMA: d = {d}")

plt.figure(figsize=(12, 6))
plt.plot(first_difference.index, first_difference.values, marker="o", linewidth=1.3)
plt.axhline(0, linewidth=1)
plt.title("Перша різниця інтегральної якості перекладу")
plt.xlabel("Період тестування")
plt.ylabel("Δ Quality Score")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_first_difference.png", dpi=180, bbox_inches="tight")
plt.show()

# ACF та PACF стаціонарного ряду - попередня оцінка p та q.
plt.figure(figsize=(10, 5))
plot_acf(first_difference, lags=18, alpha=0.05, zero=False)
plt.title("ACF першої різниці")
plt.xlabel("Лаг")
plt.ylabel("Автокореляція")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_acf_first_difference.png", dpi=180, bbox_inches="tight")
plt.show()

plt.figure(figsize=(10, 5))
plot_pacf(first_difference, lags=18, alpha=0.05, zero=False, method="ywm")
plt.title("PACF першої різниці")
plt.xlabel("Лаг")
plt.ylabel("Часткова автокореляція")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_pacf_first_difference.png", dpi=180, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# 4. Підбір моделі ARIMA(p,d,q)
# ------------------------------------------------------------
# p і q перебираються в діапазоні 0..5.
# Серед моделей, що збіглися і пройшли Ljung-Box на рівні 0.05,
# обирається модель з найменшим AIC.

candidate_rows = []
fitted_models = {}

for p in range(0, 6):
    for q in range(0, 6):
        try:
            # Для d>0 лінійний trend='t' у рівнях відповідає дрейфу
            # після диференціювання.
            trend = "t" if d > 0 else "ct"
            model = ARIMA(series, order=(p, d, q), trend=trend)
            fitted = model.fit(method_kwargs={"maxiter": 500})

            converged = bool(fitted.mle_retvals.get("converged", True))
            standardized_residuals = fitted.filter_results.standardized_forecasts_error[0]
            standardized_residuals = standardized_residuals[fitted.loglikelihood_burn:]

            model_df = p + q
            if 12 > model_df:
                lb = acorr_ljungbox(
                    standardized_residuals,
                    lags=[12],
                    model_df=model_df,
                    return_df=True,
                )
                lb_p = float(lb["lb_pvalue"].iloc[0])
            else:
                lb_p = np.nan

            row = {
                "p": p,
                "d": d,
                "q": q,
                "AIC": float(fitted.aic),
                "BIC": float(fitted.bic),
                "Ljung_Box_p12": lb_p,
                "converged": converged,
            }
            candidate_rows.append(row)
            fitted_models[(p, d, q)] = fitted
        except Exception:
            continue

candidates = pd.DataFrame(candidate_rows).sort_values("AIC").reset_index(drop=True)
candidates.to_csv(OUTPUT_DIR / "arima_model_candidates.csv", index=False, encoding="utf-8-sig")

adequate = candidates[
    (candidates["converged"] == True)
    & (candidates["Ljung_Box_p12"] >= ALPHA)
]

if len(adequate) > 0:
    best_row = adequate.iloc[0]
else:
    best_row = candidates[candidates["converged"] == True].iloc[0]

best_order = (int(best_row["p"]), int(best_row["d"]), int(best_row["q"]))
best_model = fitted_models[best_order]

print("\nПІДБІР ARIMA")
print("Топ-10 моделей за AIC:")
print(candidates.head(10).round(4).to_string(index=False))
print(f"\nОбрана модель: ARIMA{best_order}")
print(f"AIC = {best_model.aic:.3f}")
print(f"BIC = {best_model.bic:.3f}")

# Графік порівняння AIC для найкращих кандидатів.
top_models = candidates.head(10).copy()
top_models["model"] = top_models.apply(
    lambda row: f"ARIMA({int(row.p)},{int(row.d)},{int(row.q)})", axis=1
)
plt.figure(figsize=(11, 6))
plt.barh(top_models["model"][::-1], top_models["AIC"][::-1])
plt.title("Порівняння кандидатів ARIMA за критерієм AIC")
plt.xlabel("AIC (менше - краще)")
plt.ylabel("Модель")
plt.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_arima_model_selection.png", dpi=180, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# 5. Параметри обраної моделі
# ------------------------------------------------------------

params_df = pd.DataFrame({
    "parameter": best_model.param_names,
    "estimate": best_model.params,
    "std_error": best_model.bse,
    "p_value": best_model.pvalues,
})
params_df.to_csv(OUTPUT_DIR / "arima_parameters.csv", index=False, encoding="utf-8-sig")

print("\nПАРАМЕТРИ ОБРАНОЇ МОДЕЛІ")
print(params_df.round(6).to_string(index=False))

# Порівняння фактичних і підігнаних значень.
fitted_values = best_model.predict(start=series.index[1], end=series.index[-1])
plt.figure(figsize=(12, 6))
plt.plot(series.index, series, label="Фактичні значення", linewidth=1.5)
plt.plot(fitted_values.index, fitted_values, label=f"ARIMA{best_order}: підігнані значення", linewidth=1.5)
plt.title("Фактичні та підігнані значення")
plt.xlabel("Період тестування")
plt.ylabel("Quality Score")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "07_fitted_vs_actual.png", dpi=180, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# 6. Перевірка адекватності моделі
# ------------------------------------------------------------
standardized_residuals = best_model.filter_results.standardized_forecasts_error[0]
standardized_residuals = standardized_residuals[best_model.loglikelihood_burn:]

lb = acorr_ljungbox(
    standardized_residuals,
    lags=[12],
    model_df=best_order[0] + best_order[2],
    return_df=True,
)
lb_stat = float(lb["lb_stat"].iloc[0])
lb_p = float(lb["lb_pvalue"].iloc[0])

shapiro_stat, shapiro_p = shapiro(standardized_residuals)
arch_lm_stat, arch_lm_p, arch_f_stat, arch_f_p = het_arch(standardized_residuals, nlags=6)

min_ar_root = float(np.min(np.abs(best_model.arroots))) if len(best_model.arroots) else np.inf
min_ma_root = float(np.min(np.abs(best_model.maroots))) if len(best_model.maroots) else np.inf

print("\nПЕРЕВІРКА АДЕКВАТНОСТІ")
print("H0 Ljung-Box: залишки не мають автокореляції")
print(f"Ljung-Box Q(12) = {lb_stat:.4f}, p = {lb_p:.6f}")
print(f"Shapiro-Wilk: W = {shapiro_stat:.4f}, p = {shapiro_p:.6f}")
print(f"ARCH-LM(6): statistic = {arch_lm_stat:.4f}, p = {arch_lm_p:.6f}")
print(f"Мінімальний модуль AR-кореня = {min_ar_root:.4f}")
print(f"Мінімальний модуль MA-кореня = {min_ma_root:.4f}")
print(f"Модель збіглася: {best_model.mle_retvals.get('converged', True)}")

adequate_model = (
    best_model.mle_retvals.get("converged", True)
    and lb_p >= ALPHA
    and min_ar_root > 1
    and min_ma_root > 1
)

print("Висновок:", "модель є адекватною" if adequate_model else "модель потребує уточнення")

plt.figure(figsize=(12, 5))
plt.plot(np.arange(1, len(standardized_residuals) + 1), standardized_residuals, marker="o", linewidth=1)
plt.axhline(0, linewidth=1)
plt.title(f"Стандартизовані залишки ARIMA{best_order}")
plt.xlabel("Номер спостереження")
plt.ylabel("Стандартизований залишок")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "08_arima_residuals.png", dpi=180, bbox_inches="tight")
plt.show()

plt.figure(figsize=(10, 5))
plot_acf(standardized_residuals, lags=18, alpha=0.05, zero=False)
plt.title(f"ACF залишків ARIMA{best_order}")
plt.xlabel("Лаг")
plt.ylabel("Автокореляція")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "09_residual_acf.png", dpi=180, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# 7. Прогноз на 10 періодів уперед
# ------------------------------------------------------------
if adequate_model:
    forecast_result = best_model.get_forecast(steps=10)
    forecast_mean = forecast_result.predicted_mean
    forecast_ci = forecast_result.conf_int(alpha=0.05)

    forecast_df = pd.DataFrame({
        "date": forecast_mean.index,
        "forecast": forecast_mean.values,
        "lower_95": forecast_ci.iloc[:, 0].values,
        "upper_95": forecast_ci.iloc[:, 1].values,
    })
    forecast_df[["forecast", "lower_95", "upper_95"]] = forecast_df[[
        "forecast", "lower_95", "upper_95"
    ]].round(3)
    forecast_df.to_csv(OUTPUT_DIR / "forecast_10_periods.csv", index=False, encoding="utf-8-sig")

    print("\nПРОГНОЗ НА 10 ПЕРІОДІВ")
    print(forecast_df.to_string(index=False))

    plt.figure(figsize=(12, 6))
    plt.plot(series.index, series, label="Фактичні значення", linewidth=1.5)
    plt.plot(forecast_mean.index, forecast_mean, marker="o", label="Прогноз ARIMA", linewidth=1.7)
    plt.fill_between(
        forecast_mean.index,
        forecast_ci.iloc[:, 0].values,
        forecast_ci.iloc[:, 1].values,
        alpha=0.2,
        label="95% довірчий інтервал",
    )
    plt.title(f"Прогноз інтегральної якості на 10 періодів: ARIMA{best_order}")
    plt.xlabel("Період")
    plt.ylabel("Quality Score, балів (0-100)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "10_forecast_10_periods.png", dpi=180, bbox_inches="tight")
    plt.show()
else:
    forecast_df = pd.DataFrame()
    print("\nПрогноз не виконується, оскільки модель не пройшла перевірку адекватності.")

# ------------------------------------------------------------
# 8. Підсумковий файл із діагностикою
# ------------------------------------------------------------
summary = pd.DataFrame([
    {"indicator": "ADF original statistic", "value": adf_original["statistic"]},
    {"indicator": "ADF original p-value", "value": adf_original["p_value"]},
    {"indicator": "KPSS original statistic", "value": kpss_original["statistic"]},
    {"indicator": "KPSS original p-value", "value": kpss_original["p_value"]},
    {"indicator": "ADF first difference statistic", "value": adf_diff["statistic"]},
    {"indicator": "ADF first difference p-value", "value": adf_diff["p_value"]},
    {"indicator": "KPSS first difference statistic", "value": kpss_diff["statistic"]},
    {"indicator": "KPSS first difference p-value", "value": kpss_diff["p_value"]},
    {"indicator": "ARIMA p", "value": best_order[0]},
    {"indicator": "ARIMA d", "value": best_order[1]},
    {"indicator": "ARIMA q", "value": best_order[2]},
    {"indicator": "AIC", "value": best_model.aic},
    {"indicator": "BIC", "value": best_model.bic},
    {"indicator": "Ljung-Box Q(12)", "value": lb_stat},
    {"indicator": "Ljung-Box p-value", "value": lb_p},
    {"indicator": "Shapiro-Wilk p-value", "value": shapiro_p},
    {"indicator": "ARCH-LM p-value", "value": arch_lm_p},
    {"indicator": "min AR root modulus", "value": min_ar_root},
    {"indicator": "min MA root modulus", "value": min_ma_root},
])
summary.to_csv(OUTPUT_DIR / "analysis_summary.csv", index=False, encoding="utf-8-sig")

print("\n" + "=" * 78)
print("ПІДСУМКОВИЙ ВИСНОВОК")
print("=" * 78)
print(
    f"Вихідний ряд нестаціонарний; після першого диференціювання він стає "
    f"стаціонарним, тому d={d}. За AIC та діагностикою залишків обрано "
    f"ARIMA{best_order}. Ljung-Box p={lb_p:.4f}. "
    + ("Модель адекватна, прогноз на 10 періодів побудовано." if adequate_model else "Модель потребує уточнення.")
)
print(f"Результати збережено у папці: {OUTPUT_DIR.resolve()}")
