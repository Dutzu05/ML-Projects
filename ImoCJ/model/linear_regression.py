"""
Multiple linear regression for apartment rents in Cluj-Napoca.

Follows the same steps as the System Identification lab (MATLAB, lab2):
    1. identification data           -> look at the data
    2. build PHI, solve theta         -> PHI \\ Y  (here: np.linalg.lstsq)
    3. validate with the same theta   -> MSE_id vs MSE_val, plots
    4. tuning                         -> MSE(n) as the number of regressors n grows

The difference from the lab: instead of ONE input x and its powers
[1, x, x^2, ...], we have MANY inputs (area, rooms, floor, ...), so

    price = theta_0 * 1 + theta_1 * area_m2 + theta_2 * rooms + ... + theta_22 * is_agency
    PHI   = [ 1  area_m2  rooms  floor  ...  is_agency ]   (one row per apartment)

Run from the ImoCJ folder:
    python model/linear_regression.py
Figures are shown on screen and saved to model/figures/.
Each "# %%" line starts a cell (like a MATLAB section) - in VS Code you can run cell by cell.
"""
# %% 0. setup
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "processed" / "listings_model.csv"
FIG_DIR = ROOT / "model" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "price_eur"
TEST_SIZE = 0.20      # 20% validation, 80% identification
RANDOM_STATE = 42     # fixed seed -> same split every run, results are comparable

df = pd.read_csv(DATA_CSV)
features = [c for c in df.columns if c != TARGET]
X = df[features]
Y = df[TARGET]

# split into identification (train) and validation (test) data - like id / val in the lab
X_id, X_val, Y_id, Y_val = train_test_split(X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
lid, lval = len(X_id), len(X_val)
n_regressors = len(features)

print(f"{len(df)} apartments, {n_regressors} regressors")
print(f"identification: {lid} apartments, validation: {lval} apartments\n")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=120)


# %% 1. identification data
# With one regressor the lab plots Y against X. With 22 we can't draw a 23-D plot,
# so we look at the price against the most important inputs one at a time.
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, col, label in zip(axes,
                          ["area_m2", "dist_center_km", "build_year"],
                          ["Area (m²)", "Distance to centre (km)", "Year built"]):
    ax.scatter(X_id[col], Y_id, s=12, alpha=0.5)
    ax.set_xlabel(label)
    ax.set_ylabel("Rent (EUR / month)")
    ax.grid(alpha=0.3)
fig.suptitle("1. Identification data: rent against the main regressors")
save(fig, "1_identification_data.png")


# %% 2. build PHI and solve for theta (least squares)
def make_phi(X_part):
    """PHI = [1, x1, x2, ..., xn]: a column of ones (the intercept) + one column per regressor.
    Same role as PHI_id(:, i) = X_id .^ (i-1) in the lab."""
    return np.column_stack([np.ones(len(X_part)), X_part.to_numpy(dtype=float)])


PHI_id = make_phi(X_id)                                  # shape (lid, 1 + n)
theta, *_ = np.linalg.lstsq(PHI_id, Y_id.to_numpy(), rcond=None)   # MATLAB: theta = PHI_id \ Y_id
Yhat_id = PHI_id @ theta                                 # MATLAB: Yhat_id = PHI_id * theta
MSE_id = np.mean((Y_id.to_numpy() - Yhat_id) ** 2)       # MATLAB: mean((Y_id - Yhat_id).^2)

# the same thing with scikit-learn's built-in model, to check we get identical numbers
sk_model = LinearRegression().fit(X_id, Y_id)
assert np.allclose(sk_model.intercept_, theta[0]) and np.allclose(sk_model.coef_, theta[1:]), \
    "hand-made least squares and sklearn disagree"
print("theta from np.linalg.lstsq == theta from sklearn LinearRegression  ✓")


# %% 3. validation: same theta, new data
PHI_val = make_phi(X_val)
Yhat_val = PHI_val @ theta
MSE_val = mean_squared_error(Y_val, Yhat_val)            # built-in, = mean((Y_val - Yhat_val).^2)

def report(name, y, yhat):
    mse = mean_squared_error(y, yhat)
    print(f"{name:15s} MSE = {mse:9.0f}   RMSE = {np.sqrt(mse):5.0f} EUR   "
          f"mean |error| = {np.mean(np.abs(y - yhat)):4.0f} EUR   R² = {r2_score(y, yhat):.3f}")

print()
report("identification", Y_id, Yhat_id)
report("validation", Y_val, Yhat_val)
print("(RMSE = sqrt(MSE) is in EUR, so it is easier to read: the typical prediction error.)")
print("(R² = share of the price variation explained by the model; 1 = perfect, 0 = just guessing the mean.)\n")

# 3a. like the lab's validation plot: real vs model, apartment by apartment.
#     Apartments are sorted by real price so the plot is readable.
order = np.argsort(Y_val.to_numpy())
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(Y_val.to_numpy()[order], "o-", ms=3, lw=1, label="validation data (real rent)")
ax.plot(Yhat_val[order], "o", ms=3, alpha=0.7, label="model")
ax.set_xlabel("validation apartment (sorted by real rent)")
ax.set_ylabel("Rent (EUR / month)")
ax.set_title(f"3. Validation   MSE_val = {MSE_val:.0f}   (RMSE ≈ {np.sqrt(MSE_val):.0f} EUR)")
ax.legend(); ax.grid(alpha=0.3)
save(fig, "3a_validation.png")

# 3b. predicted vs real: a perfect model puts every point on the diagonal
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
lo, hi = Y.min(), Y.max()
for ax, y, yhat, title in [(axes[0], Y_id, Yhat_id, f"identification  (MSE={MSE_id:.0f})"),
                           (axes[1], Y_val, Yhat_val, f"validation  (MSE={MSE_val:.0f})")]:
    ax.scatter(y, yhat, s=12, alpha=0.5)
    ax.plot([lo, hi], [lo, hi], "r--", label="perfect prediction")
    ax.set_xlabel("real rent (EUR)"); ax.set_ylabel("predicted rent (EUR)")
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
fig.suptitle("3b. Predicted vs real rent")
save(fig, "3b_predicted_vs_real.png")

# 3c. errors (residuals): should be centred on 0, no strong pattern
res_val = Y_val.to_numpy() - Yhat_val
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].hist(res_val, bins=30)
axes[0].axvline(0, color="r", ls="--")
axes[0].set_xlabel("error = real − predicted (EUR)"); axes[0].set_ylabel("apartments")
axes[0].set_title("distribution of validation errors")
axes[1].scatter(Yhat_val, res_val, s=12, alpha=0.5)
axes[1].axhline(0, color="r", ls="--")
axes[1].set_xlabel("predicted rent (EUR)"); axes[1].set_ylabel("error (EUR)")
axes[1].set_title("error vs prediction (a funnel shape = bigger errors for expensive flats)")
for ax in axes: ax.grid(alpha=0.3)
fig.suptitle("3c. Validation errors (residuals)")
save(fig, "3c_residuals.png")


# %% understanding theta: what each regressor does to the price
# theta_j = how many EUR the rent changes when regressor j grows by 1 unit, all others equal.
# Units differ (1 m² vs 1 km vs 1 year), so to compare importance we also show
# theta_j * std(x_j) = the EUR effect of a "typical" change in that regressor.
coef = pd.DataFrame({
    "theta (EUR per unit)": theta[1:],
    "typical change (1 std)": X_id.std().to_numpy(),
}, index=features)
coef["effect of typical change (EUR)"] = coef.iloc[:, 0] * coef.iloc[:, 1]
coef = coef.sort_values("effect of typical change (EUR)", key=np.abs, ascending=False)

pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
print(f"theta_0 (intercept) = {theta[0]:,.0f}")
print(coef.to_string(), "\n")

fig, ax = plt.subplots(figsize=(9, 7))
eff = coef["effect of typical change (EUR)"][::-1]
ax.barh(eff.index, eff.to_numpy(), color=np.where(eff > 0, "tab:green", "tab:red"))
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("change in rent (EUR) for a typical (1 std) increase of the regressor")
ax.set_title("What drives the rent? (green = raises price, red = lowers it)")
ax.grid(axis="x", alpha=0.3)
save(fig, "coefficients.png")


# %% 4. tuning: MSE(n) for a growing number of regressors
# The lab grows the polynomial degree n and watches MSE_id and MSE_val.
# Here the model grows by adding regressors one at a time (forward selection):
# at each step we add the regressor that lowers MSE_id the most.
# MSE_id always goes down (more freedom to fit the training data),
# MSE_val goes down, then flattens or rises once extra regressors only fit noise.
remaining, chosen = list(features), []
MSE_id2, MSE_val2 = [], []

# n = 0: no regressors, the model is just the mean rent (only the column of ones)
MSE_id2.append(mean_squared_error(Y_id, np.full(lid, Y_id.mean())))
MSE_val2.append(mean_squared_error(Y_val, np.full(lval, Y_id.mean())))

while remaining:
    best = min(remaining, key=lambda c: mean_squared_error(
        Y_id, LinearRegression().fit(X_id[chosen + [c]], Y_id).predict(X_id[chosen + [c]])))
    chosen.append(best); remaining.remove(best)
    m = LinearRegression().fit(X_id[chosen], Y_id)
    MSE_id2.append(mean_squared_error(Y_id, m.predict(X_id[chosen])))
    MSE_val2.append(mean_squared_error(Y_val, m.predict(X_val[chosen])))

nfin = int(np.argmin(MSE_val2))                         # MATLAB: [MSE_min, nfin] = min(MSE_val2)
print(f"best number of regressors on validation: n = {nfin}  (MSE_val = {MSE_val2[nfin]:.0f})")
print("order in which regressors were added:")
for i, c in enumerate(chosen, 1):
    print(f"  {i:2d}. {c:22s} MSE_id = {MSE_id2[i]:8.0f}   MSE_val = {MSE_val2[i]:8.0f}")

fig, ax = plt.subplots(figsize=(12, 5))
ns = np.arange(len(MSE_id2))
ax.plot(ns, MSE_val2, "o-", label="MSE validation")
ax.plot(ns, MSE_id2, "o-", label="MSE identification")
ax.axvline(nfin, color="gray", ls=":", label=f"best n = {nfin}")
ax.set_xticks(ns)
ax.set_xticklabels(["mean"] + [f"+{c}" for c in chosen], rotation=60, ha="right", fontsize=8)
ax.set_xlabel("n = number of regressors (each tick adds the named one)")
ax.set_ylabel("MSE (EUR²)")
ax.set_title("4. MSE(n)")
ax.legend(); ax.grid(alpha=0.3)
save(fig, "4_mse_n.png")


# %% 5. use the model: predict the rent of an example apartment
example = pd.DataFrame([{
    "area_m2": 55, "rooms": 2, "floor": 3, "build_year": 2018, "is_new_building": 1,
    "latitude": 46.7712, "longitude": 23.6236, "dist_center_km": 2.6,   # Marasti / Iulius Mall area
    "has_lift": 1, "has_balcony": 1, "has_terrace": 0, "has_parking": 1, "has_separate_kitchen": 0,
    "has_ac": 1, "is_furnished": 1, "has_dishwasher": 1, "has_washing_machine": 1,
    "has_internet": 1, "has_security": 0, "pet_friendly": 0, "has_central_heating": 1, "is_agency": 1,
}])[features]
pred = (make_phi(example) @ theta)[0]
print(f"\nexample: 55 m², 2 rooms, floor 3, built 2018, Marasti, parking + furnished -> "
      f"{pred:.0f} EUR/month  (typical error ± {np.sqrt(MSE_val):.0f} EUR)")

plt.show()
