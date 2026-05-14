"""
Getaround – Delay Analysis Dashboard
Jedha Certification – Deployment Block
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Getaround · Delay Analysis",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

BLUE   = "#0f3460"
RED    = "#e94560"
ORANGE = "#f39c12"
GREEN  = "#27ae60"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stMetricValue"] { font-size: 2rem !important; }
  h2 { border-bottom: 2px solid #e94560; padding-bottom: 6px; }
</style>
""", unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_excel("get_around_delay_analysis.xlsx")

    # Locations consécutives avec retard précédent joint
    consec = df[df["previous_ended_rental_id"].notna()].copy()
    prev = (
        df[["rental_id", "delay_at_checkout_in_minutes"]]
        .rename(columns={
            "rental_id": "previous_ended_rental_id",
            "delay_at_checkout_in_minutes": "prev_delay",
        })
    )
    prev["previous_ended_rental_id"] = prev["previous_ended_rental_id"].astype(float)
    consec = consec.merge(prev, on="previous_ended_rental_id", how="left")
    consec["impacted"] = (
        consec["prev_delay"].fillna(0)
        > consec["time_delta_with_previous_rental_in_minutes"]
    )
    return df, consec

df, consec = load_data()
ended = df[df["state"] == "ended"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image(
    "https://lever-client-logos.s3.amazonaws.com/"
    "2bd4cdf9-37f2-497f-9096-c2793296a75f-1568844229943.png",
    use_column_width=True,
)
st.sidebar.markdown("## ⚙️ Simulateur")
threshold = st.sidebar.slider(
    "Délai minimum (minutes)", 0, 720, 60, step=15,
    help="Délai minimum imposé entre deux locations consécutives."
)
scope = st.sidebar.radio(
    "Périmètre d'application",
    ["Toutes les voitures", "Connect uniquement", "Mobile uniquement"],
)
scope_key = {"Toutes les voitures": "all",
             "Connect uniquement": "connect",
             "Mobile uniquement": "mobile"}[scope]

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Getaround** · Jedha Certification\n\n"
    "[GitHub](https://github.com/YOUR_USERNAME/getaround)"
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🚗 Getaround – Analyse des retards au checkout")
st.markdown(
    "Dashboard pour aider le **Product Manager** à choisir le bon **threshold** "
    "et le bon **scope** pour la fonctionnalité de délai minimum entre locations."
)
st.markdown("---")

# ── KPIs ──────────────────────────────────────────────────────────────────────
has_d  = ended["delay_at_checkout_in_minutes"].notna()
n_late = int((ended.loc[has_d, "delay_at_checkout_in_minutes"] > 0).sum())
n_obs  = int(has_d.sum())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total locations", f"{len(df):,}")
k2.metric("Locations en retard", f"{n_late:,}", f"{n_late/n_obs*100:.1f} % des terminées")
k3.metric("Paires consécutives", f"{len(consec):,}")
k4.metric("Conducteurs suivants impactés",
          f"{consec['impacted'].sum()}",
          f"{consec['impacted'].mean()*100:.1f} % des paires")

st.markdown("---")

# ── Section 1 – Distribution des retards ──────────────────────────────────────
st.header("1 · Distribution des retards")

late_vals = (
    ended.loc[ended["delay_at_checkout_in_minutes"] > 0, "delay_at_checkout_in_minutes"]
    .clip(upper=720)
)

col1, col2 = st.columns(2)

with col1:
    fig1 = px.histogram(
        late_vals, nbins=60,
        title="Retards au checkout (plafonné 720 min)",
        labels={"value": "Retard (min)", "count": "Locations"},
        color_discrete_sequence=[RED],
    )
    # Percentiles
    for p, c in [(50, BLUE), (75, ORANGE), (90, GREEN)]:
        v = np.percentile(late_vals, p)
        fig1.add_vline(x=v, line_dash="dash", line_color=c,
                       annotation_text=f"P{p}={v:.0f}", annotation_position="top")
    fig1.update_layout(showlegend=False, margin=dict(t=50))
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    late_by_type = (
        ended.groupby("checkin_type")
        .apply(lambda x: pd.Series({
            "En retard"     : int((x["delay_at_checkout_in_minutes"] > 0).sum()),
            "À l'heure/avance": int((x["delay_at_checkout_in_minutes"].notna() &
                                     (x["delay_at_checkout_in_minutes"] <= 0)).sum()),
        }))
        .reset_index()
        .melt(id_vars="checkin_type", var_name="statut", value_name="count")
    )
    fig2 = px.bar(
        late_by_type, x="checkin_type", y="count", color="statut",
        barmode="stack",
        title="Retards vs à l'heure par type de check-in",
        labels={"checkin_type": "Type", "count": "Locations"},
        color_discrete_map={"En retard": RED, "À l'heure/avance": BLUE},
    )
    fig2.update_layout(margin=dict(t=50))
    st.plotly_chart(fig2, use_container_width=True)

# Tableau percentiles
pct_df = pd.DataFrame({
    "Percentile": ["P50", "P75", "P90", "P95", "P99"],
    "Retard (min)": [int(np.percentile(late_vals, p)) for p in [50, 75, 90, 95, 99]],
})
st.dataframe(pct_df.set_index("Percentile").T, use_container_width=False)

st.markdown("---")

# ── Section 2 – Impact sur le prochain conducteur ─────────────────────────────
st.header("2 · Impact sur le conducteur suivant")

col3, col4 = st.columns(2)

with col3:
    delta_vals = consec["time_delta_with_previous_rental_in_minutes"].clip(upper=720)
    fig3 = px.histogram(
        delta_vals, nbins=48,
        title="Délai entre deux locations consécutives",
        labels={"value": "Délai (min)", "count": "Paires"},
        color_discrete_sequence=[BLUE],
    )
    fig3.add_vline(x=threshold, line_dash="dash", line_color=RED,
                   annotation_text=f"Threshold : {threshold} min")
    fig3.update_layout(showlegend=False, margin=dict(t=50))
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    impact_type = (
        consec.groupby("checkin_type")["impacted"]
        .agg(total="count", impacted="sum")
        .assign(pct=lambda x: (x["impacted"] / x["total"] * 100).round(1))
        .reset_index()
    )
    fig4 = px.bar(
        impact_type, x="checkin_type", y="pct",
        title="% de conducteurs suivants impactés",
        labels={"checkin_type": "Type", "pct": "% impacté"},
        color="checkin_type",
        color_discrete_map={"connect": BLUE, "mobile": RED},
        text="pct",
    )
    fig4.update_traces(texttemplate="%{text:.1f} %", textposition="outside")
    fig4.update_layout(showlegend=False, margin=dict(t=50), yaxis_range=[0, 25])
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# ── Section 3 – Simulateur ────────────────────────────────────────────────────
st.header(f"3 · Simulateur — threshold {threshold} min · {scope}")


def simulate(t: int, s: str) -> dict:
    if s == "connect":
        df_s = df[df["checkin_type"] == "connect"]
        c_s  = consec[consec["checkin_type"] == "connect"]
    elif s == "mobile":
        df_s = df[df["checkin_type"] == "mobile"]
        c_s  = consec[consec["checkin_type"] == "mobile"]
    else:
        df_s, c_s = df, consec

    blocked = df_s[
        df_s["time_delta_with_previous_rental_in_minutes"].notna() &
        (df_s["time_delta_with_previous_rental_in_minutes"] < t)
    ]
    n_prob  = c_s["impacted"].sum()
    solved  = c_s[
        c_s["impacted"] &
        (c_s["time_delta_with_previous_rental_in_minutes"] < t)
    ]
    return {
        "n_blocked"   : len(blocked),
        "pct_blocked" : len(blocked) / len(df_s) * 100 if len(df_s) else 0,
        "n_solved"    : len(solved),
        "pct_solved"  : len(solved) / n_prob * 100 if n_prob else 0,
    }


r = simulate(threshold, scope_key)
s1, s2, s3 = st.columns(3)
s1.metric("Locations bloquées",
          f"{r['n_blocked']:,}",
          f"{r['pct_blocked']:.1f} % du périmètre")
s2.metric("Cas problématiques résolus",
          f"{r['n_solved']}",
          f"{r['pct_solved']:.1f} % des cas")
s3.metric("Impact revenus (approx.)",
          f"−{r['pct_blocked']:.1f} %",
          help="Part des locations du périmètre qui seraient refusées")

# Courbes threshold sweep
thresholds = list(range(0, 721, 10))
blocked_l, solved_l = [], []
for t in thresholds:
    res = simulate(t, scope_key)
    blocked_l.append(res["pct_blocked"])
    solved_l.append(res["pct_solved"])

fig5 = go.Figure()
fig5.add_scatter(x=thresholds, y=blocked_l, mode="lines",
                 name="% locations bloquées", line=dict(color=RED, width=2))
fig5.add_scatter(x=thresholds, y=solved_l, mode="lines",
                 name="% problèmes résolus", line=dict(color=GREEN, width=2))
fig5.add_vline(x=threshold, line_dash="dash", line_color=ORANGE,
               annotation_text=f"{threshold} min")
fig5.update_layout(
    title=f"Trade-off : locations bloquées vs cas résolus ({scope})",
    xaxis_title="Threshold (minutes)",
    yaxis_title="%",
    legend=dict(orientation="h"),
    margin=dict(t=50),
)
st.plotly_chart(fig5, use_container_width=True)

st.markdown("---")

# ── Section 4 – Impact revenus par périmètre ──────────────────────────────────
st.header("4 · Comparaison des périmètres")

sweep_all = {}
for s in ["all", "connect", "mobile"]:
    sweep_all[s] = [simulate(t, s)["pct_blocked"] for t in thresholds]

fig6 = go.Figure()
style = {"all": (RED, "solid"), "connect": (BLUE, "dash"), "mobile": (ORANGE, "dot")}
for s, (c, dash) in style.items():
    fig6.add_scatter(x=thresholds, y=sweep_all[s], mode="lines",
                     name=s, line=dict(color=c, width=2, dash=dash))
fig6.add_vline(x=threshold, line_dash="dash", line_color="grey", opacity=0.5)
fig6.update_layout(
    title="% locations bloquées selon le threshold et le périmètre",
    xaxis_title="Threshold (minutes)",
    yaxis_title="% locations bloquées",
    legend=dict(orientation="h"),
    margin=dict(t=50),
)
st.plotly_chart(fig6, use_container_width=True)

# Tableau récapitulatif
scenarios = []
for t in [30, 60, 90, 120, 180]:
    for s in ["all", "connect", "mobile"]:
        res = simulate(t, s)
        scenarios.append({
            "Threshold": f"{t} min",
            "Scope": s,
            "Bloquées": f"{res['n_blocked']} ({res['pct_blocked']:.1f} %)",
            "Résolus": f"{res['n_solved']} ({res['pct_solved']:.0f} %)",
        })

st.dataframe(pd.DataFrame(scenarios), use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#aaa;font-size:0.82em;'>"
    "Getaround Delay Dashboard · Jedha Bootcamp Certification · 2024</p>",
    unsafe_allow_html=True,
)
