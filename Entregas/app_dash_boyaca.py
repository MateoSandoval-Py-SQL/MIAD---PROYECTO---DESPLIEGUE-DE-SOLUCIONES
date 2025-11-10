# app_dash_boyaca.py
# Dash app para visualizar y predecir rendimiento caficultor (Boyacá)
# Requiere: dash, dash-bootstrap-components, pandas, numpy, plotly, joblib, scikit-learn

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from dash import Dash, dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px

# -------------------- Rutas y carga de artefactos --------------------
BASE_DIR = Path(__file__).resolve().parent
ARTI = BASE_DIR / "artifacts"

DF = pd.read_csv(ARTI / "dataset_modelo.csv")
DF_RES = pd.read_csv(ARTI / "df_resultados.csv")
META = json.loads((ARTI / "metadata.json").read_text(encoding="utf-8"))
PIPE = joblib.load(ARTI / "modelo_boyaca.pkl")
FEATURE_COLS = joblib.load(ARTI / "feature_cols.pkl")

# Normalizar columnas esperadas por el tablero
# Asegurar existencia de 'fecha' como datetime y 'rendimiento' como numérico
if "fecha" in DF.columns:
    DF["fecha"] = pd.to_datetime(DF["fecha"], errors="coerce")
elif "yyyymm" in DF.columns:
    DF["fecha"] = pd.to_datetime(DF["yyyymm"], format="%Y-%m", errors="coerce")
DF["rendimiento"] = pd.to_numeric(DF["rendimiento"], errors="coerce")

# Catálogo de variables de entrada (solo las que existan)
CAND_VARS = [c for c in ["NDVI","EVI","Precipitacion","TempMax","TempMin","HumedadRelativa"] if c in DF.columns]

# -------------------- Utilidades --------------------
def mes_ciclico(mes: int):
    """Devuelve sin/cos para el mes (1..12)."""
    sin_v = np.sin(2*np.pi*mes/12)
    cos_v = np.cos(2*np.pi*mes/12)
    return sin_v, cos_v

def build_pred_row(ndvi, evi, precip, tmax, tmin, hum, mes, feature_cols):
    """Construye un DataFrame 1xN alineado con FEATURE_COLS para PIPE.predict()."""
    m_sin, m_cos = mes_ciclico(mes)
    base = {
        "NDVI": ndvi, "EVI": evi, "Precipitacion": precip,
        "TempMax": tmax, "TempMin": tmin, "HumedadRelativa": hum,
        # generamos ambas variantes por si el pipeline espera uno u otro nombre
        "month_sin": m_sin, "month_cos": m_cos,
        "mes_sin": m_sin, "mes_cos": m_cos,
    }
    # Alinear al orden que espera el pipeline
    row = pd.DataFrame([{k: base.get(k, np.nan) for k in feature_cols}])
    # Convertir a numérico por seguridad
    row = row.apply(pd.to_numeric, errors="coerce")
    return row

def kpi_card(title, value, subtitle=None, color="primary"):
    return dbc.Card(
        dbc.CardBody([
            html.Div(title, className="text-muted"),
            html.H4(value, className="card-title"),
            html.Div(subtitle or "", className="text-muted small"),
        ]),
        className=f"shadow-sm border-0 bg-{color} bg-opacity-10",
        style={"height":"100%"}
    )

# -------------------- App --------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], title="Dashboard Boyacá")

# -------- Controles de predicción --------
controls_pred = dbc.Card(
    dbc.CardBody([
        html.H5("Predicción interactiva", className="mb-3"),
        dbc.Row([
            dbc.Col(dbc.Label("Mes"), width=3),
            dbc.Col(dcc.Slider(1, 12, 1, value=1, id="in-mes",
                               marks={i:str(i) for i in range(1,13)}), width=9)
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(dbc.Label("NDVI"), md=6),
            dbc.Col(dbc.Label("EVI"), md=6),
        ], className="mt-2"),
        dbc.Row([
            dbc.Col(dcc.Slider(0.2, 0.95, 0.01, value=float(DF["NDVI"].median()) if "NDVI" in DF else 0.6,
                               id="in-ndvi"), md=6),
            dbc.Col(dcc.Slider(0.1, 0.9, 0.01, value=float(DF["EVI"].median()) if "EVI" in DF else 0.4,
                               id="in-evi"), md=6),
        ]),
        html.Hr(),
        dbc.Row([
            dbc.Col(dbc.Label("Precipitación (mm)"), md=12),
            dbc.Col(dcc.Slider(60, 300, 1,
                               value=float(DF["Precipitacion"].median()) if "Precipitacion" in DF else 160,
                               id="in-precip")), 
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(dbc.Label("Temp. Máx. (°C)"), md=6),
            dbc.Col(dbc.Label("Temp. Mín. (°C)"), md=6),
        ]),
        dbc.Row([
            dbc.Col(dcc.Slider(20, 30, 0.1,
                               value=float(DF["TempMax"].median()) if "TempMax" in DF else 25,
                               id="in-tmax"), md=6),
            dbc.Col(dcc.Slider(10, 20, 0.1,
                               value=float(DF["TempMin"].median()) if "TempMin" in DF else 15,
                               id="in-tmin"), md=6),
        ]),
        dbc.Row([
            dbc.Col(dbc.Label("Humedad Relativa (%)"), md=12),
            dbc.Col(dcc.Slider(60, 90, 1,
                               value=float(DF["HumedadRelativa"].median()) if "HumedadRelativa" in DF else 75,
                               id="in-hum")), 
        ], className="mb-3"),
        dbc.Button("Predecir rendimiento", id="btn-predict", color="primary", className="w-100"),
        html.Div(id="pred-output", className="mt-3 fw-bold"),
    ]),
    className="shadow-sm"
)

# -------- Layout --------
app.layout = dbc.Container(fluid=True, children=[
    html.Br(),
    dbc.Row([
        dbc.Col(html.H2("📊 Dashboard – Rendimiento Café (Boyacá)"), md=8),
        dbc.Col(html.Div(f"Modelo: {META.get('model', 'N/D')}"), md=4, className="text-end align-self-center")
    ]),
    html.Hr(),
    # KPIs
    dbc.Row([
        dbc.Col(kpi_card("R² (test)", f"{META['metricas_test_snapshot']['R2']:.3f}", color="success"), md=3),
        dbc.Col(kpi_card("RMSE (test)", f"{META['metricas_test_snapshot']['RMSE']:.3f}", color="danger"), md=3),
        dbc.Col(kpi_card("MAE (test)", f"{META['metricas_test_snapshot']['MAE']:.3f}", color="warning"), md=3),
        dbc.Col(kpi_card("Observaciones", f"{META.get('n_obs', len(DF)):,}"), md=3),
    ], className="g-3"),
    html.Br(),
    # Gráficos principales
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Evolución mensual del rendimiento"),
            dcc.Graph(
                id="ts-rend",
                figure=px.line(DF.sort_values("fecha"), x="fecha", y="rendimiento",
                               labels={"fecha":"Fecha","rendimiento":"t/ha"})
                        .update_layout(margin=dict(l=10,r=10,t=10,b=10))
            )
        ])), md=8, className="mb-3"),
        dbc.Col(controls_pred, md=4, className="mb-3")
    ]),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Relación con variables"),
            dbc.Row([
                dbc.Col(dcc.Dropdown(
                    id="dd-var",
                    options=[{"label":v, "value":v} for v in CAND_VARS] or [{"label":"NDVI","value":"NDVI"}],
                    value=CAND_VARS[0] if CAND_VARS else "NDVI",
                    clearable=False
                ), md=4)
            ]),
            dcc.Graph(id="scat-rel")
        ])), md=8, className="mb-3"),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Resultados de modelos"),
            dash_table.DataTable(
                id="tbl-result",
                data=DF_RES.round(4).to_dict("records"),
                columns=[{"name":c, "id":c} for c in DF_RES.columns],
                sort_action="native",
                page_size=8,
                style_table={"overflowX":"auto"},
                style_cell={"fontFamily":"Inter, system-ui", "fontSize":"14px", "padding":"6px"},
                style_header={"fontWeight":"700"}
            )
        ])), md=4, className="mb-3"),
    ]),
    html.Br()
])

# -------------------- Callbacks --------------------
@app.callback(
    Output("scat-rel", "figure"),
    Input("dd-var", "value")
)
def update_scatter(var):
    dfp = DF.dropna(subset=[var, "rendimiento"])
    fig = px.scatter(dfp, x=var, y="rendimiento", trendline="ols",
                     labels={var:var, "rendimiento":"t/ha"})
    fig.update_layout(margin=dict(l=10,r=10,t=10,b=10))
    return fig

@app.callback(
    Output("pred-output", "children"),
    Input("btn-predict", "n_clicks"),
    State("in-ndvi","value"), State("in-evi","value"),
    State("in-precip","value"), State("in-tmax","value"),
    State("in-tmin","value"), State("in-hum","value"),
    State("in-mes","value"),
    prevent_initial_call=True
)
def predict_click(nc, ndvi, evi, precip, tmax, tmin, hum, mes):
    fila = build_pred_row(ndvi, evi, precip, tmax, tmin, hum, mes, FEATURE_COLS)
    try:
        pred = float(PIPE.predict(fila)[0])
        return f"🌱 Rendimiento estimado: {pred:.2f} t/ha"
    except Exception as e:
        return f"⚠️ Error al predecir: {e}"

# -------------------- Main --------------------
if __name__ == "__main__":
    app.run(debug=True, port=8050)   # <- antes: app.run_server(...)

