import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
import io

# ==========================================
# 1. CONFIGURACIÓN DE PÁGINA Y PALETA UX/UI
# ==========================================
st.set_page_config(page_title="Optimización Heurística V5", layout="wide", initial_sidebar_state="expanded")

# Paleta Estricta Ausenco
C_NAVY = '#004764'      # Dark Blue
C_BLACK = '#101820'     # Black
C_BLUE = '#0095C8'      # Light Blue
C_LIME = '#C4D600'      # Lime Green
C_GREY = '#D1DDE6'      # Grey
C_WHITE = '#FFFFFF'     # White

C_SEQUENCE = [C_NAVY, C_BLUE, '#4EA7C9', '#003349']

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Roboto:wght@300;400;500&display=swap');

    html, body, [class*="css"] {{ font-family: 'Roboto', sans-serif !important; font-weight: 400; }}
    h1, h2, h3, h4, h5, h6 {{ font-family: 'Poppins', sans-serif !important; font-weight: 600; }}

    /* Checkboxes verdes */
    div[data-testid="stCheckbox"] > label > div[data-testid="stMarkdownContainer"] > p {{ color: {C_NAVY}; font-weight: 500; }}
    .st-cx {{ background-color: {C_LIME} !important; border-color: {C_LIME} !important; }}
    
    /* Flecha del Sidebar */
    [data-testid="collapsedControl"] svg {{ display: none; }}
    [data-testid="collapsedControl"]::before {{ content: "◀"; color: {C_NAVY}; font-size: 20px; font-weight: bold; display: flex; align-items: center; justify-content: center; height: 100%; }}
    
    /* Header Personalizado */
    .header-container {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid {C_LIME}; padding-bottom: 10px; margin-bottom: 20px; }}
    .header-title {{ color: {C_NAVY}; margin: 0; font-size: 28px; }}

    /* Tarjetas Modulares (Verticales, Anchas) */
    .kpi-container {{ display: flex; flex-direction: column; gap: 10px; margin-bottom: 20px; }}
    .kpi-card {{ background-color: {C_WHITE}; border-left: 5px solid {C_LIME}; border-radius: 4px; padding: 15px 20px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 2px 6px rgba(0,0,0,0.08); border-top: 1px solid {C_GREY}; border-right: 1px solid {C_GREY}; border-bottom: 1px solid {C_GREY}; }}
    .kpi-card-inactive {{ border-left: 5px solid {C_GREY}; opacity: 0.6; }}
    
    .kpi-disc-title {{ color: {C_NAVY}; font-family: 'Poppins', sans-serif !important; font-size: 14px; font-weight: 600; width: 30%; text-transform: uppercase; }}
    .kpi-stat-group {{ display: flex; flex-direction: column; align-items: center; width: 35%; }}
    .kpi-stat-val {{ color: {C_BLACK}; font-size: 26px; font-weight: 700; font-family: 'Poppins', sans-serif !important; line-height: 1; margin-bottom: 4px; }}
    .kpi-stat-sub {{ color: {C_BLUE}; font-size: 11px; font-weight: 600; text-transform: uppercase; }}
    .kpi-stat-val-dash {{ color: {C_GREY}; font-size: 26px; font-weight: 700; font-family: 'Poppins', sans-serif !important; line-height: 1; margin-bottom: 4px; }}
    
    /* Tabs de Streamlit */
    div[data-baseweb="tab-list"] {{ gap: 20px; }}
    div[data-baseweb="tab"] {{ font-family: 'Poppins', sans-serif !important; font-size: 16px; font-weight: 600; color: {C_NAVY}; }}
    div[aria-selected="true"] {{ border-bottom-color: {C_LIME} !important; }}

    /* Matriz Reducida al Máximo */
    [data-testid="stDataFrame"] div[class^="css"] {{ font-size: 7px !important; line-height: 1 !important; }}
    th {{ background-color: {C_NAVY} !important; color: {C_WHITE} !important; font-size: 8px !important; padding: 2px 4px !important; }}
    td {{ padding: 2px 4px !important; font-size: 7px !important; }}
    </style>
""", unsafe_allow_html=True)

st.markdown(f"""
    <div class="header-container">
        <h1 class="header-title">Balance de carga - Optimización Heurística</h1>
    </div>
""", unsafe_allow_html=True)

# ==========================================
# 2. MOTORES DE CÁLCULO
# ==========================================
@st.cache_data
def load_and_clean_data(uploaded_file, semanas_ano):
    try:
        df = pd.read_csv(uploaded_file, header=0) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file, header=0)
    except Exception as e:
        st.error(f"Error en la ingesta: {e}")
        return pd.DataFrame()

    df.columns = [str(c).strip() for c in df.columns]
    if 'Criticality' in df.columns: df['Criticality'] = pd.to_numeric(df['Criticality'], errors='coerce').fillna(1).astype(int)
    if 'Maximum_Tolerance_Value' in df.columns: df['Maximum_Tolerance_Value'] = pd.to_numeric(df['Maximum_Tolerance_Value'], errors='coerce').fillna(0).astype(int)
    if 'Work' in df.columns: df['Work'] = pd.to_numeric(df['Work'], errors='coerce').fillna(0.0)
    if 'Frequency_Value' in df.columns: df['Frequency_Value'] = pd.to_numeric(df['Frequency_Value'], errors='coerce').fillna(semanas_ano)

    df['Semanas_Intervalo'] = df['Frequency_Value']
    if 'Frequency_Unit' in df.columns and 'Average_Daily_Usage' in df.columns:
        mask = df['Frequency_Unit'].str.lower().str.contains('hour', na=False)
        adu = pd.to_numeric(df['Average_Daily_Usage'], errors='coerce').fillna(24) 
        df.loc[mask, 'Semanas_Intervalo'] = (df.loc[mask, 'Frequency_Value'] / adu) / 7
        
    df['Semanas_Intervalo'] = np.maximum(1, df['Semanas_Intervalo'].round().astype(int))
    return df

@st.cache_data
def run_baseline_engine(df, anchor_seeds, semanas_ano, tipo_base):
    HORIZONTE = semanas_ano * 3
    especialidades = df['Labour'].dropna().unique()
    calendario_base = {esp: np.zeros(HORIZONTE) for esp in especialidades}
    
    for idx, row in df.iterrows():
        esp = row['Labour']
        if pd.isna(esp): continue
        hh = row['Work']
        frecuencia = row['Semanas_Intervalo']
        anchor_id = str(row.get('Shutdown_Anchor_ID', '')).strip()
        
        if anchor_id and anchor_id.lower() != 'nan':
            match = re.search(r'^(\d+)W', anchor_id.upper())
            intervalo_parada = int(match.group(1)) if match else semanas_ano 
            sem_inicio = anchor_seeds.get(anchor_id, 1) - 1 
            semanas_ejecucion = [sem_inicio + i * intervalo_parada for i in range(HORIZONTE // intervalo_parada + 1) if (sem_inicio + i * intervalo_parada) < HORIZONTE]
        else:
            if tipo_base == "Caso Base 1 (Tradicional, Todo Sem 1)":
                offset_base = 0
            else:
                # Ruido determinista simulando error humano
                offset_base = int(hash(str(esp) + str(idx)) % frecuencia)
            semanas_ejecucion = [offset_base + i * frecuencia for i in range(HORIZONTE // frecuencia + 1) if (offset_base + i * frecuencia) < HORIZONTE]

        for s_final in semanas_ejecucion:
            calendario_base[esp][s_final] += hh
            
    return calendario_base

@st.cache_data
def run_heuristic_engine(df, anchor_seeds, semanas_ano):
    HORIZONTE = semanas_ano * 3
    especialidades = df['Labour'].dropna().unique()
    calendario = {esp: np.zeros(HORIZONTE) for esp in especialidades}
    output_records = []
    
    df_sorted = df.sort_values(by=['Criticality', 'Maximum_Tolerance_Value', 'Work'], ascending=[False, True, False])

    for idx, row in df_sorted.iterrows():
        esp = row['Labour']
        if pd.isna(esp): continue
        
        hh = row['Work']
        frecuencia = row['Semanas_Intervalo']
        tolerancia = row['Maximum_Tolerance_Value']
        anchor_id = str(row.get('Shutdown_Anchor_ID', '')).strip()
        tag = row.get('Location_Code', 'N/A')
        
        task_id = row.get('Task_ID', f"T-{idx+2}")
        if pd.isna(task_id) or str(task_id).strip() == '': task_id = f"T-{idx+2}"
        
        semanas_ejecucion = []
        clasificacion = 'Fase 3'
        
        if anchor_id and anchor_id.lower() != 'nan':
            clasificacion = 'Fase 1'
            match = re.search(r'^(\d+)W', anchor_id.upper())
            intervalo_parada = int(match.group(1)) if match else semanas_ano 
            sem_inicio = anchor_seeds.get(anchor_id, 1) - 1 
            semanas_teoricas = [sem_inicio + i * intervalo_parada for i in range(HORIZONTE // intervalo_parada + 1)]
            semanas_ejecucion = [s for s in semanas_teoricas if s < HORIZONTE]
            
        else:
            if tolerancia == 0: clasificacion = 'Fase 2'
            mejor_offset, menor_carga_promedio = 0, float('inf')
            
            for offset_candidato in range(frecuencia):
                semanas_prueba = [offset_candidato + i * frecuencia for i in range(HORIZONTE // frecuencia + 1) if offset_candidato + i * frecuencia < HORIZONTE]
                if not semanas_prueba: continue
                carga_promedio = np.mean([calendario[esp][s] for s in semanas_prueba])
                if carga_promedio < menor_carga_promedio:
                    menor_carga_promedio = carga_promedio
                    mejor_offset = offset_candidato
            
            semanas_base = [mejor_offset + i * frecuencia for i in range(HORIZONTE // frecuencia + 1) if mejor_offset + i * frecuencia < HORIZONTE]
            
            for s_base in semanas_base:
                rango_min = max(0, s_base - tolerancia)
                rango_max = min(HORIZONTE - 1, s_base + tolerancia)
                cargas_rango = [calendario[esp][s] for s in range(rango_min, rango_max + 1)]
                semana_optima = rango_min + np.argmin(cargas_rango)
                semanas_ejecucion.append(semana_optima)

        for s_final in semanas_ejecucion:
            calendario[esp][s_final] += hh
            output_records.append({
                'ID Tarea': task_id,
                'TAG': tag,
                'Disciplina': esp,
                'Frecuencia': frecuencia,
                'Tolerancia': tolerancia,
                'Planned_Date_Week': s_final + 1,
                'Estimated_Work_Hours': hh,
                'Clasificacion': clasificacion
            })
            
    return calendario, pd.DataFrame(output_records)

def calcular_costo(carga_array, cap_disp_efec, cap_disp_nom, semanas, t_int, t_ext):
    """
    Costo Interno (Fijo): Paga el turno completo (Capacidad Nominal) a tarifa interna.
    Costo Externo (Variable): Paga tarifa externa SOLO a las horas que superen la Capacidad Efectiva.
    """
    costo_int = cap_disp_nom * semanas * t_int
    costo_ext = sum([max(0, c - cap_disp_efec) for c in carga_array]) * t_ext
    return costo_int, costo_ext

def calcular_hc_optimo(carga_array, hr_efec, hr_nom, semanas, t_int, t_ext):
    if hr_efec <= 0: return 0
    costos = []
    for hc in range(0, 151):
        cap_efec = hc * hr_efec
        cap_nom = hc * hr_nom
        c_int, c_ext = calcular_costo(carga_array, cap_efec, cap_nom, semanas, t_int, t_ext)
        costos.append(c_int + c_ext)
    return np.argmin(costos)

# ==========================================
# 3. PANEL LATERAL (SIDEBAR)
# ==========================================
st.sidebar.markdown(f"<h3 style='color: {C_NAVY};'>1. Ingesta de Datos</h3>", unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Cargar Plan (CSV/Excel)", type=['csv', 'xlsx'])

if 'semanas_ano_state' not in st.session_state: st.session_state['semanas_ano_state'] = 52
st.sidebar.markdown(f"<h3 style='color: {C_NAVY}; margin-top:15px;'>2. Calendario</h3>", unsafe_allow_html=True)
semanas_ano = st.sidebar.number_input("Semanas / Año Operativo", min_value=12, max_value=104, value=st.session_state['semanas_ano_state'])
st.session_state['semanas_ano_state'] = semanas_ano
intervalo_paradas = st.sidebar.number_input("Etiquetas Gráfico (Semanas)", min_value=1, max_value=semanas_ano, value=13)
HORIZONTE = semanas_ano * 3

if uploaded_file:
    df_raw = load_and_clean_data(uploaded_file, semanas_ano)
    if df_raw.empty: st.stop()
    especialidades_unicas = sorted(df_raw['Labour'].dropna().unique().tolist())
    
    st.sidebar.markdown(f"<h3 style='color: {C_NAVY}; margin-top:15px;'>3. Filtro Global</h3>", unsafe_allow_html=True)
    filtro_global = []
    for esp in especialidades_unicas:
        if st.sidebar.checkbox(esp, value=True, key=f"chk_{esp}"): filtro_global.append(esp)
    
    anchors_detectados = [a for a in df_raw['Shutdown_Anchor_ID'].unique() if pd.notna(a) and str(a).strip().lower() != 'nan']
    anchor_seeds = {}
    if anchors_detectados:
        st.sidebar.markdown(f"<h3 style='color: {C_NAVY}; margin-top:15px;'>4. Semillas de Paradas</h3>", unsafe_allow_html=True)
        for anchor in anchors_detectados:
            match = re.search(r'^(\d+)W', anchor.upper())
            seed_default = int(match.group(1)) if match else 1
            anchor_seeds[anchor] = st.sidebar.number_input(f"Inicio {anchor}", min_value=1, max_value=HORIZONTE, value=seed_default)

    st.sidebar.markdown(f"<h3 style='color: {C_NAVY}; margin-top:15px;'>5. Escenario Base y Costos</h3>", unsafe_allow_html=True)
    tipo_base = st.sidebar.selectbox("Tipo de Caso Base", ["Caso Base 1 (Tradicional, Todo Sem 1)", "Caso Base 2 (Ruido Determinista)"])
    tarifa_interna = st.sidebar.number_input("Tarifa Interna ($/h)", value=30.0)
    tarifa_externa = st.sidebar.number_input("Tarifa Contratista ($/h)", value=80.0)

    st.sidebar.markdown(f"<h3 style='color: {C_NAVY}; margin-top:15px;'>6. Capacidad Nominal</h3>", unsafe_allow_html=True)
    col_d, col_h = st.sidebar.columns(2)
    dias_sem = col_d.number_input("Días/Semana", 1, 7, 7)
    horas_dia = col_h.number_input("Horas/Turno", 1.0, 24.0, 12.0)
    hh_nominales = dias_sem * horas_dia
    
    hc_dict, wt_dict, noprog_dict = {}, {}, {}
    
    calendario_base_temp = run_baseline_engine(df_raw, anchor_seeds, semanas_ano, tipo_base)
    
    st.sidebar.markdown("**Distribución por Disciplina:**")
    for esp in especialidades_unicas:
        st.sidebar.markdown(f"<div style='font-size:14px; font-weight:600; color:{C_NAVY}; margin-top: 15px;'>{esp}</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.sidebar.columns(3)
        
        wt_dict[esp] = c2.number_input("% WT", min_value=0, max_value=100, value=60, step=5, key=f"wt_{esp}")
        noprog_dict[esp] = c3.number_input("% NoProg", min_value=0, max_value=100, value=10, step=5, key=f"np_{esp}")
        
        hr_efec_wt_temp = hh_nominales * (wt_dict[esp] / 100.0)
        hr_efec_prog_temp = hr_efec_wt_temp * (1 - noprog_dict[esp] / 100.0)
        max_base_y1 = np.max(calendario_base_temp[esp][0:semanas_ano])
        hc_default_int = int(np.ceil(max_base_y1 / hr_efec_prog_temp)) if hr_efec_prog_temp > 0 else 5
        
        hc_dict[esp] = c1.number_input("HC Manual", min_value=0, value=hc_default_int, step=1, key=f"hc_{esp}")
        
        wt_val = wt_dict[esp]
        np_pct = noprog_dict[esp]
        np_efectivo = (wt_val * (np_pct / 100.0))
        prog_efectivo = wt_val - np_efectivo
        rem_val = max(0, 100 - wt_val)
        
        html_bar = f"""
        <div style='width: 100%; height: 8px; display: flex; border-radius: 4px; overflow: hidden; margin-top: 5px;'>
            <div style='width: {prog_efectivo}%; background-color: {C_BLUE};' title='Programable ({prog_efectivo:.1f}%)'></div>
            <div style='width: {np_efectivo}%; background-color: {C_LIME};' title='Correctivos ({np_efectivo:.1f}%)'></div>
            <div style='width: {rem_val}%; background-color: {C_GREY};' title='Pérdidas ({rem_val:.1f}%)'></div>
        </div>
        """
        st.sidebar.markdown(html_bar, unsafe_allow_html=True)

    calendario_dict, df_output = run_heuristic_engine(df_raw, anchor_seeds, semanas_ano)
    calendario_base = calendario_base_temp
    
    # ==========================================
    # LÓGICA DE PESTAÑAS Y GRAFICAS
    # ==========================================
    tab1, tab2, tab3 = st.tabs(["1. Escenario Base", "2. Escenario Optimizado", "3. Comparativa y Ahorros"])

    hc_optimo_vista_dict = {}
    terceros_pico_opt = {}
    terceros_pico_base = {}
    
    for esp in especialidades_unicas:
        carga_opt_y1 = calendario_dict[esp][0:semanas_ano]
        carga_base_y1 = calendario_base[esp][0:semanas_ano]
        hr_efec_wt = hh_nominales * (wt_dict[esp] / 100.0)
        hr_efec_prog = hr_efec_wt * (1 - noprog_dict[esp] / 100.0)
        
        hc_optimo_vista_dict[esp] = calcular_hc_optimo(carga_opt_y1, hr_efec_prog, hh_nominales, semanas_ano, tarifa_interna, tarifa_externa)
        
        cap_base = hc_dict[esp] * hr_efec_prog
        cap_opt = hc_optimo_vista_dict[esp] * hr_efec_prog
        
        terceros_pico_base[esp] = int(np.ceil(max(0, np.max(carga_base_y1) - cap_base) / hr_efec_prog)) if hr_efec_prog > 0 else 0
        terceros_pico_opt[esp] = int(np.ceil(max(0, np.max(carga_opt_y1) - cap_opt) / hr_efec_prog)) if hr_efec_prog > 0 else 0

    max_absoluto = 0
    for esp in filtro_global:
        max_absoluto = max(max_absoluto, np.max(calendario_base[esp][0:semanas_ano]), np.max(calendario_dict[esp][0:semanas_ano]))
    y_max_fijo = (max_absoluto * 1.1) if max_absoluto > 0 else 50
    tickvals = [1] + [i for i in range(intervalo_paradas, semanas_ano + 1, intervalo_paradas)]

    def render_kpi_cards(is_base):
        html_cards = "<div class='kpi-container'>"
        for esp in especialidades_unicas:
            if esp not in filtro_global:
                html_cards += f"<div class='kpi-card kpi-card-inactive'><div class='kpi-disc-title' style='color:{C_GREY};'>{esp}</div><div class='kpi-stat-group'><span class='kpi-stat-sub' style='color:{C_GREY};'>HC Propio</span><span class='kpi-stat-val-dash'>-</span></div><div class='kpi-stat-group'><span class='kpi-stat-sub' style='color:{C_GREY};'>Terceros (Pico)</span><span class='kpi-stat-val-dash'>-</span></div></div>"
                continue
            
            hc_eval = hc_dict[esp] if is_base else hc_optimo_vista_dict[esp]
            terc_eval = terceros_pico_base[esp] if is_base else terceros_pico_opt[esp]
            
            html_cards += f"""
            <div class='kpi-card'>
                <div class='kpi-disc-title'>{esp}</div>
                <div class='kpi-stat-group'><span class='kpi-stat-sub'>HC Propio</span><span class='kpi-stat-val'>{hc_eval}</span></div>
                <div class='kpi-stat-group'><span class='kpi-stat-sub'>Terceros (Pico)</span><span class='kpi-stat-val' style='color:{C_LIME if terc_eval > 0 else C_BLACK};'>{terc_eval}</span></div>
            </div>"""
        html_cards += "</div>"
        return html_cards

    def render_profile_chart(is_base, f_fase=None):
        fig = go.Figure()
        
        for i, esp in enumerate(filtro_global):
            if is_base:
                carga = calendario_base[esp][0:semanas_ano]
            else:
                df_esp = df_output[df_output['Disciplina'] == esp]
                if f_fase == "Fase 1": df_esp = df_esp[df_esp['Clasificacion'] == 'Fase 1']
                elif f_fase == "Fase 2": df_esp = df_esp[df_esp['Clasificacion'].isin(['Fase 1', 'Fase 2'])]
                df_ventana = df_esp[(df_esp['Planned_Date_Week'] > 0) & (df_esp['Planned_Date_Week'] <= semanas_ano)]
                carga = df_ventana.groupby('Planned_Date_Week')['Estimated_Work_Hours'].sum().reindex(list(range(1, semanas_ano + 1)), fill_value=0).values
                
            fig.add_trace(go.Bar(x=list(range(1, semanas_ano + 1)), y=carga, name=esp, marker_color=C_SEQUENCE[i % len(C_SEQUENCE)]))
        
        cap_total = np.zeros(semanas_ano)
        for esp in filtro_global:
            hr_efec_wt = hh_nominales * (wt_dict[esp] / 100.0)
            hr_efec_prog = hr_efec_wt * (1 - noprog_dict[esp] / 100.0)
            hc_eval = hc_dict[esp] if is_base else hc_optimo_vista_dict[esp]
            cap_total += np.full(semanas_ano, hc_eval * hr_efec_prog)
            
        fig.add_trace(go.Scatter(x=list(range(1, semanas_ano + 1)), y=cap_total, mode='lines', name='Capacidad', line=dict(color=C_BLACK, width=2, dash='dash')))
        
        fig.update_layout(barmode='stack', xaxis_title="Semana", yaxis_title="Horas Hombre", template="plotly_white", margin=dict(t=10, l=10, r=10), yaxis=dict(range=[0, y_max_fijo]), xaxis=dict(tickmode='array', tickvals=tickvals), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.8)"))
        return fig

    def render_utilization_chart(is_base):
        fig = go.Figure()
        nombres, util_pct, ocio_pct, sob_pct = [], [], [], []
        
        for esp in filtro_global:
            carga_eval = calendario_base[esp][0:semanas_ano] if is_base else calendario_dict[esp][0:semanas_ano]
            hr_efec = hh_nominales * (wt_dict[esp] / 100.0)
            hr_efec_prog = hr_efec * (1 - noprog_dict[esp] / 100.0)
            hc_eval = hc_dict[esp] if is_base else hc_optimo_vista_dict[esp]
            cap_efec_sem = hc_eval * hr_efec_prog
            
            if cap_efec_sem > 0:
                utilizado = sum(min(c, cap_efec_sem) for c in carga_eval)
                ocioso = sum(max(0, cap_efec_sem - c) for c in carga_eval)
                terceros = sum(max(0, c - cap_efec_sem) for c in carga_eval)
                total_disp = cap_efec_sem * semanas_ano 
                
                u_p = (utilizado / total_disp) * 100
                o_p = max(0, 100 - u_p)
                s_p = (terceros / total_disp) * 100 
                
                util_pct.append(u_p)
                ocio_pct.append(o_p)
                sob_pct.append(s_p)
                nombres.append(esp)
                    
        if nombres:
            fig.add_trace(go.Bar(y=nombres, x=util_pct, name='Utilización', orientation='h', marker=dict(color=C_BLUE), texttemplate='%{x:.0f}%', textposition='inside', insidetextanchor='middle'))
            fig.add_trace(go.Bar(y=nombres, x=ocio_pct, name='Ociosidad', orientation='h', marker=dict(color=C_GREY), texttemplate='%{x:.0f}%', textposition='inside', insidetextanchor='middle', textfont=dict(color=C_BLACK)))
            fig.add_trace(go.Bar(y=nombres, x=sob_pct, name='Sobrecarga (3eros)', orientation='h', marker=dict(color=C_LIME), texttemplate='%{x:.0f}%', textposition='outside', textfont=dict(color=C_BLACK)))
            fig.add_shape(type="line", x0=100, x1=100, y0=-0.5, y1=len(nombres)-0.5, line=dict(color=C_BLACK, width=2, dash="dash"))

        fig.update_layout(barmode='stack', template="plotly_white", margin=dict(t=10, l=150, b=10, r=50), xaxis=dict(showticklabels=False, range=[0, max(120, max(util_pct + np.array(ocio_pct) + np.array(sob_pct)) if util_pct else 100)]), legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="right", x=1), height=max(150, len(nombres)*60))
        return fig

    with tab1:
        st.markdown(f"<h3 style='color: {C_NAVY};'>Estructura de Headcount (Manual)</h3>", unsafe_allow_html=True)
        st.markdown(render_kpi_cards(is_base=True), unsafe_allow_html=True)
        
        st.markdown(f"<h3 style='color: {C_NAVY}; margin-top:20px;'>Perfil de Demanda (Caso Base)</h3>", unsafe_allow_html=True)
        st.plotly_chart(render_profile_chart(is_base=True), use_container_width=True)
        
        st.markdown(f"<h3 style='color: {C_NAVY}; margin-top:20px;'>Balance Operativo (Real)</h3>", unsafe_allow_html=True)
        st.plotly_chart(render_utilization_chart(is_base=True), use_container_width=True)

    with tab2:
        st.markdown(f"<h3 style='color: {C_NAVY};'>Estructura de Headcount (Óptimo)</h3>", unsafe_allow_html=True)
        st.markdown(render_kpi_cards(is_base=False), unsafe_allow_html=True)
        
        st.markdown(f"<h3 style='color: {C_NAVY}; margin-top:20px;'>Perfil de Demanda (Optimizado)</h3>", unsafe_allow_html=True)
        filtro_fase = st.selectbox("Fase", ["Fase 3", "Fase 2", "Fase 1"])
        st.plotly_chart(render_profile_chart(is_base=False, f_fase=filtro_fase), use_container_width=True)
        
        st.markdown(f"<h3 style='color: {C_NAVY}; margin-top:20px;'>Balance Operativo (Optimizado)</h3>", unsafe_allow_html=True)
        st.plotly_chart(render_utilization_chart(is_base=False), use_container_width=True)

    with tab3:
        st.markdown(f"<h3 style='color: {C_NAVY};'>Contraste de Curvas (Base vs Optimizado)</h3>", unsafe_allow_html=True)
        
        fig_comp = go.Figure()
        carga_base_total = np.zeros(semanas_ano)
        
        for esp in filtro_global:
            carga_base_total += calendario_base[esp][0:semanas_ano]
            
        fig_comp.add_trace(go.Scatter(x=list(range(1, semanas_ano + 1)), y=carga_base_total, mode='lines', name='Caso Base (Caótico)', line=dict(color=C_GREY, width=2, dash='dot'), fill='tozeroy', fillcolor='rgba(209, 221, 230, 0.4)'))
        
        for i, esp in enumerate(filtro_global):
             fig_comp.add_trace(go.Bar(x=list(range(1, semanas_ano + 1)), y=calendario_dict[esp][0:semanas_ano], name=f'{esp} (Opt)', marker_color=C_SEQUENCE[i % len(C_SEQUENCE)]))
        
        fig_comp.update_layout(barmode='stack', xaxis_title="Semana", yaxis_title="Horas Hombre Totales", template="plotly_white", margin=dict(t=10, l=10, r=10), xaxis=dict(tickmode='array', tickvals=tickvals), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.8)"))
        st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown(f"<h3 style='color: {C_NAVY}; margin-top:30px;'>Impacto Económico (Año 1)</h3>", unsafe_allow_html=True)
        
        impact_html = "<div class='kpi-container'>"
        for esp in especialidades_unicas:
            if esp not in filtro_global:
                impact_html += f"<div class='kpi-card kpi-card-inactive'><div class='kpi-disc-title' style='color:{C_GREY};'>{esp}</div><div class='kpi-stat-group'><span class='kpi-stat-sub' style='color:{C_GREY};'>Var. USD</span><span class='kpi-stat-val-dash'>-</span></div><div class='kpi-stat-group'><span class='kpi-stat-sub' style='color:{C_GREY};'>Var. HC Interno</span><span class='kpi-stat-val-dash'>-</span></div></div>"
                continue
                
            carga_base_y1 = calendario_base[esp][0:semanas_ano]
            carga_opt_y1 = calendario_dict[esp][0:semanas_ano]
            
            hr_efec = hh_nominales * (wt_dict[esp] / 100.0)
            hr_efec_prog = hr_efec * (1 - noprog_dict[esp] / 100.0)
            
            hc_base = hc_dict[esp]
            hc_opt = hc_optimo_vista_dict[esp]
            
            cap_base_efec = hc_base * hr_efec_prog
            cap_base_nom = hc_base * hh_nominales
            cap_opt_efec = hc_opt * hr_efec_prog
            cap_opt_nom = hc_opt * hh_nominales
            
            c_int_base, c_ext_base = calcular_costo(carga_base_y1, cap_base_efec, cap_base_nom, semanas_ano, tarifa_interna, tarifa_externa)
            c_int_opt, c_ext_opt = calcular_costo(carga_opt_y1, cap_opt_efec, cap_opt_nom, semanas_ano, tarifa_interna, tarifa_externa)
            
            variacion_usd = (c_int_opt + c_ext_opt) - (c_int_base + c_ext_base)
            
            hh_ext_base = sum([max(0, c - cap_base_efec) for c in carga_base_y1])
            hh_ext_opt = sum([max(0, c - cap_opt_efec) for c in carga_opt_y1])
            variacion_hh_ext = hh_ext_opt - hh_ext_base
            
            variacion_hc_int = hc_opt - hc_base
            
            color_variacion = C_LIME if variacion_usd <= 0 else C_NAVY
            signo_usd = "" if variacion_usd <= 0 else "+"
            signo_hc = "" if variacion_hc_int <= 0 else "+"
            signo_hh = "" if variacion_hh_ext <= 0 else "+"
            
            impact_html += f"""
            <div class='kpi-card'>
                <div class='kpi-disc-title'>{esp}</div>
                <div class='kpi-stat-group'><span class='kpi-stat-sub'>Var. USD</span><span class='kpi-stat-val' style='color:{color_variacion};'>{signo_usd}${variacion_usd:,.0f}</span></div>
                <div class='kpi-stat-group'><span class='kpi-stat-sub'>Var. HC Interno</span><span class='kpi-stat-val'>{signo_hc}{variacion_hc_int} Técnicos</span></div>
                <div class='kpi-stat-group'><span class='kpi-stat-sub'>Var. HH Terceros</span><span class='kpi-stat-val'>{signo_hh}{variacion_hh_ext:,.0f} HH</span></div>
            </div>"""
        impact_html += "</div>"
        st.markdown(impact_html, unsafe_allow_html=True)
            
        st.markdown(f"<h3 style='color: {C_NAVY}; margin-top:30px;'>Matriz de Asignación Temporal Panorámica (Año 1)</h3>", unsafe_allow_html=True)
        
        if not filtro_global:
            st.info("Sin datos para mostrar.")
        else:
            df_filtered = df_output[df_output['Disciplina'].isin(filtro_global)]
            df_ventana = df_filtered[(df_filtered['Planned_Date_Week'] > 0) & (df_filtered['Planned_Date_Week'] <= semanas_ano)]
            
            if not df_ventana.empty:
                pivot_cal = df_ventana.pivot_table(index=['ID Tarea', 'TAG', 'Disciplina', 'Frecuencia', 'Tolerancia'], columns='Planned_Date_Week', values='Estimated_Work_Hours', aggfunc='sum').fillna('')
                for w in range(1, semanas_ano + 1):
                    if w not in pivot_cal.columns: pivot_cal[w] = ''
                pivot_cal = pivot_cal[list(range(1, semanas_ano + 1))]
                
                def color_cells(val): return f'background-color: {C_BLUE}; color: {C_WHITE}; font-weight: bold; text-align: center;' if val != '' else ''
                st.dataframe(pivot_cal.style.map(color_cells).format(precision=0), use_container_width=True, height=250)

else:
    st.info("Carga tu plan de mantenimiento en el panel lateral para visualizar los escenarios.")
