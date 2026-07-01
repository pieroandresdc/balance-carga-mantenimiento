import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
import io

# ==========================================
# CONFIGURACIÓN DE PÁGINA
# ==========================================
st.set_page_config(page_title="Optimizador de Carga de Mantenimiento", layout="wide")
st.title("Motor Heurístico de Balance de HH")

# ==========================================
# 1. TRANSFORMADOR DEFENSIVO (PARSER)
# ==========================================
@st.cache_data
def load_and_clean_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, header=0)
        else:
            df = pd.read_excel(uploaded_file, header=0)
    except Exception as e:
        st.error(f"Error en la ingesta: {e}")
        return pd.DataFrame()

    df.columns = [str(c).strip() for c in df.columns]
    
    if 'Criticality' in df.columns:
        df['Criticality'] = pd.to_numeric(df['Criticality'], errors='coerce').fillna(1).astype(int)
    if 'Maximum_Tolerance_Value' in df.columns:
        df['Maximum_Tolerance_Value'] = pd.to_numeric(df['Maximum_Tolerance_Value'], errors='coerce').fillna(0).astype(int)
    if 'Work' in df.columns:
        df['Work'] = pd.to_numeric(df['Work'], errors='coerce').fillna(0.0)
    if 'Frequency_Value' in df.columns:
        df['Frequency_Value'] = pd.to_numeric(df['Frequency_Value'], errors='coerce').fillna(52)

    df['Semanas_Intervalo'] = df['Frequency_Value']
    if 'Frequency_Unit' in df.columns and 'Average_Daily_Usage' in df.columns:
        mask_horas = df['Frequency_Unit'].str.lower().str.contains('hour', na=False)
        adu = pd.to_numeric(df['Average_Daily_Usage'], errors='coerce').fillna(24) 
        df.loc[mask_horas, 'Semanas_Intervalo'] = (df.loc[mask_horas, 'Frequency_Value'] / adu) / 7
        
    df['Semanas_Intervalo'] = np.maximum(1, df['Semanas_Intervalo'].round().astype(int))
    return df

# ==========================================
# 2. MOTORES DE CÁLCULO (CACHÉ ACTIVADO)
# ==========================================
@st.cache_data
def run_baseline_engine(df, anchor_seeds):
    """Simula una programación tradicional sin heurística (Offset = 0)."""
    HORIZONTE = 156
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
            intervalo_parada = int(match.group(1)) if match else 52 
            sem_inicio = anchor_seeds.get(anchor_id, 1) - 1 
            semanas_ejecucion = [sem_inicio + i * intervalo_parada for i in range(HORIZONTE // intervalo_parada + 1) if (sem_inicio + i * intervalo_parada) < HORIZONTE]
        else:
            # Programación ingenua: todo empieza en la primera semana disponible de su ciclo
            semanas_ejecucion = [i * frecuencia for i in range(HORIZONTE // frecuencia + 1) if (i * frecuencia) < HORIZONTE]

        for s_final in semanas_ejecucion:
            calendario_base[esp][s_final] += hh
            
    return calendario_base

@st.cache_data
def run_heuristic_engine(df, anchor_seeds):
    """Motor de aplanamiento iterativo."""
    HORIZONTE = 156
    especialidades = df['Labour'].dropna().unique()
    calendario = {esp: np.zeros(HORIZONTE) for esp in especialidades}
    output_records = []
    
    df_sorted = df.sort_values(
        by=['Criticality', 'Maximum_Tolerance_Value', 'Work'], 
        ascending=[False, True, False]
    )

    for idx, row in df_sorted.iterrows():
        esp = row['Labour']
        if pd.isna(esp): continue
        
        hh = row['Work']
        frecuencia = row['Semanas_Intervalo']
        tolerancia = row['Maximum_Tolerance_Value']
        anchor_id = str(row.get('Shutdown_Anchor_ID', '')).strip()
        op_mode = str(row.get('Operation Mode', '')).strip().upper()
        
        task_id = row.get('Task_ID', f"Fila_{idx + 2}")
        if pd.isna(task_id) or str(task_id).strip() == '':
            task_id = f"Fila_{idx + 2}"
        
        semanas_ejecucion = []
        tipo_ejecucion = 'Internal'
        
        if anchor_id and anchor_id.lower() != 'nan':
            match = re.search(r'^(\d+)W', anchor_id.upper())
            intervalo_parada = int(match.group(1)) if match else 52 
            sem_inicio = anchor_seeds.get(anchor_id, 1) - 1 
            semanas_teoricas = [sem_inicio + i * intervalo_parada for i in range(HORIZONTE // intervalo_parada + 1)]
            semanas_ejecucion = [s for s in semanas_teoricas if s < HORIZONTE]
            tipo_ejecucion = 'Contractor_Major' 
        else:
            mejor_offset = 0
            menor_carga_promedio = float('inf')
            
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
                'Task_ID': task_id,
                'Maintenance_Item_Tag': row.get('Location_Code', 'N/A'),
                'Work_Center': esp,
                'Planned_Date_Week': s_final + 1,
                'Estimated_Work_Hours': hh,
                'Headcount_Required': row.get('Labour_Required', 1),
                'Operation_Mode': op_mode,
                'Shutdown_Code': anchor_id if anchor_id and anchor_id.lower() != 'nan' else '',
                'Execution_Type': tipo_ejecucion
            })
            
    return calendario, pd.DataFrame(output_records)

def calcular_costo(carga_array, cap_disp, t_int, t_ext):
    costo_int = sum([min(c, cap_disp) for c in carga_array]) * t_int
    costo_ext = sum([max(0, c - cap_disp) for c in carga_array]) * t_ext
    return costo_int, costo_ext

def calcular_hc_optimo(carga_array, hr_efec, t_int, t_ext):
    costos = []
    for hc in range(0, 151):
        cap = hc * hr_efec
        c_int, c_ext = calcular_costo(carga_array, cap, t_int, t_ext)
        costos.append(c_int + c_ext)
    return np.argmin(costos)

# ==========================================
# 3. INTERFAZ REACTIVA Y FLUJO PRINCIPAL
# ==========================================
st.sidebar.header("1. Ingesta de Datos")
uploaded_file = st.sidebar.file_uploader("Sube la Plantilla de Input", type=['csv', 'xlsx'])

if uploaded_file:
    df_raw = load_and_clean_data(uploaded_file)
    
    if df_raw.empty:
        st.stop()
        
    especialidades_unicas = sorted(df_raw['Labour'].dropna().unique().tolist())
    
    st.sidebar.header("2. Filtro Global de Alcance")
    filtro_global = st.sidebar.selectbox("Segmentar Análisis por:", ["Toda la Planta"] + especialidades_unicas)
    
    anchors_detectados = [a for a in df_raw['Shutdown_Anchor_ID'].unique() if pd.notna(a) and str(a).strip().lower() != 'nan']
    anchor_seeds = {}
    
    if anchors_detectados:
        st.sidebar.header("3. Semillas de Paradas Mayores")
        for anchor in anchors_detectados:
            match = re.search(r'^(\d+)W', anchor.upper())
            seed_default = int(match.group(1)) if match else 1
            anchor_seeds[anchor] = st.sidebar.number_input(f"Inicio {anchor}", min_value=1, max_value=156, value=seed_default)

    st.sidebar.header("4. Capacidades por Disciplina")
    hc_dict = {}
    wt_dict = {}
    
    for esp in especialidades_unicas:
        st.sidebar.markdown(f"<div style='font-size:12px; font-weight:bold; color:#555;'>{esp}</div>", unsafe_allow_html=True)
        col1, col2 = st.sidebar.columns(2)
        hc_dict[esp] = col1.number_input("HC", min_value=0, value=5, step=1, key=f"hc_{esp}")
        wt_dict[esp] = col2.number_input("WT", min_value=1.0, value=15.0, step=1.0, key=f"wt_{esp}")

    st.sidebar.divider()
    tarifa_interna = st.sidebar.number_input("Tarifa Interna ($/hora)", value=30.0)
    tarifa_externa = st.sidebar.number_input("Tarifa Contratista ($/hora)", value=80.0)

    # --- EJECUCIÓN REACTIVA INMEDIATA (Caché absorbe la carga) ---
    calendario_dict, df_output = run_heuristic_engine(df_raw, anchor_seeds)
    calendario_base = run_baseline_engine(df_raw, anchor_seeds)
    
    # --- CÁLCULO FINANCIERO Y AISLAMIENTO MATEMÁTICO ---
    c_int_opt_total, c_ext_opt_total = 0, 0
    c_int_base_total, c_ext_base_total = 0, 0
    hc_optimo_total = 0
    hh_anuales_total = 0
    terceros_pico_lista = []
    terceros_pico_base_lista = []
    breakdown_data = []

    esps_a_evaluar = especialidades_unicas if filtro_global == "Toda la Planta" else [filtro_global]
    
    capacidad_grafico_total = np.zeros(52)
    carga_opt_grafico_dict = {}
    carga_base_grafico_dict = {}
    
    for esp in esps_a_evaluar:
        carga_esp_y1_opt = calendario_dict[esp][0:52]
        carga_esp_y1_base = calendario_base[esp][0:52]
        
        carga_opt_grafico_dict[esp] = calendario_dict[esp]
        carga_base_grafico_dict[esp] = calendario_base[esp]
        
        hr_efec = wt_dict[esp]
        hc_disp = hc_dict[esp]
        cap_disp = hc_disp * hr_efec
        
        capacidad_grafico_total += np.full(52, cap_disp)
        
        # Costos Escenario Optimizado
        c_int_opt, c_ext_opt = calcular_costo(carga_esp_y1_opt, cap_disp, tarifa_interna, tarifa_externa)
        c_int_opt_total += c_int_opt
        c_ext_opt_total += c_ext_opt
        
        # Costos Escenario Base (Tradicional)
        c_int_base, c_ext_base = calcular_costo(carga_esp_y1_base, cap_disp, tarifa_interna, tarifa_externa)
        c_int_base_total += c_int_base
        c_ext_base_total += c_ext_base
        
        hc_opt = calcular_hc_optimo(carga_esp_y1_opt, hr_efec, tarifa_interna, tarifa_externa)
        hc_optimo_total += hc_opt
        hh_anuales_total += np.sum(carga_esp_y1_opt)
        
        # Picos de Terceros
        pico_esp_opt = np.max(carga_esp_y1_opt)
        terc_req_opt = int(np.ceil(max(0, pico_esp_opt - cap_disp) / hr_efec)) if hr_efec > 0 else 0
        terceros_pico_lista.append(terc_req_opt)
        
        pico_esp_base = np.max(carga_esp_y1_base)
        terc_req_base = int(np.ceil(max(0, pico_esp_base - cap_disp) / hr_efec)) if hr_efec > 0 else 0
        terceros_pico_base_lista.append(terc_req_base)
        
        if filtro_global == "Toda la Planta":
            breakdown_data.append({
                "Disciplina": esp,
                "HH (Y1)": int(np.sum(carga_esp_y1_opt)),
                "HC Actual": hc_disp,
                "HC Óptimo": hc_opt,
                "Ahorro Generado": f"${((c_int_base + c_ext_base) - (c_int_opt + c_ext_opt)):,.0f}",
                "Pico Contratistas (Opt)": terc_req_opt
            })
    
    terceros_pico_global = sum(terceros_pico_lista)
    terceros_pico_base_global = sum(terceros_pico_base_lista)
    
    costo_total_opt = c_int_opt_total + c_ext_opt_total
    costo_total_base = c_int_base_total + c_ext_base_total
    ahorro_total = costo_total_base - costo_total_opt

    # --- PANEL DE KPIS ESTRATÉGICOS ---
    st.markdown(f"### Resultados Estratégicos: {filtro_global}")
    cols = st.columns(4)
    
    # KPIs Comparativos
    cols[0].metric("Demanda Anual (Año 1)", f"{hh_anuales_total:,.0f} HH")
    cols[1].metric("Costo Total (Optimizado)", f"${costo_total_opt:,.0f}", delta=f"-${ahorro_total:,.0f} vs Base", delta_color="inverse")
    cols[2].metric("Pico Terceros (Optimizado)", f"{terceros_pico_global} Técnicos", delta=f"{terceros_pico_global - terceros_pico_base_global} vs Base", delta_color="inverse")
    
    kpi_html = (
        "<div style='background-color: #d4edda; padding: 10px; border-radius: 5px; text-align: center; border: 1px solid #c3e6cb;'>"
        "<h4 style='color: #155724; margin:0;'>HC Propio Óptimo</h4>"
        f"<h2 style='color: #155724; margin:0; font-size: 32px;'>{hc_optimo_total} Técnicos</h2>"
        "<span style='color: #155724;'>Minimiza el Costo Total Anual</span>"
        "</div>"
    )
    cols[3].markdown(kpi_html, unsafe_allow_html=True)

    if filtro_global == "Toda la Planta":
        st.markdown("#### Desglose Matemático por Disciplina")
        st.dataframe(pd.DataFrame(breakdown_data), use_container_width=True)

    # --- CONTROL DE VENTANA DE TIEMPO ---
    st.divider()
    vista_year = st.radio("Ventana Rodante de Visualización:", ["Año 1 (Sem 1-52)", "Año 2 (Sem 53-104)", "Año 3 (Sem 105-156)"], horizontal=True)
    
    start_w = 0
    if "Año 2" in vista_year: start_w = 52
    if "Año 3" in vista_year: start_w = 104
    end_w = start_w + 52
    eje_x = list(range(start_w + 1, end_w + 1))

    # --- GRAFICA 1: CARGA BASE VS OPTIMIZADA ---
    st.subheader("1. Perfil Dinámico: Tradicional vs Aplanamiento Heurístico")
    fig_carga = go.Figure()
    
    # Línea del Escenario Base (Caótico)
    carga_base_total = np.sum(list(carga_base_grafico_dict.values()), axis=0)[start_w:end_w]
    fig_carga.add_trace(go.Scatter(
        x=eje_x, y=carga_base_total, mode='lines',
        name='Escenario Base (Sin Optimizar)',
        line=dict(color='rgba(150, 150, 150, 0.5)', width=2, dash='dot'),
        fill='tozeroy', fillcolor='rgba(150, 150, 150, 0.1)'
    ))
    
    # Barras del Escenario Optimizado
    for esp, carga in carga_opt_grafico_dict.items():
        fig_carga.add_trace(go.Bar(x=eje_x, y=carga[start_w:end_w], name=f"{esp} (Optimizado)"))
        
    # Capacidad Dura
    fig_carga.add_trace(go.Scatter(
        x=eje_x, y=capacidad_grafico_total, mode='lines',
        name='Capacidad Base Limitada',
        line=dict(color='red', width=3, dash='dash')
    ))

    fig_carga.update_layout(barmode='stack', xaxis_title="Semana Operativa", yaxis_title="Horas Hombre", template="plotly_white", margin=dict(t=20))
    st.plotly_chart(fig_carga, use_container_width=True)

    # --- GRAFICA 2: UTILIZACIÓN ---
    st.subheader("2. Evolución de Utilización de Plantilla (Optimizado)")
    carga_ventana_total = np.sum(list(carga_opt_grafico_dict.values()), axis=0)[start_w:end_w]
    
    cap_val = capacidad_grafico_total[0]
    utilizacion_pct = (carga_ventana_total / cap_val) * 100 if cap_val > 0 else np.zeros(52)
    media_utilizacion = np.mean(utilizacion_pct)

    fig_util = go.Figure()
    fig_util.add_trace(go.Scatter(
        x=eje_x, y=utilizacion_pct, mode='lines+markers', name='% Utilización',
        line=dict(color='#17a2b8', width=2), marker=dict(size=6)
    ))
    fig_util.add_trace(go.Scatter(
        x=eje_x, y=[100] * 52, mode='lines', name='Límite Saturación (100%)',
        line=dict(color='red', width=1, dash='dot')
    ))
    fig_util.add_trace(go.Scatter(
        x=eje_x, y=[media_utilizacion] * 52, mode='lines', name=f'Promedio ({media_utilizacion:.1f}%)',
        line=dict(color='green', width=2, dash='dash')
    ))
    
    fig_util.update_layout(
        xaxis_title="Semana Operativa", yaxis_title="% de Utilización (Capacidad Instalada)",
        template="plotly_white", margin=dict(t=20), yaxis=dict(rangemode='tozero')
    )
    st.plotly_chart(fig_util, use_container_width=True)
    
    # --- MATRIZ DE CALENDARIO ---
    st.subheader(f"3. Matriz de Asignación Temporal ({vista_year})")
    df_filtered = df_output if filtro_global == "Toda la Planta" else df_output[df_output['Work_Center'] == filtro_global]
    df_ventana = df_filtered[(df_filtered['Planned_Date_Week'] > start_w) & (df_filtered['Planned_Date_Week'] <= end_w)]
    
    if df_ventana.empty:
        st.warning("No hay tareas programadas para los filtros y el año seleccionados.")
    else:
        pivot_cal = df_ventana.pivot_table(
            index=['Task_ID', 'Maintenance_Item_Tag', 'Work_Center'],
            columns='Planned_Date_Week',
            values='Estimated_Work_Hours',
            aggfunc='sum'
        ).fillna(0)
        
        for w in eje_x:
            if w not in pivot_cal.columns:
                pivot_cal[w] = 0
        pivot_cal = pivot_cal[eje_x]
        
        sums = pivot_cal.sum()
        sum_df = pd.DataFrame(sums).T
        sum_df.index = pd.MultiIndex.from_tuples([('► SUMA TOTAL', 'H-H', 'GLOBAL')])
        
        pivot_cal_final = pd.concat([sum_df, pivot_cal]).replace(0, '')
        
        def color_cells(val):
            return 'background-color: #28a745; color: white; font-weight: bold; text-align: center;' if val != '' else ''
            
        st.dataframe(pivot_cal_final.style.map(color_cells).format(precision=1), use_container_width=True, height=500)

    # --- DESCARGA ---
    st.divider()
    st.subheader("Entregable ERP (Base de Datos Cruda)")
    csv_buffer = io.StringIO()
    df_output.to_csv(csv_buffer, index=False)
    st.download_button("Descargar Plan Optimizado (CSV)", data=csv_buffer.getvalue(), file_name="Plan_Mantenimiento_SAP.csv", mime="text/csv", type="primary")

else:
    st.info("Sube la plantilla por el panel lateral para iniciar el motor heurístico.")