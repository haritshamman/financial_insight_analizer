import streamlit as st

import os
import requests
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Load Environment ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SECTORS_API_KEY = os.getenv("SECTORS_API_KEY")

BASE_URL = "https://api.sectors.app/v1"
HEADERS = {"Authorization": SECTORS_API_KEY}

# --- Init LLM ---
llm = ChatGroq(
    temperature=0.7,
    model_name="llama-3.3-70b-versatile",
    groq_api_key=GROQ_API_KEY
)

# ===================== UTILS ===================== #
def fetch_data(endpoint: str, params: dict = None):
    url = f"{BASE_URL}/{endpoint}"
    resp = requests.get(url, headers=HEADERS, params=params)
    resp.raise_for_status()
    return resp.json()

def run_llm(prompt_template: str, data: pd.DataFrame):
    prompt = PromptTemplate.from_template(prompt_template).format(data=data.to_string(index=False))
    return llm.invoke(prompt).content

def clean_python_code(raw_code: str):
    return raw_code.strip().strip("```").replace("python", "").strip()

# ===================== SECTIONS ===================== #
def sidebar_selector():

    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80)
    st.sidebar.title("📊 Financial Dashboard")
    st.sidebar.markdown("Financial Insight Analyzer.")

    subsectors = fetch_data("subsectors/")
    subsector_list = pd.DataFrame(subsectors)["subsector"].sort_values().tolist()
    selected_subsector = st.sidebar.selectbox("🔽 Pilih Subsector", subsector_list)

    companies = fetch_data("companies/", params={"sub_sector": selected_subsector})
    companies_df = pd.DataFrame(companies)
    company_options = companies_df["symbol"] + " - " + companies_df["company_name"]
    selected_company = st.sidebar.selectbox("🏢 Pilih Perusahaan", company_options)

    st.sidebar.markdown("---")
    st.sidebar.info("Powered by Harts Consulting")
    return selected_company.split(" - ")[0]

def financial_summary(symbol: str):
    financials = pd.DataFrame(fetch_data(f"financials/quarterly/{symbol}/",
                                         params={"n_quarters": "4",
                                                "report_date": "2025-06-30"}))
    financials['date'] = pd.to_datetime(financials['date'])
    financials['date'] = financials['date'].dt.strftime('%m-%Y')
    financials['DER']   = financials['total_liabilities'] / financials['total_equity']
    financials['asset_turnover']  = financials['revenue'] / financials['total_assets']

    prompt = """
    Anda berperan sebagai data analyst expert dalam bidang investasi keuangan dan memiliki keahlian dalam menganalisis laporan keuangan perusahaan publik.
    Anda juga memiliki pengalaman sebagai konsultan investasi untuk klien institusional.
    Anda memiliki kemampuan untuk menyajikan informasi keuangan yang kompleks dengan cara yang mudah dipahami oleh investor.
    Anda diminta untuk melakukan analisis mendalam terhadap laporan keuangan kuartalan perusahaan publik dan menyusun ringkasan eksekutif yang jelas dan ringkas
    berdasarkan data keuangan kuartalan berikut (dalam miliar Rupiah):

    {data}
    Buatlah summary mengenai laporan keuangan kuartalan tersebut dengan memperhatikan hal-hal berikut:
1. Soroti metrik keuangan utama
2. Buat analisa menggunakan rasio: 
DER (Debt to Equity Ratio) yaitu kolom total_liabilities dibagi dengan kolom total_equity
Asset Turnover Ratio yaitu kolom revenue dibagi dengan kolom  total_assets
Dari rasio rasio tersebut jelaskan apa maksudnya bagi perusahaan.
3. Identifikasi tren atau perubahan signifikan dibandingkan kuartal sebelumnya.
4. Jelaskan setiap indikator keuangan yang Anda soroti dan mengapa itu penting bagi investor.
5. Berikan wawasan tentang faktor-faktor yang mungkin mempengaruhi kinerja keuangan perusahaan.
6. Sajikan informasi ini dalam format yang mudah dipahami oleh investor, seperti poin-poin ringkas atau tabel.
7. Ringkasan harus ditulis dalam bahasa Indonesia.
8. Gunakan istilah yang mudah dimengerti oleh investor awam.
9. Berikan rekomendasi investasi berdasarkan analisis Anda.
10. Buat kesimpulan yang menyoroti kekuatan dan potensi risiko investasi di perusahaan ini.

    """
    summary = run_llm(prompt, financials)

    with st.container():
        st.subheader("💡 Ringkasan Keuangan")
        st.markdown(summary, unsafe_allow_html=True)

    return financials

def revenue_trend(symbol: str, financials: pd.DataFrame):
    data_sample = financials[['date','revenue','total_liabilities','total_equity','total_assets','DER','asset_turnover']]

    prompt = f"""
Buatkan kode visualisasi plotly python untuk menampilkan insight keuangan berikut:
- Tampilkan dalam 1 kanvas (menggunakan subplot) dengan 3 bagian:
    1. Stacked bar chart proporsi total_assets, total_liabilities, total_equity per kuartal (row 1).
    2. Line chart tren DER & asset_turnover (row 2).
    3. Dual axis chart: revenue (bar, y1) & DER (line, y2) (row 3).
- Gunakan struktur subplot, parameter, dan urutan kode **persis** seperti contoh berikut:

# Pastikan kolom tanggal sudah dalam format datetime
financials['date'] = pd.to_datetime(financials['date'])

# Membuat subplot 3 baris 1 kolom, dengan yaxis kedua di subplot ke-3
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=[
        "Proporsi Total Assets, Liabilities, dan Equity per Kuartal",
        "Tren DER & Asset Turnover",
        "Revenue vs DER"
    ],
    specs=[[{{"type": "bar"}}],
           [{{"type": "scatter"}}],
           [{{"secondary_y": True}}]]
)

# 1. Stacked Bar Chart
fig.add_trace(go.Bar(
    x=financials['date'],
    y=financials['total_liabilities'],
    name='Total Liabilities'
), row=1, col=1)
fig.add_trace(go.Bar(
    x=financials['date'],
    y=financials['total_equity'],
    name='Total Equity'
), row=1, col=1)
fig.add_trace(go.Bar(
    x=financials['date'],
    y=financials['total_assets'],
    name='Total Assets'
), row=1, col=1)

# 2. Line Chart DER & Asset Turnover
fig.add_trace(go.Scatter(
    x=financials['date'],
    y=financials['DER'],
    mode='lines+markers',
    name='DER'
), row=2, col=1)
fig.add_trace(go.Scatter(
    x=financials['date'],
    y=financials['asset_turnover'],
    mode='lines+markers',
    name='Asset Turnover'
), row=2, col=1)

# 3. Dual Axis: Revenue (bar) & DER (line)
fig.add_trace(go.Bar(
    x=financials['date'],
    y=financials['revenue'],
    name='Revenue',
    marker_color='rgba(55, 83, 109, 0.7)'
), row=3, col=1, secondary_y=False)
fig.add_trace(go.Scatter(
    x=financials['date'],
    y=financials['DER'],
    name='DER',
    mode='lines+markers',
    marker_color='crimson'
), row=3, col=1, secondary_y=True)

# Layout dan axis
fig.update_layout(
    barmode='stack',
    height=1200,
    title_text="Insight Visualisasi Keuangan: Assets, Liabilities, Equity, Revenue, DER, Asset Turnover",
    legend_title='Metrik',
    template='plotly_white'
)
fig.update_xaxes(title_text="Tanggal Laporan", row=3, col=1)
fig.update_yaxes(title_text="Nilai (Miliar Rupiah)", row=1, col=1)
fig.update_yaxes(title_text="Rasio", row=2, col=1)
fig.update_yaxes(title_text="Revenue (Miliar Rupiah)", row=3, col=1, secondary_y=False)
fig.update_yaxes(title_text="DER", row=3, col=1, secondary_y=True)
fig.show()

Gunakan data berikut:
{data_sample}

Tulis HANYA kode Python yang bisa langsung dieksekusi. Jangan sertakan penjelasan apapun.
    """
    code = clean_python_code(llm.invoke(prompt).content)

    with st.container():
        st.subheader("📊 Visualisasi Tren Keuangan")
        exec_locals = {}
        exec(code, {}, exec_locals)
        st.plotly_chart(exec_locals["fig"], use_container_width=True)

def trend_analysis(financials: pd.DataFrame):
    prompt = """
    Anda beerperan sebagai seorang data analyst dari mckinsey yang memiliki keahlian dalam menganalisis laporan keuangan perusahaan publik.
    Anda juga memiliki pengalaman sebagai konsultan investasi untuk klien institusional.
    Berdasarkan data kuartalan berikut:
    {data}
    Berikan insight mendalam tentang kondisi keuangan perusahaan berdasarkan visualisasi tersebut kemudian berikan rekomendasi apa yang harus diperbaiki atau harus ditingkatkan oleh perusahaan
    dan gunakan bahasa yang bisa di mengerti dan di eksekusi oleh manajemen perusahaan.
    """
    analysis = run_llm(prompt, financials)
    with st.container():
        st.subheader("🔎 Interpretasi Tren Keuangan")
        st.markdown(analysis, unsafe_allow_html=True)

def risk_analysis(financials: pd.DataFrame):
    prompt = """
    Anda adalah berperan sebagai seorang  analis risiko keuangan dan audit internal yang skeptis.
    Tugas anda andalah mencari celah potensi kerugian perusahaan atau indikator red flag yang bisa menyebabkan perusahaan mengalami penurunan / kerugian di masa mendatang.
    Periksa data keuangan berikut dengan teliti:
    {data}
    Indentifikasi 5 potensi risiko atau "red flags" yang perlu diwaspadai dari data tersebut. 
    Jelaskan secara detail mengapa hal tersebut menjadi risiko bagi perusahaan dan langsung kepada penjelasan poin poinnya.
    """
    risks = run_llm(prompt, financials)
    with st.container():
        st.subheader("⚠️ Potensi Risiko Keuangan")
        st.markdown(risks, unsafe_allow_html=True)

# ===================== MAIN APP ===================== #
def main():
    st.set_page_config(
        page_title="Financial LLM Dashboard",
        page_icon=":bar_chart:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown(
        """
        <style>
        .main {background-color: #f8fafc;}
        .block-container {padding-top: 2rem;}
        .stExpander > div > div {background: #f0f4f8;}
        </style>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1, 6])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=90)
    with col2:
        st.title("Financial LLM Dashboard")
        st.markdown(
            "<span style='font-size:18px;'>AI & LLM-based Financial Statement Analysis of Indonesian Public Companies.</span>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    symbol = sidebar_selector()

    if st.sidebar.button("🔍 Lihat Insight"):
        with st.spinner("Mengambil dan menganalisis data..."):
            financials = financial_summary(symbol)
            st.markdown("---")
            revenue_trend(symbol, financials)
            st.markdown("---")
            colA, colB = st.columns(2)
            with colA:
                trend_analysis(financials)
            with colB:
                risk_analysis(financials)
        st.success("Analisis selesai!")

if __name__ == "__main__":
    main()