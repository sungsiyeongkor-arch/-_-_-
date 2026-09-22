# -*- coding: utf-8 -*-
"""
한국 반도체 수출 구조 분석 및 수출액 예측 · Streamlit 앱
=======================================================
교수님 스타터 뼈대(데이터 로딩·레이아웃·다운로드)를 재사용하고,
시각화는 우리 EDA(시간·품목·지역·가격)로 교체, 예측 탭에 SARIMA를 통합했다.

실행:  pip install -r requirements.txt  →  streamlit run app.py
데이터: data/ 폴더에 clean_file1/2/3 CSV를 둔다.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ============================================================
# 1. 팀별로 수정할 부분
# ============================================================
PROJECT_INFO = {
    "title": "한국 반도체 수출 구조 분석 및 수출액 예측",
    "subtitle": "시간 · 지역 · 품목 3축 접근",
    "team": "OOO 팀 (성시영 외)",
    "question": "메모리·시스템·개별소자 수출은 시간·지역·품목에 따라 어떻게 다르게 움직이는가?",
}

NAVY="#1B2A4A"; COPPER="#E0952E"; STEEL="#4E6E9E"; GRAY="#B8C1CE"; GREEN="#2E8B6F"; RED="#C0392B"
TPL = "plotly_white"

# ============================================================
# 2. 데이터 로딩
# ============================================================
APP_DIR = Path(__file__).resolve().parent
def _find(name):
    for base in (APP_DIR/"data", APP_DIR):
        p = base/name
        if p.exists():
            return p
    return None

@st.cache_data
def load_csv(name):
    p = _find(name)
    if p is None:
        return None
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(p, encoding=enc)
        except UnicodeDecodeError:
            continue
    return None

def load_all():
    f2 = load_csv("clean_file2_export_trend_monthly.csv")
    f3 = load_csv("clean_file3_materials_trade_by_country.csv")
    f1 = load_csv("clean_file1_industry_trend_yearly.csv")
    if f2 is not None:
        f2 = f2.copy(); f2["date"] = pd.to_datetime(f2["date"])
    return f1, f2, f3

# ============================================================
# 3. EDA 차트 함수 (순수 plotly)
# ============================================================
def fig_trend(f2):
    s = f2[f2["품목"] == "반도체"].sort_values("date").copy()
    s["MA12"] = s["수출액_억불"].rolling(12, center=True).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s["date"], y=s["수출액_억불"], name="월별 수출", line=dict(color="#9AB0CC", width=1.2)))
    fig.add_trace(go.Scatter(x=s["date"], y=s["MA12"], name="12개월 이동평균", line=dict(color=NAVY, width=3)))
    fig.update_layout(template=TPL, title="반도체 수출 추세와 사이클 (억불)", yaxis_title="월 수출액(억불)", legend_title="")
    return fig

def fig_yoy(f2):
    s = f2[f2["품목"] == "반도체"].sort_values("date")
    colors = [COPPER if v >= 0 else RED for v in s["YoY_증감률_pct"]]
    fig = go.Figure(go.Bar(x=s["date"], y=s["YoY_증감률_pct"], marker_color=colors))
    fig.update_layout(template=TPL, title="전년동월대비 증감률(YoY) — 메모리 사이클", yaxis_title="YoY(%)")
    return fig

def fig_season(f2):
    s = f2[f2["품목"] == "반도체"].set_index("date").sort_index()["수출액_억불"]
    dev = s - s.rolling(12, center=True).mean()
    me = dev.groupby(dev.index.month).mean()
    fig = go.Figure(go.Bar(x=[f"{m}월" for m in me.index], y=me.values,
                           marker_color=[GREEN if v >= 0 else RED for v in me.values]))
    fig.update_layout(template=TPL, title="계절 패턴 (평균 대비 편차, 억불)", yaxis_title="편차(억불)")
    return fig

def fig_composition(f2):
    d = f2[f2["레벨"] == "대분류"].sort_values("date")
    fig = px.area(d, x="date", y="수출액_억불", color="품목", template=TPL,
                  color_discrete_map={"메모리": COPPER, "시스템 반도체": STEEL, "개별소자": GRAY},
                  title="품목 구성 — 누가 규모·사이클을 이끄는가")
    fig.update_layout(yaxis_title="월 수출액(억불)", legend_title="")
    return fig

def fig_positioning(f2):
    d = f2[f2["레벨"] == "대분류"]
    piv = d.pivot_table(index="date", columns="품목", values="수출액_억불")
    ann = piv.resample("YE").sum(); n = ann.index[-1].year - ann.index[0].year
    rows = [{"품목": g, "CAGR": ((ann[g].iloc[-1]/ann[g].iloc[0])**(1/n)-1)*100,
             "변동계수": piv[g].std()/piv[g].mean(), "규모": ann[g].iloc[-1]} for g in piv.columns]
    fig = px.scatter(pd.DataFrame(rows), x="CAGR", y="변동계수", size="규모", color="품목", text="품목",
                     template=TPL, size_max=60,
                     color_discrete_map={"메모리": COPPER, "시스템 반도체": STEEL, "개별소자": GRAY},
                     title="품목 성격 — 성장(가로) vs 변동성(세로)")
    fig.update_traces(textposition="top center")
    return fig

def fig_country_share(f3, year):
    c = (f3[(f3["구분"] == "수출") & (f3["연도"] == year)]
         .groupby("지역")["금액_억불"].sum().sort_values(ascending=False).head(8).reset_index())
    c["구분색"] = np.where(c["지역"].isin(["중국", "홍콩"]), "중화권", "기타")
    fig = px.bar(c, x="금액_억불", y="지역", orientation="h", color="구분색", template=TPL,
                 color_discrete_map={"중화권": COPPER, "기타": STEEL}, title=f"{year}년 국가별 반도체 수출 (억불)")
    fig.update_layout(yaxis=dict(categoryorder="total ascending"), legend_title="")
    return fig

def fig_diversification(f3):
    years = sorted(f3["연도"].unique()); rows = []
    for y in years:
        d = f3[(f3["구분"] == "수출") & (f3["연도"] == y)]; t = d["금액_억불"].sum()
        for label, cl in [("중국+홍콩", ["중국", "홍콩"]), ("베트남", ["베트남"]), ("미국", ["미국"])]:
            rows.append({"연도": y, "지역군": label, "비중": d[d["지역"].isin(cl)]["금액_억불"].sum()/t*100})
    fig = px.line(pd.DataFrame(rows), x="연도", y="비중", color="지역군", markers=True, template=TPL,
                  color_discrete_map={"중국+홍콩": COPPER, "베트남": STEEL, "미국": GREEN},
                  title="수출 대상국 다변화 추이 (비중 %)")
    fig.update_layout(yaxis_title="수출 비중(%)", legend_title="")
    return fig

def fig_price(f1):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=f1["연도"], y=f1["반도체_수출(억불)"], name="반도체 수출", line=dict(color=NAVY, width=3)), secondary_y=False)
    fig.add_trace(go.Scatter(x=f1["연도"], y=f1["DRAM_가격(달러)"], name="DRAM 가격", line=dict(color=COPPER, width=2.5)), secondary_y=True)
    fig.update_layout(template=TPL, title="DRAM 가격 ↔ 반도체 수출 (연도별)")
    fig.update_yaxes(title_text="수출(억불)", secondary_y=False)
    fig.update_yaxes(title_text="DRAM 가격(달러)", secondary_y=True)
    return fig

# ============================================================
# 4. 예측 (SARIMA) — 캐싱해 한 번만 계산
# ============================================================
@st.cache_data
def compute_forecast(f2):
    """SARIMA(0,1,0)(0,1,1,12)로 holdout 검증 + 향후 12개월 예측. statsmodels 필요."""
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except Exception:
        return {"error": "statsmodels"}
    y = f2[f2["품목"] == "반도체"].set_index("date").sort_index()["수출액_억불"].asfreq("MS")
    H = 12; order = (0, 1, 0); sorder = (0, 1, 1, 12)
    train, test = y.iloc[:-H], y.iloc[-H:]
    m = SARIMAX(np.log(train), order=order, seasonal_order=sorder,
                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    pred = np.exp(m.forecast(H).values)
    mape = float(np.mean(np.abs((test.values - pred) / test.values)) * 100)
    full = SARIMAX(np.log(y), order=order, seasonal_order=sorder,
                   enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    fc = full.get_forecast(H); fmean = np.exp(fc.predicted_mean.values); ci = np.exp(fc.conf_int().values)
    fidx = pd.date_range(y.index.max() + pd.offsets.MonthBegin(1), periods=H, freq="MS")
    table = pd.DataFrame({"연월": fidx.strftime("%Y-%m"), "예측(억불)": fmean.round(1),
                          "하한": ci[:, 0].round(1), "상한": ci[:, 1].round(1)})
    return dict(hist_x=list(y.index), hist_y=list(y.values), fx=list(fidx),
                fmean=list(fmean), flo=list(ci[:, 0]), fhi=list(ci[:, 1]), mape=mape, table=table)

def fig_forecast(res):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res["hist_x"], y=res["hist_y"], name="실제", line=dict(color=NAVY, width=2)))
    fx = res["fx"]
    fig.add_trace(go.Scatter(x=fx, y=res["fhi"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fx, y=res["flo"], line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(224,149,46,0.18)", name="95% 신뢰구간"))
    fig.add_trace(go.Scatter(x=[res["hist_x"][-1]] + list(fx), y=[res["hist_y"][-1]] + list(res["fmean"]),
                             name="예측(2026)", mode="lines+markers", line=dict(color=COPPER, width=2.5, dash="dash")))
    fig.update_layout(template=TPL, title="향후 12개월(2026) 반도체 수출 예측 — SARIMA", yaxis_title="월 수출액(억불)")
    return fig

# ============================================================
# 5. 화면
# ============================================================
def main():
    st.set_page_config(page_title=PROJECT_INFO["title"], page_icon="📊", layout="wide")
    st.title(PROJECT_INFO["title"])
    st.caption(f'{PROJECT_INFO["subtitle"]}  |  {PROJECT_INFO["team"]}')
    st.info(f'프로젝트 질문: {PROJECT_INFO["question"]}')

    f1, f2, f3 = load_all()
    if f2 is None or f3 is None or f1 is None:
        st.error("data/ 폴더에 clean_file1/2/3 CSV를 넣어 주세요.")
        st.stop()

    k1, k2, k3 = st.columns(3)
    total_2025 = f2[(f2["품목"] == "반도체") & (f2["date"].dt.year == 2025)]["수출액_억불"].sum()
    k1.metric("분석 기간(월별)", f'{f2["date"].min():%Y-%m} ~ {f2["date"].max():%Y-%m}')
    k2.metric("수출 대상국(File3)", f'{f3["지역"].nunique()}개국')
    k3.metric("2025 반도체 수출", f"{total_2025:,.0f}억불")

    t1, t2, t3, t4, t5 = st.tabs(["⏱ 시간 추이", "🧩 품목별", "🌏 국가별", "💵 가격 연동", "🔮 예측"])

    with t1:
        st.markdown("**반도체 수출은 장기 성장하지만 3~4년 주기로 크게 출렁이고, 계절성도 뚜렷하다.**")
        st.plotly_chart(fig_trend(f2), use_container_width=True)
        c1, c2 = st.columns(2)
        c1.plotly_chart(fig_yoy(f2), use_container_width=True)
        c2.plotly_chart(fig_season(f2), use_container_width=True)

    with t2:
        st.markdown("**규모·사이클은 대부분 메모리가 이끌고, 시스템반도체는 안정적으로 성장한다.**")
        st.plotly_chart(fig_composition(f2), use_container_width=True)
        st.plotly_chart(fig_positioning(f2), use_container_width=True)

    with t3:
        st.markdown("**소수 국가에 집중돼 있으나, 대중국·홍콩 의존은 완화되고 다변화되는 중이다.**")
        year = st.slider("연도 선택", int(f3["연도"].min()), int(f3["연도"].max()), int(f3["연도"].max()))
        st.plotly_chart(fig_country_share(f3, year), use_container_width=True)
        st.plotly_chart(fig_diversification(f3), use_container_width=True)

    with t4:
        st.markdown("**가격과 수출은 변화율로 중간 정도 연동되나, AI·HBM 이후 연결이 약해진다.**")
        st.plotly_chart(fig_price(f1), use_container_width=True)

    with t5:
        st.markdown("**최근 12개월 검증에서 SARIMA가 baseline을 크게 앞섰고, 2026년은 완만한 성장으로 예측된다.**")
        res = compute_forecast(f2)
        if res.get("error"):
            st.info("예측에는 statsmodels가 필요합니다. `pip install -r requirements.txt` 후 다시 실행하세요.")
        else:
            c1, c2 = st.columns(2)
            c1.metric("검증 정확도 MAPE", f"{res['mape']:.1f}%", help="최근 12개월 holdout · 계절 나이브 17.1% 대비 우수")
            c2.metric("2026 예측 평균", f"{np.mean(res['fmean']):.0f}억불/월")
            st.plotly_chart(fig_forecast(res), use_container_width=True)
            st.caption("※ 과거 패턴의 연장 예측 — 사이클 전환점(가격 급락 등)은 반영하지 못함")
            st.dataframe(res["table"], hide_index=True, use_container_width=True)

    st.divider()
    st.caption("전공세미나 · From Data to App · 한국 반도체 수출 분석")

if __name__ == "__main__":
    main()
