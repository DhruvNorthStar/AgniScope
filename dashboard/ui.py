"""Shared responsive scientific-dashboard presentation, with no external artwork."""
import streamlit as st
from dashboard import data,ROOT
from dashboard.content import COLORS,LIMITATION,OPPORTUNITY,FREQUENCY_COLORS,FORMAL_TITLE

def style():
    st.markdown('''<style>
    .stApp {background:#F8FAFC;color:#17324D}
    .stApp,[data-testid="stMarkdownContainer"] p,[data-testid="stWidgetLabel"],input,button,[data-testid="stMetricValue"] {font-family:'Segoe UI',Inter,system-ui,sans-serif!important}
    .block-container {max-width:1480px;padding:3.8rem 2.5rem 2.5rem}
    h1,h2,h3 {font-family:'Segoe UI',Inter,system-ui,sans-serif!important;letter-spacing:-.035em;color:#17324D!important;font-weight:600!important}
    h1 {font-size:2.1rem!important;line-height:1.18!important;max-width:960px;padding-bottom:.45rem!important}
    h2 {font-size:1.45rem!important;margin-top:.75rem}
    h3 {font-size:1.1rem!important;letter-spacing:-.015em}
    p,li {line-height:1.65}
    [data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] p {color:#4F6375!important;opacity:1!important;line-height:1.5}
    [data-testid="stSidebar"] {background:#16324A;border-right:1px solid #234158}
    [data-testid="stSidebar"] * {color:#D9E6EA}
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {color:#D9E6EA!important}
    [data-testid="stSidebar"] [data-testid="stAlert"] * {color:#543B2F}
    [data-testid="stSidebarContent"] {padding-top:.5rem}
    [data-testid="stLogo"] {max-width:218px!important;height:42px!important;object-fit:contain}
    [data-testid="stSidebarNav"] {padding-top:.4rem}
    [data-testid="stSidebarNavLink"] {padding:.55rem .7rem;border-left:2px solid transparent;transition:none}
    [data-testid="stSidebarNavLink"][aria-current="page"] {background:#234659;border-left:2px solid #83C6B5;border-radius:0 5px 5px 0}
    [data-testid="stSidebarNavLink"]:hover {background:#1D3D52}
    [data-testid="stSidebarNavLink"] span {white-space:normal!important;overflow:visible!important;line-height:1.35}
    [data-testid="stSidebarNavLink"] p {white-space:normal!important;overflow:visible!important;line-height:1.4}
    [data-testid="stSidebarNavLink"] {height:auto!important;min-height:30px}
    [data-testid="stAppDeployButton"] {display:none}
    [data-testid="stMetric"] {background:#FFFFFF;border:1px solid #D8E1E8;border-top:2px solid #1C8C8C;border-radius:6px;padding:.8rem .8rem 1rem;min-height:106px}
    [data-testid="stMetricValue"] {font-weight:600;color:#17324D;font-size:clamp(1.25rem,2vw,2rem);letter-spacing:-.04em;white-space:normal;overflow:visible}
    [data-testid="stMetricValue"] div {white-space:normal!important;overflow:visible!important;text-overflow:clip!important}
    [data-testid="stMetricLabel"] {color:#4F6375;font-size:.85rem}
    [data-testid="stMetricLabel"] p {white-space:normal!important;overflow:visible!important;text-overflow:clip!important;line-height:1.35}
    [data-testid="stVerticalBlockBorderWrapper"]>div {border-color:#D8E1E8;border-radius:8px}
    [data-testid="stExpander"] {background:#FFFFFF;border-color:#D8E1E8;border-radius:6px}
    [data-testid="stExpander"] summary {font-size:.9rem}
    [data-testid="stSelectbox"] label,[data-testid="stMultiSelect"] label {font-weight:600}
    [data-testid="stDeckGlJsonChart"] {border:1px solid #D8E1E8;border-radius:8px;overflow:hidden;box-shadow:0 3px 12px #17324D08}
    .eyebrow {color:#1C8C8C;font-size:.72rem;letter-spacing:.1em;font-weight:650;text-transform:uppercase;margin-bottom:.6rem}
    .hero-note {font-size:1rem;color:#4F6375;line-height:1.65;max-width:950px;margin-bottom:.6rem}
    .science-note {border-left:2px solid #D97745;padding:.55rem .85rem;background:#FFF6EE;color:#6B4D36;font-size:.88rem;line-height:1.6}
    .section-note {color:#4F6375;font-size:.9rem;margin-top:-.3rem;margin-bottom:.65rem}
    .workflow {display:flex;flex-wrap:wrap;gap:.65rem;margin:.8rem 0 1.5rem}
    .workflow span {flex:1;min-width:125px;border-top:2px solid #1C8C8C;background:#fff;padding:.8rem;font-size:.88rem;color:#17324D}
    .workflow b {display:block;color:#1C8C8C;font-size:.7rem;letter-spacing:.08em;margin-bottom:.3rem}
    hr {border-color:#D8E1E8}
    @media(max-width:760px) {.block-container{padding:3.8rem 1rem 1.5rem}h1{font-size:1.7rem!important}[data-testid="stMetric"]{min-height:88px}[data-testid="stMetricValue"]{font-size:1.5rem}}
    </style>''',unsafe_allow_html=True)

def heading(title,subtitle,section):
    st.markdown(f'<div class="eyebrow">{section} · Uttarakhand · 2019–2025</div>',unsafe_allow_html=True)
    st.title(title);st.markdown(f'<div class="hero-note">{subtitle}</div>',unsafe_allow_html=True)


def section(title,description=None):
    st.subheader(title)
    if description:st.markdown(f'<div class="section-note">{description}</div>',unsafe_allow_html=True)

def caution(text=LIMITATION):st.markdown('<div class="science-note">'+text+'</div>',unsafe_allow_html=True)

def metrics(items):
    for column,(label,value,help_text) in zip(st.columns(len(items)),items):
        with column:st.metric(label,value,help=help_text)

def build_chart(frame,x,y,*,kind="bar",color=None,height=270,x_title=None,y_title="Thermal observations",tooltip=None,horizontal=False):
    """Build a display specification without sampling or mutating the population."""
    import altair as alt
    base=alt.Chart(frame)
    palette=[COLORS['teal'],COLORS['orange'],'#81739A','#7297A2']
    if kind=="line":base=base.mark_line(point=alt.OverlayMarkDef(size=45,filled=True),strokeWidth=2.3)
    else:base=base.mark_bar(cornerRadiusEnd=2,**({'size':18} if horizontal else {}))
    if horizontal:
        channels={"y":alt.Y(x,title=x_title,sort='-x',axis=alt.Axis(labelLimit=210,labelPadding=8)),"x":alt.X(y,title=y_title)}
    else:
        channels={"x":alt.X(x,title=x_title,axis=alt.Axis(labelAngle=0,labelPadding=8)),"y":alt.Y(y,title=y_title)}
    channels['tooltip']=tooltip or list(frame.columns)
    if color:
        legend_title={'min_samples':'Minimum samples','resolution_m':'Grid resolution'}.get(color.split(':')[0])
        channels["color"]=alt.Color(color,title=legend_title,scale=alt.Scale(range=palette))
        if kind!="line":channels['yOffset' if horizontal else 'xOffset']=alt.YOffset(color) if horizontal else alt.XOffset(color)
    else:channels['color']=alt.value(COLORS['teal'])
    return (base.encode(**channels).properties(height=height,background='transparent')
        .configure_view(stroke=None).configure_axis(labelFont='Segoe UI',titleFont='Segoe UI',labelFontSize=12,titleFontSize=11,
            labelColor='#4F6375',titleColor='#4F6375',titleFontWeight='normal',titlePadding=12,domain=False,ticks=False,gridColor='#D8E1E8',gridOpacity=.7)
        .configure_axisX(grid=horizontal).configure_axisY(grid=not horizontal)
        .configure_legend(labelFont='Segoe UI',labelFontSize=11,orient='bottom',padding=12,symbolType='circle',symbolSize=65))


def chart(frame,x,y,**kwargs):
    if frame.empty:st.info("No observations match this view.");return
    st.altair_chart(build_chart(frame,x,y,**kwargs),width="stretch")

def render_map(value,key,*,height=560):
    st.pydeck_chart(value,key=key,width="stretch",height=height,on_select="ignore")
    st.caption("Pan, zoom or hover to explore · district boundaries provide the offline map context · WGS84 coordinates.")

def sample_caption(displayed,total,kind="observations"):
    if displayed<total:
        st.caption(f"Map displays {displayed:,} of {total:,} {kind}: a bounded deterministic sample in source-ID order, not a spatially representative sample. All metrics and charts use the full selected population.")
    else:st.caption(f"Map displays all {total:,} selected {kind}. Metrics and charts use the full selected population.")

def frequency_legend(k=0):
    chips=[]
    for n in range(k,8):
        color=FREQUENCY_COLORS[n];hexcolor="#"+"".join(f"{c:02x}" for c in color[:3])
        chips.append(f'<span style="display:inline-flex;align-items:center;margin-right:16px;font-size:13px"><span style="background:{hexcolor};width:14px;height:14px;border-radius:2px;display:inline-block;margin-right:5px"></span>{n}/7</span>')
    st.markdown(''.join(chips),unsafe_allow_html=True)

def optional_image(key,caption):
    row=data.catalog()["sources"].get(key)
    if row and not (ROOT/row["path"]).exists():st.info("This optional static figure is unavailable. The saved scientific tables remain the source of the displayed results.");return
    st.image(str(data.source_file(key)),caption=caption,width="stretch")

def footer():
    st.divider();st.caption(FORMAL_TITLE)
    st.caption("NASA FIRMS VIIRS_SNPP_SP · Standard Processing · approved 2021 district-union footprint · WorldCover 2021 is contextual. "+OPPORTUNITY)
