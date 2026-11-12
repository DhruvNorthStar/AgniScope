"""One terminology contract shared by all six pages."""
APP_NAME="AgniScope"
FORMAL_TITLE="AgniScope: Recurrent Thermal Hotspots in Uttarakhand Using VIIRS Satellite Detections (2019–2025)"
PAGES=(
    ("Overview","overview.py","overview"),
    ("Historical Fire Explorer","explorer.py","explorer"),
    ("DBSCAN vs HDBSCAN","clustering.py","clustering"),
    ("Recurring Hotspot Map","recurrence.py","recurrence"),
    ("Scientific Validation & Sensitivity","sensitivity.py","sensitivity"),
    ("Research Findings, Methodology & Limitations","research.py","research"),
)
QUESTION="Where does observed satellite thermal activity recur across Uttarakhand’s seven primary seasons, and how sensitive is that pattern to spatial support and clustering?"
RECURRENCE_LABEL="Observed satellite thermal-detection recurrence, 2019–2025"
LIMITATION="Thermal detections are satellite observations, not independent confirmed fire incidents. Grid support is not burned area or a prediction of future fire risk."
OPPORTUNITY="Successful retrieval does not establish complete satellite observation opportunity. Zero returned detections do not establish fire absence."
METHODS={"detection":"All detections","dbscan":"DBSCAN-supported","hdbscan":"HDBSCAN-supported"}
LANDCOVER={-1:"Unavailable / nodata",10:"Tree cover",20:"Shrubland",30:"Grassland",40:"Cropland",50:"Built-up",60:"Bare / sparse vegetation",70:"Snow and ice",80:"Permanent water",90:"Herbaceous wetland",95:"Mangroves",100:"Moss and lichen"}
COLORS={"teal":"#1C8C8C","orange":"#D97745","navy":"#17324D","muted":"#4F6375","dbscan":"#327696","hdbscan":"#C87747"}
# Increasing observed season frequency: the map and legend share this scale.
FREQUENCY_COLORS=[[232,239,242,255],[185,225,221,255],[126,199,192,255],[76,169,158,255],[26,130,122,255],[15,108,95,255],[9,84,66,255],[5,58,45,255]]
