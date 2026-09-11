"""Constants shared by the analysis and local output stages."""

# Study systems
LAKES = [
    # Okanagan system
    "Kalamalka_Lake", "Okanagan_Lake", "Skaha_Lake", "Vaseux_Lake",
    # Nelson–Winnipeg system
    "Rainy_Lake", "Lake_of_the_Woods", "Playgreen_Lake",
    "Kiskitto_Lake", "Sipiwesk_Lake", "Split_Lake",
]

VARIABLES = ["WL", "T", "P", "R", "SWE", "Evap", "RegFlow"]

# Directly connected lake pairs used for connectivity groups.
WATERWAY_CONNECTED_PAIRS = [
    ("Kalamalka_Lake", "Okanagan_Lake"),
    ("Okanagan_Lake", "Skaha_Lake"),
    ("Skaha_Lake", "Vaseux_Lake"),
    ("Rainy_Lake", "Lake_of_the_Woods"),
    ("Playgreen_Lake", "Sipiwesk_Lake"),
    ("Kiskitto_Lake", "Sipiwesk_Lake"),
    ("Sipiwesk_Lake", "Split_Lake"),
]

# Forecast evaluation
FORECAST_HORIZON = 37
ROLLING_HORIZONS = [1, 3, 6, 12]

# Embedding selection
EMBED_TAU = 1
EMBED_E_CANDIDATES = range(2, 11)

# Statistical testing
FDR_ALPHA = 0.05
