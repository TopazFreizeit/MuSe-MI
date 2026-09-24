# visualizations/mi_mapping.py

# Change Talk (CT) Mappings
DARN_LABELS = ['D+', 'AB+', 'R+', 'N+']
CAT_LABELS = ['C+', 'AC+', 'TS+']

# Sustain Talk (ST) Mappings
ST_REASONS_LABELS = ['R-', 'N-', 'O-']
ST_COMMIT_LABELS = ['AB-', 'AC-', 'CO-', 'TS-']

# Other Categories
NEUTRAL_LABELS = ['N', 'O+', 'N+'] # O+ often leans CT but previously mapped to Other CT or just N depending on the strictness. We will handle generic neutral as well.
# In the original mapping:
# C was all CT. S was all ST. N was neutral.
# Now we use t2 specific labels.

def get_broad_category(t2_label):
    """Maps a t2 label to its broad MI category."""
    t2_label = t2_label.strip()
    if t2_label in DARN_LABELS:
        return 'DARN'
    elif t2_label in CAT_LABELS:
        return 'CAT'
    elif t2_label in ST_REASONS_LABELS:
        return 'ST_REASONS'
    elif t2_label in ST_COMMIT_LABELS:
        return 'ST_COMMIT'
    else:
        return 'NEUTRAL_OTHER'
