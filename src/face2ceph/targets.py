"""Canonical target names, class order, and age strata."""

TARGETS = (
    "ANB",
    "Wits",
    "SN_MP",
    "FMA",
    "PP_MP",
    "Jarabak",
    "Y_axis",
    "LAFH_TAFH",
)

SAGITTAL_TARGETS = ("ANB", "Wits")
VERTICAL_TARGETS = ("SN_MP", "FMA", "PP_MP", "Jarabak", "Y_axis", "LAFH_TAFH")
VERTICAL_SIGNS = (1.0, 1.0, 1.0, -1.0, 1.0, 1.0)
CLASS_NAMES = {
    "sagittal": ("III", "I", "II"),
    "vertical": ("Hypo", "Normo", "Hyper"),
}

AGE_MIN = 7
NORM_AGE_MIN = 11
NORM_AGE_MAX = 30


def age_band(age: float) -> str:
    if age < 7:
        return "<7"
    if age <= 9:
        return "7-9"
    if age <= 12:
        return "10-12"
    if age <= 15:
        return "13-15"
    if age <= 17:
        return "16-17"
    return ">=18"


def age_stratum(age: float) -> str:
    if age < 7:
        return "<7"
    if age < 11:
        return "7-10"
    if age <= 30:
        return "11-30"
    return ">30"
