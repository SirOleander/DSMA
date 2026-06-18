import numpy as np
import pandas as pd

from config import (
    PROCESSED_DATA_DIR,
    SAMPLE_SIZE,
    YELP_SAMPLE_PKL,
    YELP_SAMPLE_CSV,
    WRITE_FULL_CSV_OUTPUTS,
)
from reporting import (
    print_section,
    print_subsection,
    print_table,
    build_variable_overview,
    build_numeric_distribution_overview,
    build_negative_value_check,
    build_log_skewness_comparison,
)

# ---------------------------------------------------------------------------
# Study scope
# ---------------------------------------------------------------------------

MIN_REVIEW_YEAR = 2010

SELECTED_CITY_STATE_SCOPE = [
    ("Philadelphia", "PA"),
    ("New Orleans", "LA"),
    ("Nashville", "TN"),
    ("Tampa", "FL"),
    ("Indianapolis", "IN"),
    ("Tucson", "AZ"),
    ("Saint Louis", "MO"),
    ("Reno", "NV"),
    ("Santa Barbara", "CA"),
    ("Saint Petersburg", "FL"),
]

RANDOM_STATE = 42

# Set False for a lean production run (skips the diagnostic tables but keeps
# all data transformations). Set True while exploring / writing the report.
VERBOSE = True

# String / categorical columns. Single source of truth used both for the
# "Unknown" cleaning step and for the category-dtype memory optimisation.
CATEGORICAL_COLUMNS = [
    "city",
    "state",
    "attributes.RestaurantsPriceRange2",
    "attributes.BusinessAcceptsCreditCards",
    "attributes.RestaurantsTakeOut",
    "attributes.RestaurantsDelivery",
    "attributes.OutdoorSeating",
    "attributes.BikeParking",
    "attributes.GoodForKids",
    "attributes.RestaurantsReservations",
    "attributes.Alcohol",
    "attributes.WiFi",
    "attributes.HasTV",
    "attributes.NoiseLevel",
    "attributes.RestaurantsGoodForGroups",
    "attributes.RestaurantsTableService",
]

# Count columns whose missing values genuinely mean "no record" -> fill with 0.
COUNT_COLUMNS = [
    "checkin_count",
    "tip_count",
    "tip_compliment_count",
    "photo_count",
    "photo_drink",
    "photo_food",
    "photo_inside",
    "photo_menu",
    "photo_outside",
]

# Columns that should never be negative (used by the data-quality diagnostic).
NON_NEGATIVE_COLUMNS = [
    "review_useful",
    "review_funny",
    "review_cool",
    "business_review_count",
    "user_review_count",
    "user_fans",
    "user_useful",
    "user_funny",
    "user_cool",
    "checkin_count",
    "tip_count",
    "tip_compliment_count",
    "photo_count",
    "photo_food",
    "photo_drink",
    "photo_menu",
    "photo_inside",
    "photo_outside",
    "review_text_length",
]

# Variables excluded from the pre-log skewness diagnostic: the target, the
# star-based leakage variables, date parts, and coordinates.
SKEW_EXCLUDE_COLUMNS = [
    "satisfied",
    "review_stars",
    "business_stars",
    "user_average_stars",
    "review_year",
    "review_month",
    "review_weekday",
    "latitude",
    "longitude",
]


def load_processed_tables() -> dict:
    """Load already-converted pickle files."""
    tables = {
        "business": pd.read_pickle(PROCESSED_DATA_DIR / "business.pkl"),
        "reviews": pd.read_pickle(PROCESSED_DATA_DIR / "reviews.pkl"),
        "users": pd.read_pickle(PROCESSED_DATA_DIR / "users.pkl"),
        "checkin": pd.read_pickle(PROCESSED_DATA_DIR / "checkin.pkl"),
        "tip": pd.read_pickle(PROCESSED_DATA_DIR / "tip.pkl"),
        "photo": pd.read_pickle(PROCESSED_DATA_DIR / "photo.pkl"),
    }

    return tables


def prepare_core_tables(
    business: pd.DataFrame,
    reviews: pd.DataFrame,
    users: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rename columns and create target variable."""

    business = business.rename(columns={
        "name": "business_name",
        "stars": "business_stars",
        "review_count": "business_review_count",
    })

    reviews = reviews.rename(columns={
        "stars": "review_stars",
        "date": "review_date",
        "text": "review_text",
        "useful": "review_useful",
        "funny": "review_funny",
        "cool": "review_cool",
    })

    users = users.rename(columns={
        "name": "user_name",
        "review_count": "user_review_count",
        "average_stars": "user_average_stars",
        "useful": "user_useful",
        "funny": "user_funny",
        "cool": "user_cool",
        "fans": "user_fans",
    })

    reviews["satisfied"] = (reviews["review_stars"] >= 4).astype(int)

    return business, reviews, users


def select_columns_before_merge(
    business: pd.DataFrame,
    reviews: pd.DataFrame,
    users: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Keep only relevant columns before merging.

    This reduces memory usage and runtime because unused columns are not repeated
    across millions of review rows.
    """

    reviews = reviews.copy()

    if "review_text" in reviews.columns:
        reviews["review_text_length"] = reviews["review_text"].fillna("").str.len()
        reviews = reviews.drop(columns=["review_text"])

    business_columns = [
        "business_id",
        "business_stars",
        "business_review_count",
        "is_open",
        "city",
        "state",
        "latitude",
        "longitude",
        "attributes.RestaurantsPriceRange2",
        "attributes.BusinessAcceptsCreditCards",
        "attributes.RestaurantsTakeOut",
        "attributes.RestaurantsDelivery",
        "attributes.OutdoorSeating",
        "attributes.BikeParking",
        "attributes.GoodForKids",
        "attributes.RestaurantsReservations",
        "attributes.Alcohol",
        "attributes.WiFi",
        "attributes.HasTV",
        "attributes.NoiseLevel",
        "attributes.RestaurantsGoodForGroups",
        "attributes.RestaurantsTableService",
    ]

    review_columns = [
        "review_id",
        "business_id",
        "user_id",
        "satisfied",
        "review_stars",
        "review_date",
        "review_text_length",
        "review_useful",
        "review_funny",
        "review_cool",
    ]

    user_columns = [
        "user_id",
        "user_review_count",
        "user_average_stars",
        "user_fans",
        "user_useful",
        "user_funny",
        "user_cool",
    ]

    business = business[
        [col for col in business_columns if col in business.columns]
    ].copy()

    reviews = reviews[
        [col for col in review_columns if col in reviews.columns]
    ].copy()

    users = users[
        [col for col in user_columns if col in users.columns]
    ].copy()

    return business, reviews, users


def sample_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Sample reviews before merging to keep processing manageable.

    When SAMPLE_SIZE covers the whole table (the current setting), we return
    the data as-is instead of shuffling several million rows for no reason.
    """
    if SAMPLE_SIZE >= len(reviews):
        return reviews.copy()

    return reviews.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).copy()


def create_checkin_features(checkin: pd.DataFrame) -> pd.DataFrame:
    """Create business-level check-in features."""
    checkin = checkin.copy()

    checkin["checkin_count"] = checkin["date"].fillna("").apply(
        lambda x: 0 if x == "" else len(str(x).split(","))
    )

    return checkin[["business_id", "checkin_count"]]


def create_tip_features(tip: pd.DataFrame) -> pd.DataFrame:
    """Create business-level tip features."""
    tip_features = tip.groupby("business_id").agg(
        tip_count=("text", "count"),
        tip_compliment_count=("compliment_count", "sum"),
    ).reset_index()

    return tip_features


def create_photo_features(photo: pd.DataFrame) -> pd.DataFrame:
    """Create business-level photo features."""
    photo_features = photo.groupby("business_id").agg(
        photo_count=("photo_id", "count")
    ).reset_index()

    photo_label_features = (
        photo.pivot_table(
            index="business_id",
            columns="label",
            values="photo_id",
            aggfunc="count",
            fill_value=0,
        )
        .add_prefix("photo_")
        .reset_index()
    )

    photo_features = photo_features.merge(
        photo_label_features,
        on="business_id",
        how="left"
    )

    return photo_features


def merge_tables(
    reviews: pd.DataFrame,
    business: pd.DataFrame,
    users: pd.DataFrame,
    checkin_features: pd.DataFrame,
    tip_features: pd.DataFrame,
    photo_features: pd.DataFrame
) -> pd.DataFrame:
    """Merge all information into one review-level dataset."""

    dataset = reviews.merge(
        business,
        on="business_id",
        how="left"
    )

    dataset = dataset.merge(
        users,
        on="user_id",
        how="left"
    )

    dataset = dataset.merge(
        checkin_features,
        on="business_id",
        how="left"
    )

    dataset = dataset.merge(
        tip_features,
        on="business_id",
        how="left"
    )

    dataset = dataset.merge(
        photo_features,
        on="business_id",
        how="left"
    )

    return dataset

def clean_values(dataset: pd.DataFrame) -> pd.DataFrame:
    """Clean obvious missing values and create simple derived variables."""
    dataset = dataset.copy()

    for col in COUNT_COLUMNS:
        if col in dataset.columns:
            dataset[col] = dataset[col].fillna(0)

    for col in CATEGORICAL_COLUMNS:
        if col in dataset.columns:
            dataset[col] = (
                dataset[col]
                .astype("string")
                .replace({"None": "Unknown", "nan": "Unknown"})
                .fillna("Unknown")
            )

    if "review_date" in dataset.columns:
        dataset["review_date"] = pd.to_datetime(dataset["review_date"], errors="coerce")
        dataset["review_year"] = dataset["review_date"].dt.year
        dataset["review_month"] = dataset["review_date"].dt.month
        dataset["review_weekday"] = dataset["review_date"].dt.dayofweek
    
    review_vote_columns = [
        "review_useful",
        "review_funny",
        "review_cool",
    ]

    for col in review_vote_columns:
        if col in dataset.columns:
            dataset[col] = pd.to_numeric(dataset[col], errors="coerce")
            dataset[col] = dataset[col].clip(lower=0)

    return dataset

def standardize_city_names(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize obvious duplicate city names and spelling variants.

    This avoids treating the same city as multiple cities due to casing,
    punctuation, abbreviations, or spelling differences.
    """
    dataset = dataset.copy()

    if "city" not in dataset.columns:
        return dataset

    dataset["city"] = (
        dataset["city"]
        .astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    city_replacements = {
        'Abington Township': 'Abington',
        'Abington Twp': 'Abington',
        'Algiers': 'New Orleans',
        'Antioch': 'Nashville',
        'Ashland': 'Ashland City',
        'Bala Cynwyd': 'Philadelphia',
        'Belle Chase': 'Belle Chasse',
        'Belle Meade': 'Nashville',
        'Belleair Blf': 'Belleair Bluffs',
        'Bellefontaine': 'Bellefontaine Neighbors',
        'Bellville': 'Belleville',
        'Berlin Township': 'Berlin',
        'Berlin Twp': 'Berlin',
        'Berry Hill': 'Nashville',
        'BOISE': 'Boise',
        'Bordentown Township': 'Bordentown',
        'Bordentown Twp': 'Bordentown',
        'BRANDON': 'Brandon',
        'Bristol Township': 'Bristol',
        'Bristol Twp': 'Bristol',
        'Bucktown': 'New Orleans',
        'Burlington Township': 'Burlington',
        'Burlington Twp': 'Burlington',
        'Burlngtn Township': 'Burlington',
        'Bywater': 'New Orleans',
        'Cane Ridge': 'Nashville',
        'Carrollwood': 'Tampa',
        'Center City': 'Philadelphia',
        'Chalemette': 'Chalmette',
        'Chalmette': 'Chalmette',
        'Cheltenham Township': 'Cheltenham',
        'Cheltenham Twp': 'Cheltenham',
        'Chestnut Hill': 'Philadelphia',
        'Cinnaminson Township': 'Cinnaminson',
        'Cinnaminson Twp': 'Cinnaminson',
        'CLEARWATER': 'Clearwater',
        'Clearwater Beach': 'Clearwater Beach',
        'Clearwater/ Countryside': 'Clearwater',
        'Conshohocken ': 'Conshohocken',
        'Conshy': 'Conshohocken',
        'Delran Township': 'Delran',
        'Delran Twp': 'Delran',
        'Deptford Township': 'Deptford',
        'Deptford Twp': 'Deptford',
        'DONELSON': 'Nashville',
        'Donelson': 'Nashville',
        'Downtown': 'New Orleans',
        'East Edmonton': 'Edmonton',
        'East Nashville': 'Nashville',
        'East Norriton Township': 'East Norriton',
        'East Norriton Twp': 'East Norriton',
        'East St Louis': 'East Saint Louis',
        'East St. Louis': 'East Saint Louis',
        'Eastampton Township': 'Eastampton',
        'Eastampton Twp': 'Eastampton',
        'East Passyunk': 'Philadelphia',
        'EdMonton': 'Edmonton',
        'Evesham Township': 'Evesham',
        'Evesham Twp': 'Evesham',
        'Fairview Hts': 'Fairview Heights',
        'Fairview Hts.': 'Fairview Heights',
        'Falls Township': 'Falls',
        'Falls Twp': 'Falls',
        'Feasterville': 'Feasterville-Trevose',
        'Feasterville Trevose': 'Feasterville-Trevose',
        'Fishtown': 'Philadelphia',
        'goodlettsville': 'Goodlettsville',
        'Greater Northdale': 'Tampa',
        'Haddon Township': 'Haddonfield',
        'Haddon Twp': 'Haddonfield',
        'Hamilton Township': 'Hamilton',
        'Hamilton Twp': 'Hamilton',
        'Hermitage': 'Nashville',
        'Horsham Township': 'Horsham',
        'Horsham Twp': 'Horsham',
        'HUdson': 'Hudson',
        'Huntingdon Valley': 'Huntingdon Valley',
        'Huntingdon Vly': 'Huntingdon Valley',
        'Huntingdon Vy': 'Huntingdon Valley',
        'Indianaplois': 'Indianapolis',
        'indianopolis': 'Indianapolis',
        'Inglewood': 'Nashville',
        'INpolis': 'Indianapolis',
        'Jenkintown ': 'Jenkintown',
        'KING OF PRUSSIA': 'King of Prussia',
        'King Of Prussia': 'King of Prussia',
        'Kng of Prussa': 'King of Prussia',
        'Land o Lakes': "Land O' Lakes",
        'Land o lakes': "Land O' Lakes",
        "Land o'Lakes": "Land O' Lakes",
        'Land O lakes': "Land O' Lakes",
        "Land O'Lakes": "Land O' Lakes",
        'LARGO': 'Largo',
        'Largo (Walsingham)': 'Largo',
        'lawrance': 'Lawrence',
        'Lawrence Township': 'Lawrence',
        'Lawrence Twp': 'Lawrence',
        'LITHIA': 'Lithia',
        'Lower Gwynedd Township': 'Lower Gwynedd',
        'Lower Gwynedd Twp': 'Lower Gwynedd',
        'Lower Merion Township': 'Lower Merion',
        'Lower Merion Twp': 'Lower Merion',
        'Lower Moreland Township': 'Lower Moreland',
        'Lower Moreland Twp': 'Lower Moreland',
        'Lower Providence Township': 'Lower Providence',
        'Lower Southampton Township': 'Lower Southampton',
        'Lower Southampton Twp': 'Lower Southampton',
        'Lumberton Township': 'Lumberton',
        'Lumberton Twp': 'Lumberton',
        'Lutz fl': 'Lutz',
        'MADISON': 'Nashville',
        'Madison': 'Nashville',
        'Manayunk': 'Philadelphia',
        'Mantua Township': 'Mantua',
        'Mantua Twp': 'Mantua',
        'Mc Cordsville': 'McCordsville',
        'Mccordsville': 'McCordsville',
        'Medford Lakes': 'Medford',
        'Meridan': 'Meridian',
        'Metarie': 'Metairie',
        'Middletown Township': 'Middletown',
        'Middletown Twp': 'Middletown',
        'Monroe Township': 'Monroe',
        'Monroe Twp': 'Monroe',
        'Mount Airy': 'Philadelphia',
        'Mount Holly Township': 'Mount Holly',
        'Mount Laurel Township': 'Mount Laurel',
        'Mt Airy': 'Philadelphia',
        'Mt Ephraim': 'Mount Ephraim',
        'Mt Holly': 'Mount Holly',
        'Mt Holly Twp': 'Mount Holly',
        'Mt Juliet': 'Mount Juliet',
        'Mt Laurel': 'Mount Laurel',
        'Mt Laurel Township': 'Mount Laurel',
        'Mt Royal': 'Mount Royal',
        'Mt. Ephraim': 'Mount Ephraim',
        'Mt. Holly': 'Mount Holly',
        'Mt. Juliet': 'Mount Juliet',
        'Mt. Laurel': 'Mount Laurel',
        'Mt. Royal': 'Mount Royal',
        'Mt.Juliet': 'Mount Juliet',
        'N Wales': 'North Wales',
        'N Redngtn Bch': 'North Redington Beach',
        'NASHVILLE': 'Nashville',
        'Nashville ': 'Nashville',
        'Nashville-Davidson metropolitan gover': 'Nashville',
        'NEW ORLEANS': 'New Orleans',
        'Newtown Sq': 'Newtown Square',
        'Newtown Square': 'Newtown Square',
        'Nolensville': 'Nolensville',
        'North Redingtn Bch': 'North Redington Beach',
        'Northampton Township': 'Northampton',
        'Northampton Twp': 'Northampton',
        'Northern Liberties': 'Philadelphia',
        'NW Edmonton': 'Edmonton',
        'O Fallon': "O'Fallon",
        "O' Fallon": "O'Fallon",
        "O'fallon": "O'Fallon",
        'Old City': 'Philadelphia',
        'Old Hickory': 'Nashville',
        'Palm harbor': 'Palm Harbor',
        'Passyunk Square': 'Philadelphia',
        'Pemberton Township': 'Pemberton',
        'Pemberton Twp': 'Pemberton',
        'Pennsauken Township': 'Pennsauken',
        'Pennsauken Twp': 'Pennsauken',
        'Pensauken': 'Pennsauken',
        'PHILADELPHIA': 'Philadelphia',
        'Philadelphia ': 'Philadelphia',
        'Phialdelphia': 'Philadelphia',
        'Phila': 'Philadelphia',
        'Philly': 'Philadelphia',
        'PINELLAS PARK': 'Pinellas Park',
        'Pinellas': 'Pinellas Park',
        'Pinellas park': 'Pinellas Park',
        'Pittsgrove Township': 'Pittsgrove',
        'Plainfiled': 'Plainfield',
        'Plymouth Meeting ': 'Plymouth Meeting',
        'Plymouth Mtng': 'Plymouth Meeting',
        'Queen Village': 'Philadelphia',
        'Redington Shore': 'Redington Shores',
        'Redington Shor': 'Redington Shores',
        'RENO': 'Reno',
        'reno': 'Reno',
        'Rittenhouse': 'Philadelphia',
        'Rittenhouse Square': 'Philadelphia',
        'Riveridge': 'River Ridge',
        'RIVERVIEW': 'Riverview',
        'Riverview Fl': 'Riverview',
        'Roxborough': 'Philadelphia',
        'Saftey Harbor': 'Safety Harbor',
        'Saint Pete Beach': 'Saint Pete Beach',
        'Saint Petersburg': 'Saint Petersburg',
        'Saint Louis,': 'Saint Louis',
        'Saintt Petersburg': 'Saint Petersburg',
        'Scott AFB': 'Scott Air Force Base',
        'Scott Afb': 'Scott Air Force Base',
        'South Philadelphia': 'Philadelphia',
        'South Tampa': 'Tampa',
        'Southampton Township': 'Southampton',
        'Southampton Twp': 'Southampton',
        'Southeast Edmonton': 'Edmonton',
        'Southwest Tampa': 'Tampa',
        'SPRINGHILL': 'Spring Hill',
        'Springfield Township': 'Springfield',
        'Springfield Twp': 'Springfield',
        'St Ann': 'Saint Ann',
        'St Charles': 'Saint Charles',
        'St Louis': 'Saint Louis',
        'St Louis County': 'Saint Louis',
        'St Louis Downtown': 'Saint Louis',
        'St Pete': 'Saint Petersburg',
        'St Pete Beach': 'Saint Pete Beach',
        'St Petersburg': 'Saint Petersburg',
        'St. Ann': 'Saint Ann',
        'St. Charles': 'Saint Charles',
        'St. Louis': 'Saint Louis',
        'St. Louis County': 'Saint Louis',
        'St. Pete Beach': 'Saint Pete Beach',
        'St. Peters': 'Saint Peters',
        'St. Petersburg': 'Saint Petersburg',
        'St. Rose': 'Saint Rose',
        'St.Ann': 'Saint Ann',
        'St.Charles': 'Saint Charles',
        'St.Louis': 'Saint Louis',
        'St.Petersburg': 'Saint Petersburg',
        'St.Rose': 'Saint Rose',
        'ST LOUIS': 'Saint Louis',
        'ST. Louis': 'Saint Louis',
        'ST. PETE BEACH': 'Saint Pete Beach',
        'ST. PETERSBURG': 'Saint Petersburg',
        'TAMPA': 'Tampa',
        'Tampa Bay': 'Tampa',
        'Tampa,Fl': 'Tampa',
        'Tarpon springs': 'Tarpon Springs',
        'TEMPLE TERR': 'Tampa',
        'Temple Terr': 'Tampa',
        'Temple Terrace': 'Tampa',
        'Thonotossasa': 'Thonotosassa',
        'Town & Country': 'Town And Country',
        'Town and Country': 'Town And Country',
        'Town And Country': 'Town And Country',
        "Town 'N' Country": 'Tampa',
        "Town 'n' Country": 'Tampa',
        'Town n Country': 'Tampa',
        'Tredyffrin Township': 'Tredyffrin',
        'Tredyffrin Twp': 'Tredyffrin',
        'Trevose': 'Feasterville-Trevose',
        'TUCSON': 'Tucson',
        'tucson': 'Tucson',
        'University City': 'Philadelphia',
        'Upper Darby Township': 'Upper Darby',
        'Upper Darby Twp': 'Upper Darby',
        'Upper Gwynedd Township': 'Upper Gwynedd',
        'Upper Gwynedd Twp': 'Upper Gwynedd',
        'Upper Merion Township': 'Upper Merion',
        'Upper Merion Twp': 'Upper Merion',
        'Upper Moreland Township': 'Upper Moreland',
        'Upper Moreland Twp': 'Upper Moreland',
        'Upper Pittsgrove': 'Pittsgrove',
        'Upper Southampton Township': 'Upper Southampton',
        'Upper Southampton Twp': 'Upper Southampton',
        'Voorhees Township': 'Voorhees',
        'Voorhees Twp': 'Voorhees',
        'Warrington Township': 'Warrington',
        'Washington Twp': 'Washington Township',
        'Washington Twp.': 'Washington Township',
        'Wesley chapel': 'Wesley Chapel',
        'West Chester PA': 'West Chester',
        'WEST CHESTER': 'West Chester',
        'West Deptford Township': 'West Deptford',
        'West Deptford Twp': 'West Deptford',
        'West Edmonton': 'Edmonton',
        'West Norriton Township': 'West Norriton',
        'West Norriton Twp': 'West Norriton',
        'Westampton Township': 'Westampton',
        'Westampton Twp': 'Westampton',
        'Westchase': 'Tampa',
        'Westchester': 'West Chester',
        'Whitemarsh Township': 'Whitemarsh',
        'Whitemarsh Twp': 'Whitemarsh',
        'Whites Creek': 'Whites Creek',
        'wilmington': 'Wilmington',
        'wimauma': 'Wimauma',
        'Winslow Township': 'Winslow',
        'Winslow Twp': 'Winslow',
        'Woolwich Township': 'Woolwich',
        'Woolwich Twp': 'Woolwich',
    }

    dataset["city"] = dataset["city"].replace(city_replacements)

    return dataset


def add_log_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Add log-transformed versions of skewed non-negative count variables."""
    dataset = dataset.copy()

    log_candidates = [
        # Business activity / popularity
        "business_review_count",

        # User activity / history
        "user_review_count",
        "user_useful",
        "user_funny",
        "user_cool",
        "user_fans",

        # Business engagement / activity
        "checkin_count",
        "tip_count",
        "tip_compliment_count",
        "photo_count",
        "photo_food",
        "photo_drink",
        "photo_menu",
        "photo_inside",
        "photo_outside",

        # Review-level behavior
        "review_useful",
        "review_funny",
        "review_cool",
        "review_text_length",
    ]

    for col in log_candidates:
        if col in dataset.columns:
            values = pd.to_numeric(dataset[col], errors="coerce")
            # log1p of the non-negative value. Missing values stay NaN on
            # purpose: imputation is deferred to the modelling pipeline so it
            # can be fit on training data only. Genuine count columns have
            # already been 0-filled in clean_values(), so their logs are 0.
            dataset[f"log_{col}"] = np.log1p(values.clip(lower=0))

    return dataset

def filter_study_scope(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Restrict the dataset to the selected study scope.

    Scope:
    - selected top 10 city-state restaurant markets
    - reviews from 2010 onward, including 2010

    This filter is applied after city standardization and date cleaning, so the
    selected city names match the cleaned city names and the NOAA merge keys.
    """
    dataset = dataset.copy()

    required_columns = {"city", "state", "review_year"}
    missing_columns = required_columns - set(dataset.columns)

    if missing_columns:
        raise ValueError(
            "Cannot filter study scope because these columns are missing: "
            f"{sorted(missing_columns)}"
        )

    rows_before = len(dataset)

    selected_pairs = pd.DataFrame(
        SELECTED_CITY_STATE_SCOPE,
        columns=["city", "state"],
    )

    dataset = dataset[dataset["review_year"] >= MIN_REVIEW_YEAR].copy()

    dataset = dataset.merge(
        selected_pairs,
        on=["city", "state"],
        how="inner",
    )

    rows_after = len(dataset)

    print("\nApplied study-scope filter.")
    print(f"Rows before filter: {rows_before:,}")
    print(f"Rows after filter:  {rows_after:,}")
    print(f"Minimum review year: {MIN_REVIEW_YEAR}")

    print("\nReviews by selected city-state:")
    city_summary = (
        dataset.groupby(["city", "state"], observed=True)
        .size()
        .reset_index(name="review_count")
        .sort_values("review_count", ascending=False)
    )
    print(city_summary.to_string(index=False))

    print("\nReviews by year:")
    year_summary = (
        dataset.groupby("review_year", observed=True)
        .size()
        .reset_index(name="review_count")
        .sort_values("review_year")
    )
    print(year_summary.to_string(index=False))

    missing_scope_pairs = selected_pairs.merge(
        city_summary[["city", "state"]],
        on=["city", "state"],
        how="left",
        indicator=True,
    )

    missing_scope_pairs = missing_scope_pairs[
        missing_scope_pairs["_merge"] == "left_only"
    ][["city", "state"]]

    if not missing_scope_pairs.empty:
        print("\nWARNING: These selected city-state pairs have zero rows:")
        print(missing_scope_pairs.to_string(index=False))

    if dataset.empty:
        raise ValueError("Study-scope filter removed all rows.")

    return dataset

def select_model_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    """Select final columns for EDA and modeling."""

    model_columns = [
        # IDs
        "review_id",
        "business_id",
        "user_id",

        # Target
        "satisfied",

        # LEAKAGE WARNING: keep for EDA only, exclude from modelling.
        # review_stars: the target is derived directly from it.
        # business_stars / user_average_stars (below): all-time averages that
        # already include this review's rating, so they leak the target and are
        # also temporal leakage. See config.LEAKAGE_COLS.
        "review_stars",

        # Review-level variables
        "review_text_length",
        "review_useful",
        "review_funny",
        "review_cool",
        "log_review_useful",
        "log_review_funny",
        "log_review_cool",
        "log_review_text_length",
        "review_year",
        "review_month",
        "review_weekday",
        "review_date",

        # Business-level variables
        "business_stars",
        "business_review_count",
        "log_business_review_count",
        "is_open",
        "city",
        "state",
        "latitude",
        "longitude",
        "attributes.RestaurantsPriceRange2",
        "attributes.BusinessAcceptsCreditCards",
        "attributes.RestaurantsTakeOut",
        "attributes.RestaurantsDelivery",
        "attributes.OutdoorSeating",
        "attributes.BikeParking",
        "attributes.GoodForKids",
        "attributes.RestaurantsReservations",
        "attributes.Alcohol",
        "attributes.WiFi",
        "attributes.HasTV",
        "attributes.NoiseLevel",
        "attributes.RestaurantsGoodForGroups",
        "attributes.RestaurantsTableService",

        # User-level variables
        "user_review_count",
        "user_average_stars",
        "user_fans",
        "user_useful",
        "user_funny",
        "user_cool",
        "log_user_review_count",
        "log_user_fans",
        "log_user_useful",
        "log_user_funny",
        "log_user_cool",

        # Restaurant activity features
        "checkin_count",
        "tip_count",
        "tip_compliment_count",
        "photo_count",
        "photo_food",
        "photo_drink",
        "photo_menu",
        "photo_inside",
        "photo_outside",
        "log_checkin_count",
        "log_tip_count",
        "log_tip_compliment_count",
        "log_photo_count",
        "log_photo_food",
        "log_photo_drink",
        "log_photo_menu",
        "log_photo_inside",
        "log_photo_outside",
    ]

    available_columns = [col for col in model_columns if col in dataset.columns]

    missing_columns = [col for col in model_columns if col not in dataset.columns]
    if missing_columns:
        print("\nColumns not found and therefore skipped:")
        print(missing_columns)

    return dataset[available_columns].copy()

def optimize_dtypes(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce memory use without changing any values.

    - Known string/attribute columns become 'category'.
    - Integer-valued numeric columns with no missing values are downcast to the
      smallest safe integer type. Columns containing NaN (e.g. the 7 unmatched
      users) or genuine floats (star ratings, log features) are left untouched.
    """
    dataset = dataset.copy()

    for col in CATEGORICAL_COLUMNS:
        if col in dataset.columns:
            dataset[col] = dataset[col].astype("category")

    for col in dataset.select_dtypes(include="number").columns:
        values = dataset[col]
        if values.isna().any():
            continue
        if (values == values.round()).all():
            dataset[col] = pd.to_numeric(values, downcast="integer")

    return dataset


def save_outputs(model_data: pd.DataFrame) -> None:
    """Save processed sample dataset (pickle always; CSV only if enabled)."""
    model_data.to_pickle(YELP_SAMPLE_PKL)
    print(f"Saved pickle: {YELP_SAMPLE_PKL}")

    if WRITE_FULL_CSV_OUTPUTS:
        model_data.to_csv(YELP_SAMPLE_CSV, index=False)
        print(f"Saved CSV:    {YELP_SAMPLE_CSV}")


def main(verbose: bool = VERBOSE) -> None:
    print("Loading processed tables...")
    tables = load_processed_tables()

    business = tables["business"]
    reviews = tables["reviews"]
    users = tables["users"]
    checkin = tables["checkin"]
    tip = tables["tip"]
    photo = tables["photo"]

    business, reviews, users = prepare_core_tables(
        business=business,
        reviews=reviews,
        users=users,
    )

    if verbose:
        print_section("INITIAL DATASET")
        initial_tables = {
            "business": business,
            "reviews": reviews,
            "users": users,
            "checkin_raw": checkin,
            "tip_raw": tip,
            "photo_raw": photo,
        }
        for name, frame in initial_tables.items():
            print_subsection(f"[{name}] variable overview")
            print_table(build_variable_overview(frame))

    business, reviews, users = select_columns_before_merge(
        business=business,
        reviews=reviews,
        users=users,
    )

    review_sample = sample_reviews(reviews)

    checkin_features = create_checkin_features(checkin)
    tip_features = create_tip_features(tip)
    photo_features = create_photo_features(photo)

    if verbose:
        print_section("NEW VARIABLES AFTER AGGREGATION")
        aggregated_tables = {
            "checkin_features": checkin_features,
            "tip_features": tip_features,
            "photo_features": photo_features,
        }
        for name, frame in aggregated_tables.items():
            print_subsection(f"[{name}] variable overview")
            print_table(build_variable_overview(frame))

    print("\nMerging tables...")
    dataset = merge_tables(
        reviews=review_sample,
        business=business,
        users=users,
        checkin_features=checkin_features,
        tip_features=tip_features,
        photo_features=photo_features,
    )

    print("\nCleaning values.")
    dataset = clean_values(dataset)
    dataset = standardize_city_names(dataset)
    
    print("\nFiltering study scope.")
    dataset = filter_study_scope(dataset)


    if verbose:
        print_section("NEGATIVE VALUE CHECK")
        print_table(build_negative_value_check(dataset, NON_NEGATIVE_COLUMNS))

        print_section("NUMERIC DISTRIBUTION AND SKEWNESS BEFORE LOG FEATURES")
        print_table(
            build_numeric_distribution_overview(
                dataset,
                exclude_columns=SKEW_EXCLUDE_COLUMNS,
            )
        )

    print("\nAdding log features...")
    dataset = add_log_features(dataset)

    if verbose:
        print_section("SKEWNESS COMPARISON BEFORE AND AFTER LOG TRANSFORMATION")
        print_table(build_log_skewness_comparison(dataset))

    print("\nSelecting final columns...")
    model_data = select_model_columns(dataset)

    print("\nOptimizing dtypes...")
    model_data = optimize_dtypes(model_data)

    if verbose:
        print_section("FINAL VARIABLES FOR EDA AND MODELING")
        print_table(build_variable_overview(model_data))

    print("\nSaving model data...")
    save_outputs(model_data)

    print("\nProcessing completed.")

    return model_data


if __name__ == "__main__":
    main()