"""
End-to-end orchestrator for the data preparation pipeline.

Runs the stages in order, each of which is also runnable on its own:
    1. load_raw_data  parse pgAdmin JSON exports        -> per-table pickles
    2. process_data   build the review-level Yelp set   -> YELP_SAMPLE_PKL
    3. weather        download + merge NOAA weather      -> WEATHER_ENRICHED_PKL

The processed dataset is passed to the weather stage in memory to avoid an
extra read of the multi-million-row pickle.

Usage:
    python data_pipeline.py                    # full run, cached weather
    python data_pipeline.py --refresh-weather  # force a fresh NOAA download
    python data_pipeline.py --skip-raw         # reuse existing per-table pickles
"""

import sys

import load_raw_data
import process_data
import weather


def main() -> None:
    refresh_weather = "--refresh-weather" in sys.argv
    skip_raw = "--skip-raw" in sys.argv

    if skip_raw:
        print("Skipping raw load (reusing existing per-table pickles).")
    else:
        load_raw_data.main()

    model_data = process_data.main()

    enriched = weather.build_weather_enriched(
        yelp=model_data,
        refresh=refresh_weather,
    )

    print("\nPipeline completed.")
    print(f"Final enriched dataset: {enriched.shape[0]:,} rows x {enriched.shape[1]} columns")


if __name__ == "__main__":
    main()
