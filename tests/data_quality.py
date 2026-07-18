import sys
import pandas as pd
import great_expectations as gx
from great_expectations.core.batch import Batch
from great_expectations.execution_engine import PandasExecutionEngine
from great_expectations.validator.validator import Validator


def validate_data(path: str):
    print(f"\n🔍 Validating: {path}")

    # -------------------------
    # Load Data
    # -------------------------
    df = pd.read_csv(path)

    # -------------------------
    # Basic Checks
    # -------------------------
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    assert df["date"].notna().all(), "❌ Invalid or missing dates"
    assert df["date"].between("2010-01-01", "2025-12-31").all(), "❌ Dates out of expected range"

    df["zipcode_str"] = df["zipcode"].astype(str).str.zfill(5)

    # ------------------------------------------------------
    # Remove invalid rows (optional cleaning before validation)
    # ------------------------------------------------------

    # Remove invalid median_sale_price
    df = df[
        (df["median_sale_price"].isna()) |
        (
            (df["median_sale_price"] >= 0) &
            (df["median_sale_price"] <= 19_000_000)
        )
    ]

    # Remove invalid median_list_price
    df = df[
        (df["median_list_price"].isna()) |
        (
            (df["median_list_price"] >= 0) &
            (df["median_list_price"] <= 19_000_000)
        )
    ]

    # -------------------------
    # Great Expectations
    # -------------------------
    context = gx.get_context(mode="ephemeral")

    batch = Batch(data=df)

    validator = Validator(
        execution_engine=PandasExecutionEngine(),
        batches=[batch],
        data_context=context,
    )

    # -------------------------
    # Expectations
    # -------------------------
    validator.expect_column_values_to_not_be_null("price")

    validator.expect_column_values_to_be_between(
        "price",
        min_value=1000,
        max_value=12_000_000,
    )

    validator.expect_column_values_to_be_between(
        "median_sale_price",
        min_value=0,
        max_value=19_000_000,
    )

    validator.expect_column_values_to_be_between(
        "median_list_price",
        min_value=0,
        max_value=19_000_000,
    )

    validator.expect_column_values_to_be_between(
        "homes_sold",
        min_value=0,
    )

    validator.expect_column_values_to_be_between(
        "pending_sales",
        min_value=0,
    )

    validator.expect_column_values_to_be_between(
        "median_dom",
        min_value=0,
        max_value=10000,
    )

    validator.expect_column_values_to_be_between(
        "avg_sale_to_list",
        min_value=0,
        max_value=2.0,
    )

    validator.expect_column_values_to_not_be_null("city_full")

    validator.expect_column_value_lengths_to_equal(
        "zipcode_str",
        5,
    )

    validator.expect_column_values_to_be_between(
        "Total Population",
        min_value=0,
    )

    validator.expect_column_values_to_be_between(
        "Median Age",
        min_value=0,
        max_value=120,
    )

    validator.expect_column_values_to_be_between(
        "Median Home Value",
        min_value=0,
    )

    # -------------------------
    # Validate
    # -------------------------
    results = validator.validate()

    total = len(results["results"])
    passed = sum(r["success"] for r in results["results"])
    failed = total - passed

    print(f"\n📊 {path}")
    print(f"✅ Passed: {passed}/{total}")

    if failed:
        print("\n❌ Failed Expectations:")

        for r in results["results"]:
            if not r["success"]:
                cfg = r["expectation_config"]

                print(f"\nExpectation : {cfg.type}")
                print(f"Column      : {cfg.kwargs.get('column')}")

                result = r.get("result", {})

                if "unexpected_count" in result:
                    print(
                        f"Unexpected  : {result['unexpected_count']}/{result['element_count']}"
                    )

        sys.exit(1)

    print("🎉 All data quality checks passed!")


if __name__ == "__main__":

    datasets = [
        "data/processed/cleaning_train.csv",
        "data/processed/cleaning_eval.csv",
        "data/processed/cleaning_holdout.csv",
    ]

    for dataset in datasets:
        validate_data(dataset)

    print("\n✅ Data Quality Validation Completed Successfully!")