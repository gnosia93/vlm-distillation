import polars as pl

path = "/home/ubuntu/.cache/huggingface/hub/datasets--HuggingFaceFV--finevideo/snapshots/84c74091e1c6ee7a5dffabfafb5c9033e4718883/data/train-00000-of-01357.parquet"
j = pl.col("json").struct

df = (
    pl.scan_parquet(path)
      .select(
          j.field("content_parent_category").alias("parent"),
          j.field("content_fine_category").alias("fine"),
      )
      .collect()
)

print("=== parent categories ===")
print(df["parent"].value_counts(sort=True))
print("=== fine categories (상위 30) ===")
print(df["fine"].value_counts(sort=True).head(30))
