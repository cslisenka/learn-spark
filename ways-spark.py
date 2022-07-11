from pyspark.sql import SparkSession, Window
from math import radians, cos, sin, asin, sqrt

from pyspark.sql.functions import posexplode, split, col, lead, udf, sum
from pyspark.sql.types import StructType, StructField, StringType, DoubleType


def distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r


spark = SparkSession.builder.appName('ways').getOrCreate()
spark.udf.register("distance", distance, DoubleType())
# Testing UDF
#spark.sql("SELECT distance(1.0, 1.0, 2.0, 2.0)").show()

# load data in DataFrame
nodes_schema = StructType([
    StructField("node_id", StringType(), True),
    StructField("lat", DoubleType(), True),
    StructField("lon", DoubleType(), True)
])

ways_schema = StructType([
    StructField("way_id", StringType(), True),
    StructField("nodes", StringType(), True),
])

nodes_df = spark.read.option("header", True).schema(nodes_schema).csv("data/nodes.csv")
nodes_df.printSchema()

ways_df = spark.read.option("header", True).schema(ways_schema).csv("data/ways.csv")
ways_df.printSchema()

distanceUDF = udf(lambda lat1, lon1, lat2, lon2: distance(lat1, lon1, lat2, lon2), DoubleType())

# data frames
distances_df = ways_df.select(
    "way_id", posexplode(split("nodes", " ")).alias("position", "node_id")
).join(
    nodes_df, "node_id", "left"
).withColumn(
    "next_node_id", lead("node_id").over(Window.partitionBy("way_id").orderBy(col("position").desc()))
).withColumn(
    "next_node_lat", lead("lat").over(Window.partitionBy("way_id").orderBy(col("position").desc()))
).withColumn(
    "next_node_lon", lead("lon").over(Window.partitionBy("way_id").orderBy(col("position").desc()))
).filter(
    col("next_node_id").isNotNull()
).withColumn(
    "path_distance", distanceUDF(col("lat"), col("lon"), col("next_node_lat"), col("next_node_lon"))
)

distances_df.show()

distances_df.groupBy("way_id").agg(
    sum("path_distance").alias("way_distance")
).orderBy(col("way_distance").desc()).show()

# sql
nodes_df.createOrReplaceTempView("nodes")
spark.sql("SELECT * FROM nodes").show()
ways_df.createOrReplaceTempView("ways")
spark.sql("SELECT * FROM ways").show()

distances_sql_df = spark.sql(
    """
    WITH exploded AS (
        SELECT way_id, posexplode(arr) AS (position, node)
        FROM (SELECT way_id, split(nodes, ' ') as arr FROM ways)
    ), exploded_with_coordinates  AS (
        SELECT way_id, position, node, lat, lon
        FROM exploded
        LEFT JOIN nodes ON exploded.node = nodes.node_id
    ), paths AS (
        SELECT *
        FROM (
            SELECT
                way_id, node, lat, lon,
                lead(node, 1) OVER (PARTITION BY way_id ORDER BY position DESC) as next_node,
                lead(lat, 1) OVER (PARTITION BY way_id ORDER BY position DESC) as next_lat,
                lead(lon, 1) OVER (PARTITION BY way_id ORDER BY position DESC) as next_lon
            FROM exploded_with_coordinates
        )
        WHERE next_node is not null
    ), path_with_distance AS (
        SELECT
            way_id, node, lat, lon, next_node, next_lat, next_lon,
            distance(lat, lon, next_lat, next_lon) as path_distance
        FROM paths
    ) SELECT * FROM path_with_distance
    """)

distances_sql_df.show()
distances_sql_df.createOrReplaceTempView("path_distance")

# Calculate TOP ways
spark.sql("""
    SELECT way_id, sum(path_distance) as way_distance
    FROM path_distance
    GROUP BY way_id
    ORDER BY 2 DESC
""").show()
