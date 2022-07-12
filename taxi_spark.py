from pyspark.sql import SparkSession, Window
from math import radians, cos, sin, asin, sqrt

from pyspark.sql.functions import posexplode, split, col, lead, udf, sum, window, count, from_unixtime, when, \
    countDistinct, unix_timestamp, avg, percentile_approx
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, IntegerType, BooleanType

spark = SparkSession.builder.appName('taxi').getOrCreate()

# load data in DataFrame
ride_request_schema = StructType([
    StructField("timestamp", LongType(), True),
    StructField("request_id", StringType(), True),
    StructField("passenger_id", StringType(), True),
    StructField("ds", StringType(), True),
    StructField("hr", IntegerType(), True),
    StructField("comment", StringType(), True),
])

ride_accept_schema = StructType([
    StructField("timestamp", LongType(), True),
    StructField("request_id", StringType(), True),
    StructField("passenger_id", StringType(), True),
    StructField("driver_id", StringType(), True),
    StructField("is_accept", BooleanType(), True),
    StructField("ds", StringType(), True),
    StructField("hr", IntegerType(), True),
    StructField("comment", StringType(), True),
])

event_ride_request_df = spark.read.option("header", True).schema(ride_request_schema).csv("data/event_ride_request.csv")
event_ride_request_df = event_ride_request_df.withColumn("timestamp", from_unixtime("timestamp"))
event_ride_request_df.printSchema()

event_ride_accept_df = spark.read.option("header", True).schema(ride_accept_schema).csv("data/event_ride_accept.csv")
event_ride_accept_df = event_ride_accept_df.withColumn("timestamp", from_unixtime("timestamp"))
event_ride_accept_df.printSchema()

# data frames

# sql
event_ride_request_df.createOrReplaceTempView("event_ride_request")
spark.sql("SELECT * FROM event_ride_request").show(truncate=False)
event_ride_accept_df.createOrReplaceTempView("event_ride_accept")
spark.sql("SELECT * FROM event_ride_accept").show(truncate=False)

rides_df = spark.sql("""
    WITH distinct_ride_accept AS (
        SELECT *
        FROM (
            SELECT *, RANK() OVER (PARTITION BY request_id ORDER BY is_accept DESC, timestamp DESC) AS rank
            FROM event_ride_accept
        )
        WHERE rank = 1 
    )
    SELECT
        req.request_id,
        req.timestamp as request_timestamp,
        ac.timestamp as accept_timestamp,
        req.passenger_id,
        ac.driver_id,
        ac.is_accept,
        ac.comment as accept_comment,
        req.comment as request_comment,
        req.ds,
        req.hr
    FROM event_ride_request AS req
    LEFT JOIN distinct_ride_accept AS ac ON req.request_id = ac.request_id
    ORDER BY request_id
""").cache()

rides_df.orderBy("request_timestamp").show(truncate=False)

# supply and demand report
rides_df.groupBy(
    window("request_timestamp", "10 minutes")
).agg(
    count("request_id").alias("rides_requested"),
    sum(when(col("is_accept") == True, 1).otherwise(0)).alias("rides_accepted"),
    sum(when(col("is_accept") == False, 1).otherwise(0)).alias("rides_cancelled"),
    sum(when(col("is_accept").isNull(), 1).otherwise(0)).alias("rides_ignored"),
    countDistinct("passenger_id").alias("riders_requested_ride"),
    countDistinct(when(col("is_accept") == True, col("passenger_id")), "passenger_id").alias("riders_accepted_ride"),
    countDistinct(when(col("is_accept") == False, col("passenger_id")), "passenger_id").alias("riders_cancelled_ride"),
    countDistinct(when(col("is_accept") == True, col("driver_id")), "driver_id").alias("drivers_accepted_ride"),
    countDistinct(when(col("is_accept") == False, col("driver_id")), "driver_id").alias("drivers_cancelled_ride")
).show(truncate=False)

# waiting time report
rides_df.filter(
    col("is_accept").isNotNull()
).withColumn(
    "waiting_time_seconds", unix_timestamp("accept_timestamp") - unix_timestamp("request_timestamp")
).groupBy(
    window("request_timestamp", "10 minutes")
).agg(
    sum(when(col("is_accept") == True, 1).otherwise(0)).alias("rides_accepted"),
    sum(when(col("is_accept") == False, 1).otherwise(0)).alias("rides_cancelled"),
    avg(when(col("is_accept") == True, col("waiting_time_seconds"))).alias("time_to_accept_avg"),
    percentile_approx(when(col("is_accept") == True, col("waiting_time_seconds")), 0.5).alias("time_to_accept_p50"),
    percentile_approx(when(col("is_accept") == True, col("waiting_time_seconds")), 0.95).alias("time_to_accept_p90"),
    avg(when(col("is_accept") == False, col("waiting_time_seconds"))).alias("time_to_cancel_avg"),
    percentile_approx(when(col("is_accept") == False, col("waiting_time_seconds")), 0.5).alias("time_to_cancel_p50"),
    percentile_approx(when(col("is_accept") == False, col("waiting_time_seconds")), 0.95).alias("time_to_cancel_p90")
).show(truncate=False)
