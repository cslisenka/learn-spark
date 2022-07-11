from pyspark.sql import SparkSession, Window

# create session
from pyspark.sql import SparkSession
from pyspark.sql.functions import max, col, rank
# create session
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

spark = SparkSession.builder.appName('salary').getOrCreate()

# load data in DataFrame
schema = StructType([
    StructField("employee_id", StringType(), True),
    StructField("department_id", StringType(), True),
    StructField("salary", DoubleType(), True)
])

df = spark.read.option("header", True).schema(schema).csv("data/salary.csv")
df.printSchema()

# show data frame
df.show()

# work as data frame/data set API
# find max salary by department
df.groupBy("department_id").agg(
    max("salary").alias("max_salary")
).orderBy(
    col("max_salary").desc()
).show()

# find max 3 salaries by department using window functions
df.withColumn(
    "rank",
    rank().over(Window.partitionBy("department_id").orderBy(col("salary").desc()))
).filter(
    col("rank") <= 3
).orderBy("department_id", "rank").show()

# work as SQL
df.createOrReplaceTempView("salary")

# find max salary by department
spark.sql("SELECT department_id, max(salary) "
          "FROM salary "
          "GROUP BY department_id "
          "ORDER BY max(salary) DESC"
          ).show()

# find max 3 salaries by department using window functions
spark.sql("SELECT * FROM "
          "(SELECT department_id, employee_id, salary, "
          "RANK() OVER (PARTITION BY department_id ORDER BY salary DESC) AS rank "
          "FROM salary)"
          "WHERE rank <= 3 "
          "ORDER BY department_id, rank"
          ).show()
