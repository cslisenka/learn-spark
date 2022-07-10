from ctypes import Array

from pyspark.sql import SparkSession

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

# work as SQL
df.createOrReplaceTempView("salary")
spark.sql("SELECT * FROM salary").show()

# find max salary by department
spark.sql("SELECT department_id, max(salary) "
          "FROM salary "
          "GROUP BY department_id "
          "ORDER BY max(salary) DESC"
          ).show()

# find max 2 salaries by department using window functions
spark.sql("SELECT * FROM "
          "(SELECT department_id, employee_id, salary, "
          "RANK() OVER (PARTITION BY department_id ORDER BY salary DESC) AS rank "
          "FROM salary)"
          "WHERE rank <= 3 "
          "ORDER BY department_id, rank"
          ).show()
