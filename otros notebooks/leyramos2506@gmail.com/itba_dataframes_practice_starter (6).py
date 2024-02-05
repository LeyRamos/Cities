# Databricks notebook source
# MAGIC %md
# MAGIC ## Analytics con DataFrames sobre datos semi estructurados 

# COMMAND ----------

import urllib

with urllib.request.urlopen('https://github.com/juanpampliega/datasets/raw/master/http_access_200304.log.gz') as response:
  gzipcontent = response.read()

with open("/tmp/test_http_access_log.gz", 'wb') as f:
  f.write(gzipcontent)

dbutils.fs.cp("file:/tmp/test_http_access_log.gz", "/tmp/")

# COMMAND ----------

# access_logs = sc.textFile("dbfs:/tmp/test_http_access_log.gz")
access_logs = sc.textFile("file:/tmp/test_http_access_log.gz")
access_logs.take(5)

# COMMAND ----------

import re
import json

def parse_access_log_line(line):
  
  format_pat= re.compile( 
      r"(?P<host>[\d\.]+)\s" 
      r"(?P<identity>\S*)\s" 
      r"(?P<user>\S*)\s"
      r"\[(?P<time>.*?)\]\s"
      r'"(?P<request>.*?)"\s'
      r"(?P<status>\d+)\s"
      r"(?P<bytes>\S*)\s"
      r'"(?P<referer>.*?)"\s' # [SIC]
      r'"(?P<user_agent>.*?)"\s*' 
  )

  match = format_pat.match(line)
  return json.dumps(match.groupdict())


# COMMAND ----------

access_logs = sc.textFile("file:/tmp/test_http_access_log.gz")
access_logs_json = access_logs.map(parse_access_log_line)

access_logs_json.take(5)


# COMMAND ----------

logs_df = spark.read.json(access_logs_json)

logs_df.printSchema()
logs_df.show()

logs_df.createOrReplaceTempView("logs")

# COMMAND ----------

logs_df.createOrReplaceTempView("logs")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   time,
# MAGIC   unix_timestamp(substring(time,0,11), 'dd/MMM/yyyy'),
# MAGIC   from_unixtime(unix_timestamp(substring(time,0,11), 'dd/MMM/yyyy'),"yyyy-MM-dd")
# MAGIC FROM logs
# MAGIC limit 7

# COMMAND ----------

sdf = spark.sql("""
SELECT 
  from_unixtime(unix_timestamp(substring(time,0,11), 'dd/MMM/yyyy'),"yyyy-MM-dd") as day,
  status,
  count(1) as mount
FROM logs
GROUP BY from_unixtime(unix_timestamp(substring(time,0,11), 'dd/MMM/yyyy'),"yyyy-MM-dd"), status
ORDER BY 1, 2
""")
sdf.show()

display(sdf)

# COMMAND ----------

#print(logs)
statuses = logs.groupBy("status").count().orderBy("count")
display(statuses)


# COMMAND ----------

files = ['ipligence-lite.csv', 'http_access_200304.log.gz', 'http_access_200306.log.gz', 'http_access_200307.log.gz']

# COMMAND ----------

def download_file(file):
  with urllib.request.urlopen('https://github.com/juanpampliega/datasets/raw/master/{f}'.format(f=file)) as response:
    gzipcontent = response.read()

  with open('/tmp/{f}'.format(f=file), 'wb') as f:
    f.write(gzipcontent)

# COMMAND ----------

for f in files:
  download_file(f)

# COMMAND ----------

logs2 = sc.textFile("file:/tmp/http_access_2003*")
records = logs2.map(parse_access_log_line)

records.take(2)

# COMMAND ----------

all_logs = spark.read.json(records)

all_logs.printSchema()

all_logs.createOrReplaceTempView("all_logs")

# COMMAND ----------

geo_ip = sqlContext.read.format('csv').options(header='true', inferSchema='true').load('dbfs:/tmp/ipligence-lite.csv')

geo_ip.createOrReplaceTempView("geo_ips")

display(geo_ip)

# COMMAND ----------

# MAGIC %scala
# MAGIC def ipToNumber(ipAddr: String): Long = {
# MAGIC   try {
# MAGIC     val parts = ipAddr.split("\\.")
# MAGIC     parts(3).toLong + (parts(2).toLong * 256L) + (parts(1).toLong * 256L * 256L) + (parts(0).toLong * 256L * 256L * 256L)
# MAGIC   } catch {
# MAGIC     case e: Exception => {
# MAGIC       e.printStackTrace
# MAGIC       0
# MAGIC     }
# MAGIC   }
# MAGIC }
# MAGIC
# MAGIC sqlContext.udf.register("INET_ATON", (ip:String) => ipToNumber(ip))

# COMMAND ----------

# MAGIC %sql CACHE TABLE tbl_ip_country AS
# MAGIC SELECT all_logs.host, geo_ips.country_iso
# MAGIC FROM all_logs INNER JOIN geo_ips
# MAGIC     ON 
# MAGIC     geo_ips.from_ip <= INET_ATON(all_logs.host) AND 
# MAGIC     geo_ips.to_ip >= INET_ATON(all_logs.host)
# MAGIC --LIMIT 100

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT country_iso, COUNT(1) AS count 
# MAGIC FROM tbl_ip_country
# MAGIC GROUP BY country_iso
# MAGIC ORDER BY count DESC
# MAGIC LIMIT 8

# COMMAND ----------

# https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT 
# MAGIC  CASE WHEN country_iso = 'US' THEN 'USA'
# MAGIC       WHEN country_iso = 'NL' THEN 'NLD'
# MAGIC       WHEN country_iso = 'CA' THEN 'CAN'
# MAGIC       WHEN country_iso = 'SE' THEN 'SRB'
# MAGIC       WHEN country_iso = 'GB' THEN 'GBR'
# MAGIC       WHEN country_iso = 'AU' THEN 'AUS'
# MAGIC       WHEN country_iso = 'IT' THEN 'ITA'
# MAGIC       WHEN country_iso = 'SA' THEN 'ZAF'
# MAGIC       ELSE NULL END as country_iso, 
# MAGIC COUNT(1) AS count 
# MAGIC FROM tbl_ip_country
# MAGIC GROUP BY country_iso
# MAGIC ORDER BY count DESC
# MAGIC LIMIT 8