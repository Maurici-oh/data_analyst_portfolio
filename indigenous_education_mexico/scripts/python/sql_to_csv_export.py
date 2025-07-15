# Import necessary libraries
from sqlalchemy.engine import URL
from sqlalchemy import create_engine, text
import pandas as pd

# Define a class to manage SQL Server connections and data operations
class ConnectionHandler:
    def __init__(self, host, user, password, db):
        # Store database credentials
        self.host = host
        self.user = user
        self.password = password
        self.db = db
        
        # Define the ODBC driver for SQL Server
        driver = "ODBC Driver 17 for SQL Server"
        
        # Build the raw ODBC connection string
        connection_string = (
            f"DRIVER={{{driver}}};"
            f"SERVER={self.host};"
            f"DATABASE={self.db};"
            f"UID={self.user};"
            f"PWD={self.password};"
            f"TrustServerCertificate=yes;"  # Skip SSL certificate validation
        )
        
        # Create a SQLAlchemy-compatible connection URL
        connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})
        
        # Create SQLAlchemy engine and establish a connection
        self.engine = create_engine(connection_url, use_setinputsizes=False, echo=False)
        self.db_connection = self.engine.connect()

    # Fetch data as a pandas DataFrame from a SQL query
    def fetch_data(self, query):
        return pd.read_sql(query, con=self.db_connection)

    # Insert a pandas DataFrame into a SQL table
    def insert_data(self, df, tablename, truncate=False, chunksize=1000):
        # Optionally truncate the table before inserting
        if truncate:
            self.db_connection.execute(text(f"TRUNCATE TABLE {tablename}"))

        # Split tablename into schema and table (e.g., 'gold.fact_table')
        schema, table = tablename.split(".") if "." in tablename else (None, tablename)

        # Use pandas to insert the data into SQL
        df.to_sql(
            name=table,
            schema=schema,
            con=self.db_connection,
            if_exists='append',  # Append data to the existing table
            index=False,         # Don't write DataFrame index to the DB
            chunksize=chunksize  # Number of rows to write at a time
        )

    # Execute a raw SQL query without fetching results
    def execute_query(self, query):
        self.db_connection.execute(text(query))

    # Destructor to close the database connection gracefully
    def __del__(self):
        try:
            self.db_connection.close()
        except:
            pass  # Ignore errors during object destruction

# -----------------------------------------------------------------------------
# ::: USAGE SECTION :::
# Replace the following variables with your actual connection credentials
host = 'HOST_NAME'
user = 'USER_NAME'
password = 'PASSWORD'
database = 'DATABASE'

# Create an instance of the connection handler
CH = ConnectionHandler(host, user, password, database)

# Define the SQL query to retrieve data from the gold layer
query = """
SELECT * FROM gold.fact_indigenous_education
ORDER BY idx
"""

# Fetch data from the database
df = CH.fetch_data(query)
print(df.head())  # Display the first few rows in the console

# Export the data to a CSV file
df.to_csv("C:/Desktop/facts_indigenous_education.csv", index=False)

# Confirm export
print("Data exported to facts_indigenous_education.csv")
