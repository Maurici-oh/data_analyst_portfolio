/*
=============================================================
Create Database and Schemas
=============================================================
Script Purpose:
    This script creates a new database named 'INEGI_2020' after checking if it already exists. 
    If the database exists, it is dropped and recreated. Additionally, the script sets up three schemas 
    within the database: 'bronze', 'silver', and 'gold'.
	
WARNING:
    Running this script will drop the entire 'DataWarehouse' database if it exists. 
    All data in the database will be permanently deleted. Proceed with caution 
    and ensure you have proper backups before running this script.
*/

USE master;
GO

-- Drop and recreate the 'INEGI_2020' database
IF EXISTS (SELECT 1 FROM sys.databases WHERE name = 'INEGI_2020')
BEGIN
    ALTER DATABASE INEGI_2020 SET SINGLE_USER WITH ROLLBACK IMMEDIATE;
    DROP DATABASE INEGI_2020;
END;
GO

-- Create the 'DataWarehouse' database
CREATE DATABASE INEGI_2020;
GO

USE INEGI_2020;
GO

-- Create Schemas
CREATE SCHEMA bronze;
GO

CREATE SCHEMA silver;
GO

CREATE SCHEMA gold;
GO
