/*
=========================================================================================
Stored Procedure: Load Bronze Layer (Source -> csv files)
=========================================================================================
Script Purpose:
    This stored procedure loads data into the 'bronze' schema from external CSV files. 
    It performs the following actions:
    - Truncates the bronze tables before loading data.
    - Uses the `BULK INSERT` command to load data from csv Files to bronze tables.

Parameters:
    None. 
	  This stored procedure does not accept any parameters or return any values.

Usage Example:
    EXEC bronze.load_bronze;
=========================================================================================
*/

CREATE OR ALTER PROCEDURE bronze.load_bronze AS 
BEGIN
	DECLARE @start_time DATETIME, @end_time DATETIME, @batch_start_time DATETIME, @batch_end_time DATETIME;

	BEGIN TRY
		SET @batch_start_time = GETDATE();
		PRINT '=============================================';
		PRINT '   ::: Loading Bronze Layer :::';
		PRINT '=============================================';

		PRINT '---------------------------------------------';
		PRINT 'Loading first table: location';
		PRINT '---------------------------------------------';

		SET @start_time = GETDATE();
		PRINT '>> Truncating Table: bronze.location';
		TRUNCATE TABLE bronze.location;
		PRINT '>> Inserting Data Into: bronze.location';
		BULK INSERT bronze.location 
		FROM 'C:\SQLData\INEGI_2020\datasets\facts_and_dimension\location.csv'
		WITH (
			-- Ignore the first (title) row
			FIRSTROW = 2,
			-- Define delimiter
			FIELDTERMINATOR = ',',
			FORMAT = 'CSV',
			-- Lock the table as it loads
			TABLOCK 
		);
		SET @end_time = GETDATE();
		PRINT '>> Load Duration: ' + CAST(DATEDIFF(SECOND, @start_time, @end_time) AS NVARCHAR) + ' seconds.';

		PRINT '---------------------------------------------';
		PRINT 'Loading second table: population';
		PRINT '---------------------------------------------';

		SET @start_time = GETDATE();
		PRINT '>> Truncating Table: bronze.population';
		TRUNCATE TABLE bronze.population;
		PRINT '>> Inserting Data Into: bronze.population';
		BULK INSERT bronze.population
		FROM 'C:\SQLData\INEGI_2020\datasets\facts_and_dimension\population.csv'
		WITH (
			FIRSTROW = 2,
			FIELDTERMINATOR = ',',
			FORMAT = 'CSV',
			TABLOCK 
		);
		SET @end_time = GETDATE();
		PRINT '>> Load Duration: ' + CAST(DATEDIFF(SECOND, @start_time, @end_time) AS NVARCHAR) + ' seconds.';
		
		PRINT '---------------------------------------------';
		PRINT 'Loading third table: indigenous';
		PRINT '---------------------------------------------';

		SET @start_time = GETDATE();
		PRINT '>> Truncating Table: bronze.indigenous';
		TRUNCATE TABLE bronze.indigenous;
		PRINT '>> Inserting Data Into: bronze.indigenous';
		BULK INSERT bronze.indigenous
		FROM 'C:\SQLData\INEGI_2020\datasets\facts_and_dimension\ind_population.csv'
		WITH (
			FIRSTROW = 2,
			FIELDTERMINATOR = ',',
			FORMAT = 'CSV',
			TABLOCK 
		);
		SET @end_time = GETDATE();
		PRINT '>> Load Duration: ' + CAST(DATEDIFF(SECOND, @start_time, @end_time) AS NVARCHAR) + ' seconds.';

		PRINT '---------------------------------------------';
		PRINT 'Loading fourth table: education';
		PRINT '---------------------------------------------';

		SET @start_time = GETDATE();
		PRINT '>> Truncating Table: bronze.education';
		TRUNCATE TABLE bronze.education;
		PRINT '>> Inserting Data Into: bronze.education';
		BULK INSERT bronze.education
		FROM 'C:\SQLData\INEGI_2020\datasets\facts_and_dimension\education.csv'
		WITH (
			FIRSTROW = 2,
			FIELDTERMINATOR = ',',
			FORMAT = 'CSV',
			TABLOCK 
		);
		SET @end_time = GETDATE();
		PRINT '>> Load Duration: ' + CAST(DATEDIFF(SECOND, @start_time, @end_time) AS NVARCHAR) + ' seconds.';
		
		SET @batch_end_time = GETDATE();
		PRINT '=============================================';
		PRINT 'Loading Bronze Layer is Completed'
		PRINT '> Total Load Duration; ' + CAST(DATEDIFF(SECOND, @batch_start_time, @batch_end_time) AS NVARCHAR) + ' seconds.';
		PRINT '=============================================';
	END TRY

	BEGIN CATCH
		PRINT '=============================================';
		PRINT 'ERROR OCURRED DURING LOADING BRONZE LAYER'
		PRINT 'Error Message ' + ERROR_MESSAGE();
		PRINT 'Error Number ' + CAST (ERROR_NUMBER() AS NVARCHAR);
		PRINT 'Error State ' + CAST (ERROR_STATE() AS NVARCHAR);
		PRINT '=============================================';
	END CATCH
END
