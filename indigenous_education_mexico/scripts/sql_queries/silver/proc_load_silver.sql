/*
===============================================================================
Stored Procedure: Load Silver Layer (Bronze -> Silver) - INEGI_2020
===============================================================================
Script Purpose:
    This stored procedure performs the ETL (Extract, Transform, Load) process to 
    populate the 'silver' schema tables from the 'bronze' schema.
	Actions Performed:
		- Truncates Silver tables.
		- Inserts transformed and cleansed data from Bronze into Silver tables.
		
Parameters:
    None. 
	  This stored procedure does not accept any parameters or return any values.

Usage Example:
    EXEC silver.load_silver;
===============================================================================
*/

CREATE OR ALTER PROCEDURE silver.load_silver AS
BEGIN
    DECLARE @start_time DATETIME, @end_time DATETIME, @batch_start_time DATETIME, @batch_end_time DATETIME; 
    BEGIN TRY
        SET @batch_start_time = GETDATE();
        PRINT '================================================';
        PRINT 'Loading Silver Layer';
        PRINT '================================================';

		--PRINT '------------------------------------------------';
		--PRINT 'Loading Tables: location';
		--PRINT '------------------------------------------------';

		-- Loading silver.location
        SET @start_time = GETDATE();
		PRINT '>> Truncating Table: silver.location';
		TRUNCATE TABLE silver.location;
		PRINT '>> Inserting Data Into: silver.location';
		INSERT INTO silver.location (
			ENTIDAD,
			NOM_ENT,
			state_abbreviation,
			region,
			MUN,
			NOM_MUN,
			LOC,
			NOM_LOC,
			LONGITUD,
			LATITUD,
			locality_index
		)
		SELECT
			ENTIDAD,
			NOM_ENT,
			CASE NOM_ENT
				WHEN 'Aguascalientes' THEN 'AGU'
				WHEN 'Baja California' THEN 'BCN'
				WHEN 'Baja California Sur' THEN 'BCS'
				WHEN 'Campeche' THEN 'CAM'
				WHEN 'Chiapas' THEN 'CHP'
				WHEN 'Chihuahua' THEN 'CHH'
				WHEN 'Ciudad de Mexico' THEN 'CMX'
				WHEN 'Coahuila de Zaragoza' THEN 'COA'
				WHEN 'Colima' THEN 'COL'
				WHEN 'Durango' THEN 'DUR'
				WHEN 'Guanajuato' THEN 'GUA'
				WHEN 'Guerrero' THEN 'GRO'
				WHEN 'Hidalgo' THEN 'HID'
				WHEN 'Jalisco' THEN 'JAL'
				WHEN 'Mexico' THEN 'MEX'
				WHEN 'Michoacan de Ocampo' THEN 'MIC'
				WHEN 'Morelos' THEN 'MOR'
				WHEN 'Nayarit' THEN 'NAY'
				WHEN 'Nuevo Leon' THEN 'NLE'
				WHEN 'Oaxaca' THEN 'OAX'
				WHEN 'Puebla' THEN 'PUE'
				WHEN 'Queretaro' THEN 'QUE'
				WHEN 'Quintana Roo' THEN 'ROO'
				WHEN 'San Luis Potosi' THEN 'SLP'
				WHEN 'Sinaloa' THEN 'SIN'
				WHEN 'Sonora' THEN 'SON'
				WHEN 'Tabasco' THEN 'TAB'
				WHEN 'Tamaulipas' THEN 'TAM'
				WHEN 'Tlaxcala' THEN 'TLA'
				WHEN 'Veracruz de Ignacio de la Llave' THEN 'VER'
				WHEN 'Yucatan' THEN 'YUC'
				WHEN 'Zacatecas' THEN 'ZAC'
				ELSE 'Total'
			END AS state_abbreviation,
			CASE 
				WHEN ENTIDAD IN (2, 3, 25, 26) THEN 'Northwest'
				WHEN ENTIDAD IN (8, 5, 10, 19, 28) THEN 'Northeast'
				WHEN ENTIDAD IN (18, 14, 6, 16) THEN 'West'
				WHEN ENTIDAD IN (1, 11, 22, 24, 32) THEN 'Midwest'
				WHEN ENTIDAD IN (9, 15, 13, 17, 21, 29) THEN 'Center'
				WHEN ENTIDAD IN (30, 27) THEN 'East'
				WHEN ENTIDAD IN (12, 20, 7) THEN 'South'
				WHEN ENTIDAD IN (4, 23, 31) THEN 'Southeast'
				ELSE 'Total'
			END AS region,
			MUN,
			NOM_MUN,
			LOC,
			NOM_LOC,
			LONGITUD,
			LATITUD,
			ROW_NUMBER() OVER (ORDER BY ENTIDAD, MUN, LOC) AS locality_index
		FROM bronze.location
		WHERE LOC NOT IN (0, 9999, 9998)
		ORDER BY ENTIDAD, MUN, LOC;
		SET @end_time = GETDATE();
        PRINT '>> Load Duration: ' + CAST(DATEDIFF(SECOND, @start_time, @end_time) AS NVARCHAR) + ' seconds';
        PRINT '>> ---------------------------------------------';

		-- Loading silver.population
        SET @start_time = GETDATE();
		PRINT '>> Truncating Table: silver.population';
		TRUNCATE TABLE silver.population;
		PRINT '>> Inserting Data Into: silver.population';
		INSERT INTO silver.population (
			ENTIDAD,
			MUN,
			LOC,
			POBTOT,
			POBFEM,
			POBMAS,
			P_0A2,
			P_0A4,
			P_3YMAS,
			P_5YMAS,
			P_15YMAS,
			P_18YMAS,
			locality_index
		)
		SELECT
			ENTIDAD,
			MUN,
			LOC,
			POBTOT,
			POBFEM,
			POBMAS,
			P_0A2,
			P_0A4,
			P_3YMAS,
			P_5YMAS, 
			P_15YMAS,
			P_18YMAS,
			ROW_NUMBER() OVER (ORDER BY ENTIDAD, MUN, LOC) AS locality_index
		FROM bronze.population
		WHERE LOC NOT IN (0, 9999, 9998)
		ORDER BY ENTIDAD, MUN, LOC;
        SET @end_time = GETDATE();
        PRINT '>> Load Duration: ' + CAST(DATEDIFF(SECOND, @start_time, @end_time) AS NVARCHAR) + ' seconds';
        PRINT '>> ---------------------------------------------';

        -- Loading silver.indigenous
        SET @start_time = GETDATE();
		PRINT '>> Truncating Table: silver.indigenous';
		TRUNCATE TABLE silver.indigenous;
		PRINT '>> Inserting Data Into: silver.indigenous';
		INSERT INTO silver.indigenous (
			ENTIDAD,
			MUN,
			LOC,
			PHOG_IND,
			P3YM_HLI,
			P3HLINHE,
			P3HLIHE,
			P5_HLI,
			P5_HLI_NHE,
			P5_HLI_HE,
			locality_index
		)
		SELECT
			ENTIDAD,
			MUN,
			LOC,
			PHOG_IND,
			P3YM_HLI,
			P3HLINHE,
			P3HLIHE,
			P5_HLI,
			P5_HLI_NHE,
			P5_HLI_HE,
			ROW_NUMBER() OVER (ORDER BY ENTIDAD, MUN, LOC) AS locality_index
		FROM bronze.indigenous
		WHERE LOC NOT IN (0, 9999, 9998)
		ORDER BY ENTIDAD, MUN, LOC;
        SET @end_time = GETDATE();
        PRINT '>> Load Duration: ' + CAST(DATEDIFF(SECOND, @start_time, @end_time) AS NVARCHAR) + ' seconds';
        PRINT '>> ---------------------------------------------';

        -- Loading silver.education
        SET @start_time = GETDATE();
		PRINT '>> Truncating Table: silver.education';
		TRUNCATE TABLE silver.education;
		PRINT '>> Inserting Data Into: silver.education';
		INSERT INTO silver.education (
			ENTIDAD,
			MUN,
			LOC,
			P15YM_AN,
			P15YM_SE,
			P15PRI_IN,
			P15PRI_CO,
			P15SEC_IN,
			P15SEC_CO,
			P18YM_PB,
			GRAPROES,
			GRAPROES_F,
			GRAPROES_M,
			locality_index
		)
		SELECT
			ENTIDAD,
			MUN,
			LOC,
			P15YM_AN,
			P15YM_SE,
			P15PRI_IN,
			P15PRI_CO,
			P15SEC_IN,
			P15SEC_CO,
			P18YM_PB,
			GRAPROES,
			GRAPROES_F,
			GRAPROES_M,
			ROW_NUMBER() OVER (ORDER BY ENTIDAD, MUN, LOC) AS locality_index
		FROM bronze.education
		WHERE LOC NOT IN (0, 9999, 9998)
		ORDER BY ENTIDAD, MUN, LOC;
	    SET @end_time = GETDATE();
        PRINT '>> Load Duration: ' + CAST(DATEDIFF(SECOND, @start_time, @end_time) AS NVARCHAR) + ' seconds';
        PRINT '>> ---------------------------------------------';

		SET @batch_end_time = GETDATE();
		PRINT '================================================'
		PRINT 'Loading Silver Layer is Completed';
        PRINT '   - Total Load Duration: ' + CAST(DATEDIFF(SECOND, @batch_start_time, @batch_end_time) AS NVARCHAR) + ' seconds';
		PRINT '================================================'
		
	END TRY
	BEGIN CATCH
		PRINT '================================================'
		PRINT 'ERROR OCCURED DURING LOADING BRONZE LAYER'
		PRINT 'Error Message' + ERROR_MESSAGE();
		PRINT 'Error Message' + CAST (ERROR_NUMBER() AS NVARCHAR);
		PRINT 'Error Message' + CAST (ERROR_STATE() AS NVARCHAR);
		PRINT '================================================'
	END CATCH
END
