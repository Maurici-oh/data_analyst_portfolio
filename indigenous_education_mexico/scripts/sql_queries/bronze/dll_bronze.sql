/*
===============================================================================
DDL Script: Create Bronze Tables (INEGI_2020)
===============================================================================
Script Purpose:
    This script creates tables in the 'bronze' schema, dropping existing tables 
    if they already exist.
	 
	Run this script to re-define the DDL structure of 'bronze' Tables
===============================================================================
*/
PRINT '---------------------------------------------';
PRINT 'Creating first table: location';
PRINT '---------------------------------------------';
IF OBJECT_ID ('bronze.location', 'U') IS NOT NULL -- U is for user created table
	DROP TABLE bronze.location;
PRINT '>> Droping table if exists: bronze.location';
GO

CREATE TABLE bronze.location (
	ENTIDAD			INT,
	NOM_ENT			NVARCHAR(50),
	MUN				INT,
	NOM_MUN			NVARCHAR(100),
	LOC				INT,
	NOM_LOC			NVARCHAR(100),
	LONGITUD		FLOAT,
	LATITUD			FLOAT
);
PRINT '>> Table created: bronze.location';
GO

PRINT '---------------------------------------------';
PRINT 'Creating second table: population';
PRINT '---------------------------------------------';
IF OBJECT_ID ('bronze.population', 'U') IS NOT NULL
	DROP TABLE bronze.population;
PRINT '>> Droping table if exists: bronze.population';
GO

CREATE TABLE bronze.population (
	ENTIDAD		INT,
	MUN			INT,
	LOC			INT,
	POBTOT		INT,
	POBFEM		INT,
	POBMAS		INT,
	P_0A2		INT,
	P_0A4		INT,
	P_3YMAS		INT,
	P_5YMAS		INT, 
	P_15YMAS	INT,
	P_18YMAS	INT
);
PRINT '>> Table created: bronze.population';
GO

PRINT '---------------------------------------------';
PRINT 'Creating third table: indigenous';
PRINT '---------------------------------------------';
IF OBJECT_ID ('bronze.indigenous', 'U') IS NOT NULL
	DROP TABLE bronze.indigenous;
PRINT '>> Droping table if exists: bronze.indigenous';
GO

CREATE TABLE bronze.indigenous (
	ENTIDAD		INT,
	MUN			INT,
	LOC			INT,
	PHOG_IND	INT,
	P3YM_HLI	INT,
	P3HLINHE	INT,
	P3HLIHE		INT,
	P5_HLI		INT,
	P5_HLI_NHE	INT,
	P5_HLI_HE	INT
);
PRINT '>> Table created: bronze.ind_population';
GO

PRINT '---------------------------------------------';
PRINT 'Creating fourth table: education';
PRINT '---------------------------------------------';
IF OBJECT_ID ('bronze.education', 'U') IS NOT NULL
	DROP TABLE bronze.education;
PRINT '>> Droping table if exists: bronze.education';
GO

CREATE TABLE bronze.education (
	ENTIDAD		INT,
	MUN			INT,
	LOC			INT,
	P15YM_AN	INT,
	P15YM_SE	INT,
	P15PRI_IN	INT,
	P15PRI_CO	INT,
	P15SEC_IN	INT,
	P15SEC_CO	INT,
	P18YM_PB	INT,
	GRAPROES	FLOAT,
	GRAPROES_F	FLOAT,
	GRAPROES_M	FLOAT
);
PRINT '>> Table created: bronze.education';
GO