/*
===============================================================================
DDL Script: Create Silver Tables (INEGI_2020)
===============================================================================
Script Purpose:
    This script creates tables in the 'silver' schema, dropping existing tables 
    if they already exist.

	Run this script to re-define the DDL structure of 'bronze' Tables
===============================================================================
*/

IF OBJECT_ID ('silver.location', 'U') IS NOT NULL -- U is for user created table
	DROP TABLE silver.location;
GO

CREATE TABLE silver.location (
	ENTIDAD			INT,
	NOM_ENT			NVARCHAR(50),
	state_abbreviation	NVARCHAR(10),
	region			NVARCHAR(50),
	MUN			INT,
	NOM_MUN			NVARCHAR(100),
	LOC			INT,
	NOM_LOC			NVARCHAR(100),
	LONGITUD		FLOAT,
	LATITUD			FLOAT,
	locality_index		INT
);

GO

IF OBJECT_ID ('silver.population', 'U') IS NOT NULL
	DROP TABLE silver.population;
GO

CREATE TABLE silver.population (
	ENTIDAD			INT,
	MUN			INT,
	LOC			INT,
	POBTOT			INT,
	POBFEM			INT,
	POBMAS			INT,
	P_0A2			INT,
	P_0A4			INT,
	P_3YMAS			INT,
	P_5YMAS			INT, 
	P_15YMAS		INT,
	P_18YMAS		INT,
	locality_index		INT
);
GO

IF OBJECT_ID ('silver.indigenous', 'U') IS NOT NULL
	DROP TABLE silver.indigenous;
GO

CREATE TABLE silver.indigenous (
	ENTIDAD			INT,
	MUN			INT,
	LOC			INT,
	PHOG_IND		INT,
	P3YM_HLI		INT,
	P3HLINHE		INT,
	P3HLIHE			INT,
	P5_HLI			INT,
	P5_HLI_NHE		INT,
	P5_HLI_HE		INT,
	locality_index		INT
);
GO

IF OBJECT_ID ('silver.education', 'U') IS NOT NULL
	DROP TABLE silver.education;
GO

CREATE TABLE silver.education (
	ENTIDAD			INT,
	MUN			INT,
	LOC			INT,
	P15YM_AN		INT,
	P15YM_SE		INT,
	P15PRI_IN		INT,
	P15PRI_CO		INT,
	P15SEC_IN		INT,
	P15SEC_CO		INT,
	P18YM_PB		INT,
	GRAPROES		FLOAT,
	GRAPROES_F		FLOAT,
	GRAPROES_M		FLOAT,
	locality_index		INT
);
GO
