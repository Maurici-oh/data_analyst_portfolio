/*
===============================================================================
DDL Script: Create Gold Views
===============================================================================
Script Purpose:
    This script creates views for the Gold layer in the data warehouse. 
    The Gold layer represents the final dimension and fact tables (Star Schema)

    Each view performs transformations and combines data from the Silver layer 
    to produce a clean, enriched, and business-ready dataset.

Usage:
    - These views can be queried directly for analytics and reporting.
===============================================================================
*/

-- =============================================================================
-- Create Dimension: gold.dim_location
-- =============================================================================
IF OBJECT_ID('gold.dim_location', 'V') IS NOT NULL
    DROP VIEW gold.dim_location;
GO

CREATE VIEW gold.dim_location AS
SELECT
	locality_index AS idx,
	ENTIDAD AS state_id,
	NOM_ENT AS state,
	state_abbreviation,
	region,
	NOM_MUN AS municipality,
	NOM_LOC AS locality
FROM silver.location;
GO

-- =============================================================================
-- Create Dimension: gold.dim_population
-- =============================================================================
IF OBJECT_ID('gold.dim_population', 'V') IS NOT NULL
    DROP VIEW gold.dim_population;
GO

CREATE VIEW gold.dim_population AS
SELECT
	locality_index AS idx,
	POBTOT AS total_population,
	P_3YMAS AS population_over_3,
	P_15YMAS AS population_over_15,
	P_18YMAS AS population_over_18
FROM silver.population;
GO

-- =============================================================================
-- Create Dimension: gold.dim_indigenous
-- =============================================================================
IF OBJECT_ID('gold.dim_indigenous', 'V') IS NOT NULL
    DROP VIEW gold.dim_indigenous;
GO

CREATE VIEW gold.dim_indigenous AS
SELECT 
	ind.locality_index AS idx,
	pop.POBTOT AS total_population,
	ind.P3YM_HLI AS indigenous_language_over_3,
	ind.P3HLINHE AS indigenous_language_over_3_no_spanish,
	ind.P3HLIHE AS indigenous_language_over_3_speaks_spanish,
	ind.PHOG_IND AS population_indigenous_households,
	CAST(ind.P3YM_HLI AS FLOAT) / NULLIF(pop.POBTOT, 0) * 100 AS perc_indigenous,
	CAST(ind.P3HLINHE AS FLOAT) / NULLIF(ind.P3YM_HLI, 0) * 100 AS perc_no_spanish,
    CAST(ind.PHOG_IND AS FLOAT) / NULLIF(pop.POBTOT, 0) * 100 AS perc_household_indigenous,
	CASE 
        WHEN CAST(ind.P3YM_HLI AS FLOAT) / NULLIF(pop.POBTOT, 0) > 0.1 THEN 'High Indigenous'
        ELSE 'Low Indigenous'
    END AS group_type

FROM silver.indigenous ind
LEFT JOIN silver.population pop
	ON ind.locality_index = pop.locality_index;
GO

-- =============================================================================
-- Create Dimension: gold.dim_education
-- =============================================================================
IF OBJECT_ID('gold.dim_education', 'V') IS NOT NULL
    DROP VIEW gold.dim_education;
GO

CREATE VIEW gold.dim_education AS
SELECT
	edu.locality_index AS idx,
	edu.P15YM_AN AS illiterate_over_15,
	edu.P15YM_SE AS no_schooling_over_15,
	edu.P15PRI_CO AS primary_complete_over_15,
	edu.P15PRI_IN AS primary_incomplete_over_15,
	edu.P15SEC_CO AS secondary_complete_over_15,
	edu.P15SEC_IN AS secondary_incomplete_over_15,
	edu.P18YM_PB AS higher_education_over_18,
	edu.GRAPROES AS avg_years_of_schooling_over_15,
	CAST(edu.P15YM_AN AS FLOAT) / NULLIF(pop.P_15YMAS, 0) * 100 AS perc_illiterate,
	CAST(edu.P15YM_SE AS FLOAT) / NULLIF(pop.P_15YMAS, 0) * 100 AS perc_no_schooling,
    CAST(edu.P18YM_PB AS FLOAT) / NULLIF(pop.P_18YMAS, 0) * 100 AS perc_higher_ed
FROM silver.education edu
LEFT JOIN silver.population pop
	ON edu.locality_index = pop.locality_index;
GO


-- =============================================================================
-- Create Fact Table: gold.fact_indigenous_education
-- =============================================================================
IF OBJECT_ID('gold.fact_indigenous_education', 'V') IS NOT NULL
    DROP VIEW gold.fact_indigenous_education;
GO

CREATE VIEW gold.fact_indigenous_education AS
SELECT
	loc.locality_index AS idx,
	loc.ENTIDAD AS state_id,
	loc.NOM_ENT AS state,
	loc.state_abbreviation,
	loc.region,
	loc.NOM_MUN AS municipality,
	loc.NOM_LOC AS locality,
	loc.LONGITUD AS longitude,
	loc.LATITUD AS latitude,
	pop.POBTOT AS total_population,
	pop.P_3YMAS AS population_over_3,
	pop.P_15YMAS AS population_over_15,
	pop.P_18YMAS AS population_over_18,
	ind.P3YM_HLI AS indigenous_language_over_3,
	ind.P3HLINHE AS indigenous_language_over_3_no_spanish,
	ind.P3HLIHE AS indigenous_language_over_3_speaks_spanish,
	ind.PHOG_IND AS population_indigenous_households,
	CAST(ind.P3YM_HLI AS FLOAT) / NULLIF(pop.POBTOT, 0) * 100 AS perc_indigenous,
	CAST(ind.P3HLINHE AS FLOAT) / NULLIF(ind.P3YM_HLI, 0) * 100 AS perc_no_spanish,
    CAST(ind.PHOG_IND AS FLOAT) / NULLIF(pop.POBTOT, 0) * 100 AS perc_household_indigenous,
	CASE 
        WHEN CAST(ind.P3YM_HLI AS FLOAT) / NULLIF(pop.POBTOT, 0) > 0.1 THEN 'High Indigenous'
        ELSE 'Low Indigenous'
    END AS group_type,
	edu.P15YM_AN AS illiterate_over_15,
	edu.P15YM_SE AS no_schooling_over_15,
	edu.P15PRI_CO AS primary_complete_over_15,
	edu.P15PRI_IN AS primary_incomplete_over_15,
	edu.P15SEC_CO AS secondary_complete_over_15,
	edu.P15SEC_IN AS secondary_incomplete_over_15,
	edu.P18YM_PB AS higher_education_over_18,
	edu.GRAPROES AS avg_years_of_schooling_over_15,
	CAST(edu.P15YM_AN AS FLOAT) / NULLIF(pop.P_15YMAS, 0) * 100 AS perc_illiterate,
	CAST(edu.P15YM_SE AS FLOAT) / NULLIF(pop.P_15YMAS, 0) * 100 AS perc_no_schooling,
    CAST(edu.P18YM_PB AS FLOAT) / NULLIF(pop.P_18YMAS, 0) * 100 AS perc_higher_ed
FROM silver.location loc
LEFT JOIN silver.population pop
	ON loc.locality_index = pop.locality_index
LEFT JOIN silver.indigenous ind
	ON loc.locality_index = ind.locality_index
LEFT JOIN silver.education edu
	ON loc.locality_index = edu.locality_index;
GO
