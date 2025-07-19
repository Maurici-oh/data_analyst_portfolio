# 🎓 The Intersection of Indigenous Identity and Educational Attainment in Mexico 🇲🇽

This data analytics project aims to explore the distribution of the population in Mexico that either speaks an indigenous language or belongs to an indigenous household, and analyze how this demographic characteristic correlates with levels of education and illiteracy. Indigenous populations in Mexico represent a vital part of the country’s cultural heritage, yet they often face systemic inequalities, including disparities in access to education.

Using publicly available census and survey data from the mexican National Institute of Statistics and Geography (INEGI), this project investigates spatial, demographic, and educational trends among indigenous communities. The analysis focuses on identifying patterns in indigenous language usage across different states and regions, and quantifies the relationship between these factors and educational attainment.

By visualizing and modeling this relationship, the project seeks to provide insights into how ethnicity and language are associated with educational outcomes in Mexico. These insights can inform policymakers, researchers, and social organizations aiming to reduce educational inequality and support the inclusion and empowerment of indigenous populations.

<img src="https://github.com/Maurici-oh/data_analyst_portfolio/blob/1de38015fb8c6a088af97fa75dc650fd0aedadb8/indigenous_education_mexico/docs/img/map2.png" alt="alt text" width="400" height="300">

## 📊 Project Outputs
* **Jupyter Notebooks** for importing raw data from the INEGI website, creating smaller and more manageable DataFrames (location, population, indigenous_population, education) to perform initial data cleansing and preprocessing, and exporting the results to individual CSV files.
* **Excel File** containing the data catalog and table contents.
* **SQL Scripts** for creating a SQL Server database, including DDL scripts structured using the Medallion Architecture (Bronze, Silver, Gold) and stored procedures to populate the tables with data from the CSV files.
* **Python Script** that fetches data from the database using the SQLAlchemy library and exports it to a new CSV file for visualization purposes.
* **Tableau** Dashboard featuring key metrics, tables, and charts derived from the analysis.

## 📋 Specifications

* **Data Source**: The dataset used in this project comes from the INEGI 2020 Population and Housing Census.
[INEGI open data](https://www.inegi.org.mx/datosabiertos/)

  The route to acces the data is the following:

<pre>
  **Informacion Demografica y Social**
      └──Censos y Conteos 
        └──Censos y Conteos de Poblacion y vivienda 
          └──2020  
            └──Principales resultados por localidad (ITER) 
              └──Estados Unidos Mexicanos
</pre>
  
  Informacion Demografica y Social > Censos y Conteos > Censos y Conteos de Poblacion y vivienda >
  2020 > Principales resultados por localidad (ITER) > Estados Unidos Mexicanos

* **Data Quality**: Cleanse and trim the dataset for easy handling and processing.
* **Documentation**: Provide clear documentation of the data model to support both business stakeholderd and analytics teams.

## 📈 BI: Analytics & Reporting (Data Analytics)
**Objective**  

To be added.....

## 🛠️ Tech Stack

* **Jupyter Notebook**  
* **Python** 
* **SQL Server**  
* **Tableau**  



