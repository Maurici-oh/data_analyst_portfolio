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
    Informacion Demografica y Social
        └──Censos y Conteos 
          └──Censos y Conteos de Poblacion y vivienda 
            └──2020  
              └──Principales resultados por localidad (ITER) 
                └──Estados Unidos Mexicanos
  </pre>
  
* **Folder Structure**
  <pre>
  indigenous_education_mexico/  
  ├── docs/ 
  │   └── img/
  │   │   ├── map1.png
  │   │   └── map2.png
  │   └── INEGI_iter_cpv2020_Data_Catalog.pdf
  │
  ├── scripts/
  │   ├── jupyter_notebooks/  
  │   │   ├── table_education_cpv2020.ipynb
  │   │   ├── table_indegenous_cpv2020.ipynb
  │   │   ├── table_location_cpv2020.ipynb
  │   │   └── table_population_cpv2020.ipynb
  │   │
  │   ├── python/ 
  │   │   └── sql_to_csv_export.py
  │   │
  │   └── sql_queries/ 
  │       ├── bronze/ 
  │       │   ├── dll_bronze.sql
  │       │   └── proc_load_bronze.sql
  │       ├── silver/ 
  │       │   ├── dll_silver.sql
  │       │   └── proc_load_silver.sql
  │       ├── gold/ 
  │       │   └── ddl_gold.sql
  │       │
  │       └── init_database.sql
  │
  ├── tableau/
  │
  └── README.md
  </pre>

* **Documentation**: Provide clear documentation of the data model to support both business stakeholderd and analytics teams.

## 📈 Visualuzation & Dashboard
To visualize the data and findings I created a Tableau dashboard that can be found in Tableau Public.  
[Indigenous Distribution and Education in Mexico](https://public.tableau.com/views/Indigenous_Population_Distribution_and_Education/IndigenousDashboard?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

<img src="https://github.com/Maurici-oh/data_analyst_portfolio/blob/3e5c6d626a374452760efad17a6d5db8806f8523/indigenous_education_mexico/docs/img/Indigenous_Dashboard.png" alt="alt text" width="400" height="300">

## 💡 Findings and Conclusions
The analysis revealed a clear and concerning correlation: regions with higher concentrations of Indigenous communities (specifically southern states) tend to experience significantly higher rates of illiteracy and a greater proportion of individuals with no formal schooling. This pattern is evident in states such as Oaxaca, Chiapas, and Guerrero. However, Quintana Roo stands out as an exception, with a high concentration of Indigenous communities and one of the highest rates of higher education in the country.

<img src="https://github.com/Maurici-oh/data_analyst_portfolio/blob/d702f69228991b618245b81071595504573cf833/indigenous_education_mexico/docs/img/map3.png" alt="alt text" width="400" height="300">

These findings underscore the persistent educational disparities affecting Indigenous populations in Mexico, highlighting the need for more inclusive and culturally responsive educational policies. Addressing these systemic inequalities is not only a matter of social justice but also essential for promoting equitable development and opportunity across all regions of the country.

Future research and action should aim to better understand the root causes of these disparities and support interventions that respect Indigenous cultures while improving access to quality education.

## 🛠️ Tech Stack

* **Jupyter Notebook**
* **Google Sheets**
* **Python** 
* **SQL Server**  
* **Tableau**  



