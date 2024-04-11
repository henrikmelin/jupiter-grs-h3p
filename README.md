# Ionospheric irregularities at Jupiter observed by JWST
This is a code and data repository for the manuscript 'Ionospheric irregularities at Jupiter observed by JWST' by Melin et al.
 
The Figure source data files are located in the `source_data` directory. 

The `h3ppy` spectral fits are contained in the `spectral_fits` directory. 

The Jupyter notebook `Melin2024_Jupiter_GRS_h3p.ipynb` contains the code to produce the figures. 

Other scripts include: 
* `ERS_GRS_extract_ions.py` - extract the H3+ spectrum by subtracting the CH4 component. 
* `ERS_GRS_fit_h3p.py` - fit the extracted H3+ spectrum using `h3ppy`.
* `JWSTSolarSystemPointing.py` - assign observational geometry to the JWST NIRSpec data. 
* `ch4.py` - the simplified modelling of non-LTE CH4. 