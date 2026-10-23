# Ash mixed with Sulfate Aerosol

There are two data files that are space delimited.

data_calbuco.txt contains data from all ash-sulfate spectra from the Calbuco eruption of 2015.  
data_pcc.txt contains data from all ash-sulfate spectra from the Puyehue Cordon Caulle eruption of 2011.

| Symbol | Description | Units |  
| occ | Occultation name, ss: sunset, sr: sunrise followed by orbit number |  
| alt | Tangent altitude of the measurement |  
| year | Year of the measurement |  
| month | Month of the measurement |  
| day | Day of the measurement | day |    
| lat| Latitude at 30 km tangent height |degrees |    
| long | Longitude at 30 km tangent height | degrees |  
| chisquared | chisquared value from the least square fit | degrees |  
| mct_scale | Baseline scaling of the MCT detector (< 1835 wavenumber) | unitless |  
| mct_scale_error | Error on the baseline scaling of the MCT detector (< 1835 wavenumber) | unitless |  
| insb_scale | Baseline scaling of the InSb detector (> 1835 wavenumber) | unitless |  
| insb_scale_error | Error on the baseline scaling of the InSb detector (> 1835 wavenumber) | unitless |    
| Nl_sul | Sulfate column density | particles / cm^2 |  
| Nl_sul_error | Error in the sulfate column density | particles / cm^2 |   
| rm_sul | Sulfate median radius | micrometer |  
| rm_sul_error | Error in the sulfate median radius | micrometer |   
| Nl_ash | Ash column density | particles / cm^2 |   
| Nl_ash_error | Error in the ash column density | particles / cm^2 |  
| rm_ash | Ash median radius | micrometer |   
| rm_ash_error | Error in the ash median radius | micrometer |   
| wt_sul | Sulfuric acid (H2SO4) weight percentage | % |   
| T | Temperature | K |   


The third file ash-sulfate_missionwide.txt identifies all spectra in the ace dataset confirmed to be ash-sulfate mixtures. 
Spectra are assigned to either Calbuco, Puyehue Cordon Caulle or other. 