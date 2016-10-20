# PlantCV extractor
This extractor triggers when ten images from a Danforth Bellweather Phenotyping Facility snapshot are uploaded to Clowder.
 - 4x NIR side-view = NIR_SV_0, NIR_SV_90, NIR_SV_180, NIR_SV_270
 - 1x NIR top-view = NIR_TV
 - 4x VIS side-view = VIS_SV_0, VIS_SV_90, VIS_SV_180, VIS_SV_270
 - 1x VIS top-view = VIS_TV

It invokes PlantCV image analyses to generate per-image metadata and a per-snapshot trait data CSV (averaged across images).

### BETYdb integration
If a BETYdb API URL and key are configured, the extractor will attempt to push the traits CSV into that BETY instance. 

The following fields (with sample value) are generated in the CSV:
- entity = Fp001AB006772
- cultivar = BTx642
- treatment = 80%: 173.6 ml water (37.5% VWC)
- local_datetime = 2014-06-05T16:50:49.828
- sv_area = 35109.25
- tv_area = 796.0
- hull_area = 419271.625
- solidity = 0.693557208424
- height = 458
- perimeter = 4422.85527804
- access_level = 2
- species = Sorghum bicolor
- site = Danforth Plant Science Center Bellweather Phenotyping Facility
- citation_author = Fahlgren, Noah
- citation_year = 2016
- citation_title = Unpublished Data from Sorghum Test Runs
- method = PlantCV

The BETYdb upload may fail under some normal circumstances. For example, very young/small plants may not be identifiable in PlantCV, meaning traits cannot be determined. 

In cases like these the resulting average traits will be blank and BETY will disallow uploading. In cases where small plants are visible from _some_ images but not all, traits can still be determined.
