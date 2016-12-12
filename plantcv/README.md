# PlantCV extractor
This extractor processes VIS/NIR images captured at several angles to generate trait metadata. The trait metadata is associated with the source images in Clowder, and uploaded to the configured BETYdb instance.

_Input_

  - Evaluation is triggered whenever a file is added to a dataset
  - Following images from a Danforth Bellweather Phenotyping Facility snapshot must be found:
    - 4x NIR side-view = NIR_SV_0, NIR_SV_90, NIR_SV_180, NIR_SV_270
    - 1x NIR top-view = NIR_TV
    - 4x VIS side-view = VIS_SV_0, VIS_SV_90, VIS_SV_180, VIS_SV_270
    - 1x VIS top-view = VIS_TV
  - Per-image metadata in Clowder is required for BETYdb submission; this is how barcode/genotype/treatment/timestamp are determined.

_Output_

  - Each image will have new metadata appended in Clowder including measures like height, area, perimeter, and longest_axis
  - Average traits for the dataset (10 images) are inserted into a CSV file and added to the Clowder dataset
  - If configured, the CSV will also be sent to BETYdb

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

### Docker
The Dockerfile included in this directory can be used to launch this extractor in a container.

_Building the Docker image_
```
docker build -f Dockerfile -t terra-ext-plantcv .
```

_Running the image locally_
```
docker run \
  -p 5672 -p 9000 --add-host="localhost:{LOCAL_IP}" \
  -e RABBITMQ_URI=amqp://{RMQ_USER}:{RMQ_PASSWORD}@localhost:5672/%2f \
  -e RABBITMQ_EXCHANGE=clowder \
  -e REGISTRATION_ENDPOINTS=http://localhost:9000/clowder/api/extractors?key={SECRET_KEY} \
  terra-ext-plantcv
```
Note that by default RabbitMQ will not allow "guest:guest" access to non-local addresses, which includes Docker. You may need to create an additional local RabbitMQ user for testing.

_Running the image remotely_
```
docker run \
  -e RABBITMQ_URI=amqp://{RMQ_USER}:{RMQ_PASSWORD}@rabbitmq.ncsa.illinois.edu/clowder \
  -e RABBITMQ_EXCHANGE=terra \
  -e REGISTRATION_ENDPOINTS=http://terraref.ncsa.illinosi.edu/clowder//api/extractors?key={SECRET_KEY} \
  terra-ext-plantcv
```
