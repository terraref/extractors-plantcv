#!/usr/bin/env python
import re
import os
import logging
import json

from pyclowder.utils import CheckMessage
from pyclowder.datasets import download_metadata as download_ds_metadata, \
    upload_metadata as upload_ds_metadata
from pyclowder.files import download_metadata as download_file_metadata, \
    upload_metadata as upload_file_metadata, upload_to_dataset
from terrautils.extractors import TerrarefExtractor, is_latest_file, build_metadata
from terrautils.betydb import add_arguments, submit_traits

import PlantcvClowderIndoorAnalysis as pcia
import cv2
import plantcv as pcv


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def add_local_arguments(parser):
    # add any additional arguments to parser
    add_arguments(parser)

class PlantCVIndoorAnalysis(TerrarefExtractor):
    def __init__(self):
        super(PlantCVIndoorAnalysis, self).__init__()

        add_local_arguments(self.parser)

        # parse command line and load default logging configuration
        self.setup(sensor='plantcv', site='danforth')

        # assign other arguments
        self.bety_url = self.args.bety_url
        self.bety_key = self.args.bety_key

    def check_message(self, connector, host, secret_key, resource, parameters):
        # For now if the dataset already has metadata from this extractor, don't recreate
        md = download_ds_metadata(connector, host, secret_key,
                               resource['id'], self.extractor_info['name'])
        if len(md) > 0:
            for m in md:
                if 'agent' in m and 'name' in m['agent']:
                    if m['agent']['name'].find(self.extractor_info['name']) > -1:
                        logger.info("skipping dataset %s, already processed" % resource['id'])
                        return CheckMessage.ignore

        # Expect at least 10 relevant files to execute this processing
        relevantFiles = 0
        for f in resource['files']:
            raw_name = re.findall(r"(VIS|NIR|vis|nir)_(SV|TV|sv|tv)(_\d+)*", f["filename"])
            if raw_name:
                relevantFiles += 1

        if relevantFiles >= 6:
            return CheckMessage.download
        else:
            logger.info("not enough relevant files in %s (found %s)" % (resource['id'], relevantFiles))
            return CheckMessage.ignore

    def process_message(self, connector, host, secret_key, resource, parameters):
        self.start_message()

        (fields, traits) = pcia.get_traits_table()

        # get imgs paths, filter out the json paths
        img_paths = []
        for p in resource['local_paths']:
            if p[-4:] == '.jpg' or p[-4:] == '.png':
                img_paths.append(p)

            # Get metadata for avg_traits from file metadata
            elif p.endswith("_metadata.json") and not p.endswith("_dataset_metadata.json"):
                with open(p) as ds_md:
                    logger.info("loading metadata from %s" % p)
                    md_set = json.load(ds_md)
                    for md in md_set:
                        if 'content' in md:
                            needed_fields = ['plant_barcode', 'genotype', 'treatment', 'imagedate']
                            for fld in needed_fields:
                                if traits[fld] == '' and fld in md['content']:
                                    if fld == 'imagedate':
                                        traits[fld] = md['content'][fld].replace(" ", "T")
                                    else:
                                        traits[fld] = md['content'][fld]

        # build list of file descriptor dictionaries with sensor info
        file_objs = []
        for f in resource['files']:
            found_info = False
            image_id = f['id']
            # Get from file metadata if possible
            logger.info("Downloading file metadata from %s" % f['id'])
            file_md = download_file_metadata(connector, host, secret_key, f['id'])

            for md in file_md:
                if 'content' in md:
                    mdc = md['content']
                    if ('rotation_angle' in mdc) and ('perspective' in mdc) and ('camera_type' in mdc):
                        found_info = True

                        # experiment ID determines what PlantCV code gets executed
                        if 'experiment_id' not in mdc:
                            logger.info("Attempting to determine experiment ID from %s" % f['filepath'])
                            # e.g. danforth/raw_data/TM015_F_051616/snapshot24815/NIR_SV_0_z1_h1_g0_e15000_v500_51959_0.png
                            try:
                                snapdir = os.path.split(f['filepath'])[0]
                                experiment = snapdir.split("/")[-2]
                            except:
                                logger.error("Unable to determine experiment ID from path.")
                                experiment = "unknown"
                        else:
                            experiment = mdc['experiment_id']
                        # perspective = 'side-view' / 'top-view'
                        perspective = mdc['perspective']
                        # angle = -1, 0, 90, 180, 270; set top-view angle to be -1 for later sorting
                        angle = mdc['rotation_angle'] if perspective != 'top-view' else -1
                        # camera_type = 'visible/RGB' / 'near-infrared'
                        camera_type = mdc['camera_type']

                        for pth in img_paths:
                            if re.findall(f['filename'], pth):
                                file_objs.append({
                                    'perspective': perspective,
                                    'angle': angle,
                                    'camera_type': camera_type,
                                    'image_path': pth,
                                    'image_id': image_id,
                                    'experiment_id': experiment,
                                    'filename': f['filename']
                                })

            if not found_info:
                logger.info("No file metadata found for %s; parsing from filename" % f['id'])
                # Get from filename if no metadata is found
                raw_name = re.findall(r"(VIS|NIR|vis|nir)_(SV|TV|sv|tv)(_\d+)*", f["filename"])
                if raw_name:
                    raw_int = re.findall('\d+', raw_name[0][2])
                    angle = -1 if raw_int == [] else int(raw_int[0])  # -1 for top-view, else angle
                    camera_type = raw_name[0][0]
                    perspective = raw_name[0][1].lower()

                    for pth in img_paths:
                        if re.findall(f['filename'], pth):
                            file_objs.append({
                                'perspective': 'side-view' if perspective == 'sv' else 'top-view',
                                'angle': angle,
                                'camera_type': 'visible/RGB' if camera_type == 'vis' else 'near-infrared',
                                'image_path': pth,
                                'image_id': image_id,
                                'experiment_id': 'unknown',
                                'filename': f['filename']
                            })

        # sort file objs by angle
        file_objs = sorted(file_objs, key=lambda k: k['angle'])

        # process images by matching angles with plantcv
        for i in [x for x in range(len(file_objs)) if x % 2 == 0]:
            if file_objs[i]['camera_type'] == 'visible/RGB':
                vis_src = file_objs[i]['image_path']
                nir_src = file_objs[i + 1]['image_path']
                vis_id = file_objs[i]['image_id']
                nir_id = file_objs[i + 1]['image_id']
                experiment_id = file_objs[i]['experiment_id']
                vis_filename = file_objs[i]['filename']
                nir_filename = file_objs[i + 1]['filename']
            else:
                vis_src = file_objs[i + 1]['image_path']
                nir_src = file_objs[i]['image_path']
                vis_id = file_objs[i + 1]['image_id']
                nir_id = file_objs[i]['image_id']
                experiment_id = file_objs[i + 1]['experiment_id']
                vis_filename = file_objs[i + 1]['filename']
                nir_filename = file_objs[i]['filename']
            logger.info('Processing: %s + %s' % (os.path.basename(vis_src), os.path.basename(nir_src)))

            # Read VIS image
            img, path, filename = pcv.readimage(vis_src)
            brass_mask = cv2.imread('masks/mask_brass_tv_z1_L1.png')
            # Read NIR image
            nir, path1, filename1 = pcv.readimage(nir_src)
            nir2 = cv2.imread(nir_src, -1)

            try:
                out_dir = os.path.join(self.sensors.base, 'danforth', 'Level_1', 'plantcv', experiment,
                                       resource['dataset_info']['name'])
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                vis_out = os.path.join(out_dir, vis_filename)
                nir_out = os.path.join(out_dir, nir_filename)
                if i == 0:
                    logger.info("Processing TV images")
                    vn_traits = pcia.process_tv_images_core(vis_id, img, nir_id, nir, nir2, brass_mask, traits,
                                                            experiment_id, vis_out, nir_out)
                else:
                    logger.info("Processing SV images")
                    vn_traits = pcia.process_sv_images_core(vis_id, img, nir_id, nir, nir2, traits, experiment_id,
                                                            vis_out, nir_out)

                logger.info("Uploading resulting metadata")
                # upload the individual file metadata
                metadata = build_metadata(host, self.extractor_info, nir_id, vn_traits[0], 'file')
                upload_file_metadata(connector, host, secret_key, vis_id, metadata)
                metadata = build_metadata(host, self.extractor_info, nir_id, vn_traits[1], 'file')
                upload_file_metadata(connector, host, secret_key, nir_id, metadata)
                # Add PlantCV analysis images to dataset
                logger.info("Uploading %s analysis images" % len(vn_traits[2]))
                for image in vn_traits[2]:
                    upload_to_dataset(connector, host, secret_key, resource['id'], image)
                    self.created += 1
                    self.bytes += os.path.getsize(image)

            except Exception as e:
                logger.error("Error generating vn_traits data; no metadata uploaded")
                logger.error(e)

        # compose the summary traits
        trait_list = pcia.generate_traits_list(traits)

        # generate output CSV & send to Clowder + BETY
        logger.info("Error generating vn_traits data; no metadata uploaded")
        tmp_csv = 'avg_traits.csv'
        pcia.generate_average_csv(tmp_csv, fields, trait_list)
        submit_traits(tmp_csv, self.bety_key)
        os.remove(tmp_csv)

        # Flag dataset as processed by extractor
        metadata = build_metadata(host, self.extractor_info, resource['id'],
                                  {"status": "COMPLETED"}, 'dataset')
        upload_ds_metadata(connector, host, secret_key, resource['id'], metadata)

        self.end_message()

if __name__ == "__main__":
    extractor = PlantCVIndoorAnalysis()
    extractor.start()
