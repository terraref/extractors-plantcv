#!/usr/bin/env python

import re
import os
import logging
import requests
import json

from pyclowder.extractors import Extractor
from pyclowder.utils import CheckMessage
import pyclowder.files
import pyclowder.datasets

import PlantcvClowderIndoorAnalysis as pcia
import cv2
import plantcv as pcv

class PlantCVIndoorAnalysis(Extractor):
    def __init__(self):
        Extractor.__init__(self)

        # add any additional arguments to parser
        # self.parser.add_argument('--max', '-m', type=int, nargs='?', default=-1,
        #                          help='maximum number (default=-1)')
        self.parser.add_argument('--output', '-o', dest="output_dir", type=str, nargs='?',
                                 default="/home/extractor/sites/danforth/Level_1/plantcv",
                                 help="root directory where timestamp & output directories will be created")
        self.parser.add_argument('--betyURL', dest="bety_url", type=str, nargs='?',
                                 default="https://terraref.ncsa.illinois.edu/bety/api/beta/traits.csv",
                                 help="traits API endpoint of BETY instance that outputs should be posted to")
        self.parser.add_argument('--betyKey', dest="bety_key", type=str, nargs='?', default=False,
                                 help="API key for BETY instance specified by betyURL")

        # parse command line and load default logging configuration
        self.setup()

        # setup logging for the exctractor
        logging.getLogger('pyclowder').setLevel(logging.DEBUG)
        logging.getLogger('__main__').setLevel(logging.DEBUG)

        # assign other arguments
        self.output_dir = self.args.output_dir
        self.bety_url = self.args.bety_url
        self.bety_key = self.args.bety_key

    def check_message(self, connector, host, secret_key, resource, parameters):
        # For now if the dataset already has metadata from this extractor, don't recreate
        md = pyclowder.datasets.download_metadata(connector, host, secret_key,
                                                  resource['id'], self.extractor_info['name'])
        if len(md) > 0:
            for m in md:
                if 'agent' in m and 'name' in m['agent']:
                    if m['agent']['name'].find(self.extractor_info['name']) > -1:
                        print("skipping dataset %s, already processed" % resource['id'])
                        return CheckMessage.ignore

        # Expect at least 10 relevant files to execute this processing
        relevantFiles = 0
        for f in resource['files']:
            raw_name = re.findall(r"(VIS|NIR|vis|nir)_(SV|TV|sv|tv)(_\d+)*" , f["filename"])
            if raw_name != []:
                relevantFiles += 1

        if relevantFiles >= 10:
            return CheckMessage.download
        else:
            return CheckMessage.ignore

    def process_message(self, connector, host, secret_key, resource, parameters):
        (fields, traits) = pcia.get_traits_table()

        # get imgs paths, filter out the json paths
        img_paths = []
        for p in resource['local_paths']:
            if p[-4:] == '.jpg' or p[-4:] == '.png':
                img_paths.append(p)

            # Get metadata for avg_traits from file metadata
            elif p.endswith("_metadata.json") and not p.endswith("_dataset_metadata.json"):
                with open(p) as ds_md:
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
            file_md = pyclowder.files.download_metadata(connector, host, secret_key, f['id'])
            for md in file_md:
                if 'content' in md:
                    mdc = md['content']
                    if ('rotation_angle' in mdc) and ('perspective' in mdc) and ('camera_type' in mdc):
                        found_info = True
                        # perspective = 'side-view' / 'top-view'
                        perspective = mdc['perspective']
                        # angle = -1, 0, 90, 180, 270; set top-view angle to be -1 for later sorting
                        angle = mdc['rotation_angle'] if perspective != 'top-view' else -1
                        # camera_type = 'visible/RGB' / 'near-infrared'
                        camera_type = mdc['camera_type']

                        for pth in img_paths:
                            if re.findall(f['filename'], pth) != []:
                                file_objs.append({
                                    'perspective': perspective,
                                    'angle': angle,
                                    'camera_type': camera_type,
                                    'image_path': pth,
                                    'image_id': image_id
                                })

            if not found_info:
                # Get from filename if no metadata is found
                raw_name = re.findall(r"(VIS|NIR|vis|nir)_(SV|TV|sv|tv)(_\d+)*" , f["filename"])
                if raw_name != []:
                        raw_int = re.findall('\d+', raw_name[0][2])
                        angle = -1 if raw_int == [] else int(raw_int[0]) # -1 for top-view, else angle
                        camera_type = raw_name[0][0]
                        perspective = raw_name[0][1].lower()

                        for pth in img_paths:
                            if re.findall(f['filename'], pth) != []:
                                file_objs.append({
                                    'perspective': 'side-view' if perspective == 'sv' else 'top-view',
                                    'angle': angle,
                                    'camera_type': 'visible/RGB' if camera_type == 'vis' else 'near-infrared',
                                    'image_path': pth,
                                    'image_id': image_id
                                })

        # sort file objs by angle
        file_objs = sorted(file_objs, key=lambda k: k['angle'])

        # process images by matching angles with plantcv
        for i in [x for x in range(len(file_objs)) if x % 2 == 0]:
            if file_objs[i]['camera_type'] == 'visible/RGB':
                vis_src = file_objs[i]['image_path']
                nir_src = file_objs[i+1]['image_path']
                vis_id = file_objs[i]['image_id']
                nir_id = file_objs[i+1]['image_id']
            else:
                vis_src = file_objs[i+1]['image_path']
                nir_src = file_objs[i]['image_path']
                vis_id = file_objs[i+1]['image_id']
                nir_id = file_objs[i]['image_id']
            logging.info('...processing: %s + %s' % (os.path.basename(vis_src), os.path.basename(nir_src)))

            # Read VIS image
            img, path, filename = pcv.readimage(vis_src)
            brass_mask = cv2.imread('masks/mask_brass_tv_z1_L1.png')
            # Read NIR image
            nir, path1, filename1 = pcv.readimage(nir_src)
            nir2 = cv2.imread(nir_src, -1)

            try:
                if i == 0:
                    vn_traits = pcia.process_tv_images_core(vis_id, img, nir_id, nir, nir2, brass_mask, traits)
                else:
                    vn_traits = pcia.process_sv_images_core(vis_id, img, nir_id, nir, nir2, traits)

                logging.info("...uploading resulting metadata")
                # upload the individual file metadata
                metadata = {
                    # TODO: Generate JSON-LD context for additional fields
                    "@context": ["https://clowder.ncsa.illinois.edu/contexts/metadata.jsonld"],
                    "content": vn_traits[0],
                    "agent": {
                        "@type": "cat:extractor",
                        "extractor_id": host + "/api/extractors/" + self.extractor_info['name']
                    }
                }
                pyclowder.files.upload_metadata(connector, host, secret_key, vis_id, metadata)
                metadata = {
                    # TODO: Generate JSON-LD context for additional fields
                    "@context": ["https://clowder.ncsa.illinois.edu/contexts/metadata.jsonld"],
                    "content": vn_traits[1],
                    "agent": {
                        "@type": "cat:extractor",
                        "extractor_id": host + "/api/extractors/" + self.extractor_info['name']
                    }
                }
                pyclowder.files.upload_metadata(connector, host, secret_key, nir_id, metadata)
            except Exception as e:
                logging.error("...error generating vn_traits data; no metadata uploaded")
                logging.error(e)

        # compose the summary traits
        trait_list = pcia.generate_traits_list(traits)

        # generate output CSV & send to Clowder + BETY
        outfile = os.path.join(self.output_dir, resource['dataset_info']['name'], 'avg_traits.csv')
        logging.debug("...output file: %s" % outfile)
        out_dir = outfile.replace(os.path.basename(outfile), "")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        pcia.generate_average_csv(outfile, fields, trait_list)
        pyclowder.files.upload_to_dataset(connector, host, secret_key, resource['id'], outfile)
        submitToBety(self.bety_url, self.bety_key, outfile)

        # Flag dataset as processed by extractor
        metadata = {
            # TODO: Generate JSON-LD context for additional fields
            "@context": ["https://clowder.ncsa.illinois.edu/contexts/metadata.jsonld"],
            "dataset_id": resource['id'],
            "content": {"status": "COMPLETED"},
            "agent": {
                "@type": "cat:extractor",
                "extractor_id": host + "/api/extractors/" + self.extractor_info['name']
            }
        }
        pyclowder.datasets.upload_metadata(connector, host, secret_key, resource['id'], metadata)

def submitToBety(bety_url, bety_key, csvfile):
    if bety_url != "":
        sess = requests.Session()
        r = sess.post("%s?key=%s" % (bety_url, bety_key),
                  data=file(csvfile, 'rb').read(),
                  headers={'Content-type': 'text/csv'})

        if r.status_code == 200 or r.status_code == 201:
            logging.info("...CSV successfully uploaded to BETYdb.")
        else:
            logging.error("...error uploading CSV to BETYdb %s" % r.status_code)

if __name__ == "__main__":
    extractor = PlantCVIndoorAnalysis()
    extractor.start()
