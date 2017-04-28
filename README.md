# ift6266-h17-project

Download the COCO dataset from 
http://lisaweb.iro.umontreal.ca/transfert/lisa/datasets/mscoco_inpaiting/

Build the following folder structure:
- project_root/coco/images/train2014/<images>
- project_root/coco/images/val2014/<images>

Compress your images in a numpy archive file with:
python dataprep.py

Clone this repository: https://github.com/pdollar/coco
git clone https://github.com/pdollar/coco

In PythonAPI, run setup.py:
python setup.py build_ext --inplace

Copy over the generated files in PythonAPI/pycocotools to project_root/lib/pycocotools:
- _mask.c
- _mask.pyd
_ _mask.pyx
_ *.so files if on linux

Run your model with:
python run_model.py run 'modelName'