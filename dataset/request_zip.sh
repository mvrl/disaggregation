#!/bin/bash


wget 'https://gis.hennepin.us/publicgisdata/hennepin_county_parcels.zip'

wget 'https://usbuildingdata.blob.core.windows.net/usbuildings-v1-1/Minnesota.zip'

unzip hennepin_county_parcels.zip -d hennepin_county_parcels/

unzip Minnesota.zip -d hennepin_county_parcels/

ogr2ogr hennepin_county_parcels/Minnesota_ESPG26915.shp -t_srs "EPSG:26915" hennepin_county_parcels/Minnesota.geojson

rm -f hennepin_county_parcels.zip

rm -f Minnesota.zip