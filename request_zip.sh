#!/bin/bash


wget -o hennepin_county_parcels.zip 'https://gis.hennepin.us/publicgisdata/hennepin_county_parcels.zip'

unzip hennepin_county_parcels.zip -d hennepin_county_parcels/

rm -f hennepin_county_parcels.zip