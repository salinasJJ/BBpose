#!/bin/bash

while getopts 'r:I:D:T:R:j:a:m:d:i:' OPTION; do
    case "${OPTION}" in
        r)
            use_records="${OPTARG}" ;;
        I)
            IMG_DIR="${OPTARG}" ;;
        D)
            DATA_DIR="${OPTARG}" ;;
        T)
            TFDS_DIR="${OPTARG}" ;;
        R)
            RECORDS_DIR="${OPTARG}" ;;
        j)
            data_url="${OPTARG}" ;;
        a)
            annotations_url="${OPTARG}" ;;
        m) 
            detections_url="${OPTARG}" ;;
        d)
            dataset="${OPTARG}" ;;
        i)
            download_images="${OPTARG}" ;;
    esac
done

mkdir -p "${DATA_DIR}" "${TFDS_DIR}"

if [ "${use_records}" == "True" ]; then
    mkdir -p "${RECORDS_DIR}"
fi

rm -rf "${DATA_DIR}"* "${TFDS_DIR}"* "${RECORDS_DIR}"*

if [ "${dataset}" == "mpii" ]; then
    wget -q -O "${DATA_DIR}annotations.json" \
    "${annotations_url}"

    wget -q -O "${DATA_DIR}detections.mat" \
    "${detections_url}"

    if [ "${download_images}" == "True" ]; then
        wget -q -O "${IMG_DIR}{dataset}_images.tar.gz" \
        "${data_url}"
        
        tar -xzvf "${IMG_DIR}{dataset}_images.tar.gz" \
        -C "${IMG_DIR}"

        mv "${IMG_DIR}images/" "${IMG_DIR}${dataset}/"
        
        rm -rf "${IMG_DIR}{dataset}_images.tar.gz"
    fi
fi






