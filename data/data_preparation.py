import os
import csv
import logging
import chardet
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    """
    Load configuration from a YAML file.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def detect_encoding(file_path, sample_size=10000):
    """
    Detect file encoding using a sample of the file.
    """
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read(sample_size))
        return result.get('encoding', 'utf-8')

def convert_to_csv(input_dat, output_csv, header, columns_to_keep=None):
    """
    Convert a .dat file (with "::" separators) into a CSV file.
    """
    encoding_detected = detect_encoding(input_dat)
    logger.info(f"Detected encoding: {encoding_detected}")

    if not os.path.exists(output_csv):
        open(output_csv, 'w').close()

    with open(input_dat, 'r', encoding=encoding_detected) as dat_file, \
         open(output_csv, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        filtered_header = header if columns_to_keep is None else [header[i] for i in columns_to_keep]
        writer.writerow(filtered_header)

        for line_num, line in enumerate(dat_file, start=1):
            try:
                columns = line.strip().split("::")
                if columns_to_keep is not None:
                    columns = [columns[i] for i in columns_to_keep]
                writer.writerow(columns)
            except Exception as e:
                logger.warning(f"Skipped problematic line {line_num}: {e}")

def prepare_dataset():
    """
    Prepare the dataset by converting the .dat file to a simplified CSV.
    """
    input_dat = "../data/movielens_1M/users.dat"
    output_csv = "../data/movielens_1M/users.csv"

    convert_to_csv(
        input_dat=input_dat,
        output_csv=output_csv,
        header=["UserID", "Gender", "Age", "Occupation", "Zip-code"],
        columns_to_keep=[0, 1, 2, 3,4]
    )
    logger.info("Dataset preparation complete.")

if __name__ == '__main__':
    prepare_dataset()
