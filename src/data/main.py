import logging
from preprocess import load_data, clean_data

logging.basicConfig(level=logging.INFO)


def main():
    logging.info("Loading data...")
    data = load_data()

    logging.info("Cleaning data...")
    clean = clean_data(data)

    logging.info(f"Processed data: {clean}")


if __name__ == "__main__":
    main()
