import logging
import yaml
from src.data_processing import DataProcessing
from src.model import Model
import src.cleaning_engineering_functions

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Running machine learning pipeline")

    data_processor = DataProcessing()

    logger.info("Data Processing Complete")

    clean_data = data_processor.clean_data()
    logger.info("Clean_data complete")
    modeling = Model(clean_data)
    modeling.build_model_random_forest()
    modeling.build_model_logistic_regression()
    modeling.build_model_KNN()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter