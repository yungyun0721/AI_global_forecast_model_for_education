import argparse
from supporting_module.inference_helper import graphcast_model

def main(input_data, output_folder, fore_hr):
    graphcast_model(input_data, output_folder, fore_hr)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", required=True, help="input_data_file (.nc)")
    parser.add_argument("--output_folder", required=True, help="output_folder (folder)")
    parser.add_argument("--fore_hr", help="how many hours", default=240)
    args = parser.parse_args()
    main(args.input_data, args.output_folder, args.fore_hr) 