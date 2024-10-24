import argparse
from modules.inference_weather import FCN2_weather

def main(input_data, output_folder, fore_hr, device='cpu'):
    FCN2_weather(input_data, output_folder, fore_hr, device=device)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", required=True, help="input_data_file (.nc)")
    parser.add_argument("--output_folder", required=True, help="output_folder (folder)")
    parser.add_argument("--fore_hr", help="how many hours", default=240)
    parser.add_argument("--device", help="cpu or cuda", default='cpu')
    args = parser.parse_args()
    main(args.input_data, args.output_folder, args.fore_hr, device=args.device) 